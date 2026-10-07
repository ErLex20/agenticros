/**
 * Observe → decide → act loop that runs until the goal is reached or a bound
 * (steps, time, exploration budget) is hit. Without a map, a target that is not
 * seen during a full search turn is looked for from a new spot: the robot walks
 * toward the most open direction measured during the turn and searches again.
 *
 * With Nav2 running on a known map, named places (rooms) come from
 * ros2_list_places: "go to the bedroom" navigates there, and an object that
 * belongs to a known place ("go to the bed") is reached by navigating to that
 * place first and then approaching it visually.
 *
 * Nemotron on Nebius interprets the request and chooses every action through
 * tool calls. Deterministic code owns perception geometry, validates each
 * proposed action against the latest observation and falls back to a simple
 * policy when the model returns nothing usable. All robot I/O goes through
 * AgenticROS tools, and the robot is always stopped on exit.
 */

import type OpenAI from "openai";
import { COCO_CLASSES, resolveCocoClassId } from "@agenticros/object-detection";
import { Perception, parseJsonObject, type TargetSpec } from "./perception.js";
import {
  FRONTIER_MAX_EXPLORATIONS,
  autopilot,
  isGoalReached,
  newSearchState,
  recordAction,
  recordObservation,
  round,
  planIntent,
  type Action,
  type BBoxNorm,
  type Intent,
  type Observation,
  type Plan,
  type PolicyLimits,
  type SearchState,
  type TargetObservation,
} from "./policy.js";
import type { NavigationCapabilities, Place, RobotTools } from "./robot.js";

/**
 * Corridor straight ahead (about ±12°, roughly the body width at 1–2 m), above
 * the horizon so the floor in front of the robot is not read as an obstacle.
 * A wider band catches objects beside the path, e.g. pillars flanking the target.
 */
const FRONT_ROI = { x_min: 0.4, y_min: 0.25, x_max: 0.6, y_max: 0.5 };

/** Upper bound on explicit motions parsed from one request. */
const MAX_EXPLICIT_MOVES = 8;

/** Consecutive failed frontier trips before falling back to walking toward open space. */
const MAX_FRONTIER_FAILURES = 2;

/** Inner part of a detection box: mostly target pixels, not background or neighbours at the edges. */
function innerBox(b: BBoxNorm): BBoxNorm {
  const w = b.x_max - b.x_min;
  const h = b.y_max - b.y_min;
  return { x_min: b.x_min + w * 0.25, x_max: b.x_max - w * 0.25, y_min: b.y_min + h * 0.15, y_max: b.y_max - h * 0.15 };
}

export interface AgentConfig {
  reasoningModel: string;
  maxSteps: number;
  maxSeconds: number;
  limits: PolicyLimits;
  /** "llm": Nemotron decides each step; "auto": deterministic policy only (debugging). */
  policy: "llm" | "auto";
}

type GoalMode = "approach" | "describe" | "move" | "stop" | "goto";

interface ParsedGoal {
  mode: GoalMode;
  target: TargetSpec | null;
  /** Known place to navigate to first (mode goto, or where an approach target is). */
  place: string | null;
  stopDistanceM: number | null;
  searchDirection: 1 | -1;
  moves: Action[];
}

interface HistoryEntry {
  step: number;
  observation: string;
  action: string;
  outcome: string;
}

function log(line: string): void {
  process.stderr.write(`${line}\n`);
}

function elapsed(startedAt: number): string {
  return ((performance.now() - startedAt) / 1000).toFixed(2);
}

function describeAction(action: Action): string {
  switch (action.kind) {
    case "rotate":
      return `rotate(${round(action.degrees, 0)} deg)`;
    case "advance":
      return `advance(${round(action.meters)} m)`;
    case "frontier":
      return "nav2_frontier()";
    case "finish":
      return `finish(${action.success ? "success" : "failed"})`;
  }
}

function describeIntent(intent: Intent): string {
  switch (intent.kind) {
    case "search":
      return intent.direction ? `search(${intent.direction > 0 ? "left" : "right"})` : "search()";
    case "explore":
      return "explore()";
    case "face_target":
      return "face_target()";
    case "approach_target":
      return intent.meters != null ? `approach_target(${round(intent.meters)} m)` : "approach_target()";
    default:
      return describeAction(intent);
  }
}

function summarizeObservation(obs: Observation): string {
  const t = obs.target;
  const target = !t
    ? "no target"
    : t.visible
      ? `target visible (${t.source}${t.confidence != null ? ` ${t.confidence}` : ""}) bearing ${t.bearing_deg} deg, ` +
        `distance ${t.distance_m != null ? `${t.distance_m} m` : "unknown"}, height ${t.height_frac}`
      : "target not visible";
  const doorways = obs.doorway_bearings_deg?.length ? `; doorways at ${obs.doorway_bearings_deg.join(", ")} deg` : "";
  return `${target}; front clearance ${obs.front_clearance_m != null ? `${obs.front_clearance_m} m` : "unknown"}${doorways}`;
}

const DECISION_TOOLS: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "search",
      description: "The target is not visible: turn about 45 degrees to look elsewhere. Direction defaults to the side where the target was last seen.",
      parameters: {
        type: "object",
        properties: { direction: { type: "string", enum: ["left", "right"] } },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "explore",
      description:
        "Only after a full search turn (searched_deg >= 360) without seeing the target: look from another spot. The " +
        "controller drives with Nav2 to the nearest unexplored frontier of the live map, or without a map walks a couple " +
        "of meters toward the most open direction measured during the turn (never straight back). Then search again " +
        "from there. Use it to reach other rooms. While exploring is true, keep calling explore.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "face_target",
      description: "Turn in place so the visible target is centred in the camera. The controller computes the angle.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "approach_target",
      description:
        "Move toward the visible target: the controller first centres it if needed, otherwise walks forward, " +
        "never closer than the stop distance and never into obstacles. Optional meters caps the step.",
      parameters: {
        type: "object",
        properties: { meters: { type: "number", minimum: 0.1, maximum: 1.5 } },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "rotate",
      description: "Low-level: rotate in place by the given degrees (positive = left). Prefer search/face_target.",
      parameters: {
        type: "object",
        properties: { degrees: { type: "number", minimum: -180, maximum: 180 } },
        required: ["degrees"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "advance",
      description: "Low-level: walk straight forward by the given meters (clamped for safety). Prefer approach_target.",
      parameters: {
        type: "object",
        properties: { meters: { type: "number", minimum: 0.1, maximum: 1.5 } },
        required: ["meters"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "finish",
      description: "End the task. success=true only when goal_reached is true. The message is shown to the user, in Italian.",
      parameters: {
        type: "object",
        properties: { success: { type: "boolean" }, message: { type: "string" } },
        required: ["success", "message"],
      },
    },
  },
];

export class GoalAgent {
  private readonly perception: Perception;
  /** Navigation stack detected at the start of the run. */
  private navigation: NavigationCapabilities = { navigate: false, explore: false };

  constructor(
    private readonly api: OpenAI,
    private readonly robot: RobotTools,
    perception: Perception,
    private readonly cfg: AgentConfig,
  ) {
    this.perception = perception;
  }

  private async chat(
    messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
    tools?: OpenAI.Chat.Completions.ChatCompletionTool[],
  ): Promise<OpenAI.Chat.Completions.ChatCompletionMessage | undefined> {
    const startedAt = performance.now();
    const response = await this.api.chat.completions.create({
      model: this.cfg.reasoningModel,
      temperature: 0,
      messages,
      ...(tools ? { tools, tool_choice: "auto" as const, parallel_tool_calls: false } : {}),
    });
    log(`[Timing] Nebius reasoning: ${elapsed(startedAt)}s`);
    return response.choices[0]?.message;
  }

  async parseGoal(prompt: string, places: Place[] = []): Promise<ParsedGoal> {
    const placeRules = places.length
      ? "The robot has a map and navigates with Nav2 to these known places: " +
        places.map((p) => `${p.name} (${p.description ?? "no description"})`).join("; ") +
        ". mode=goto: go to one of these places, e.g. 'vai in camera da letto' -> place bedroom; " +
        "target_description stays null unless the user also names an object to reach there. " +
        "For mode=approach set place to the known place where the target is (e.g. bed -> bedroom), " +
        "or null when the target is not tied to a known place. place must be exactly one of the known names or null. "
      : "To go to a room or place, use mode=approach with the object that identifies it as target: bedroom -> bed, " +
        "kitchen -> refrigerator, living room -> couch, bathroom -> toilet, dining room -> dining table. ";
    const message = await this.chat([
      {
        role: "system",
        content:
          "You translate a user's request for a quadruped robot into a JSON task. Return JSON only with this schema: " +
          `{"mode": "approach"|"describe"|"move"|"stop"${places.length ? '|"goto"' : ""}, ` +
          '"target_description": string|null (English, e.g. "red fire hydrant"), ' +
          '"coco_class": string|null, "attributes": string[], "stop_distance_m": number|null, ' +
          '"search_direction": "left"|"right"|null, "moves": [{"rotate_deg": number} or {"advance_m": number}]' +
          `${places.length ? ', "place": string|null' : ""}}. ` +
          "mode=approach: find an object and go near it or reach it. mode=describe: only look and describe, no motion. " +
          placeRules +
          "mode=move: explicit motions only (rotate_deg positive = left; advance_m forward). mode=stop: stop the robot. " +
          `coco_class must be exactly one of these COCO classes when the target is one of them, otherwise null: ${COCO_CLASSES.join(", ")}. ` +
          "attributes lists properties a generic detector cannot verify (colour, material, size), e.g. [\"red\"]. " +
          "stop_distance_m is the requested final distance from the target, or null if unspecified.",
      },
      { role: "user", content: prompt },
    ]);
    const parsed = parseJsonObject(message?.content ?? "") ?? {};
    log(`[Goal] ${JSON.stringify(parsed)}`);

    const placeNames = new Set(places.map((p) => p.name));
    const place = typeof parsed["place"] === "string" && placeNames.has(parsed["place"]) ? parsed["place"] : null;
    const allowedModes = places.length
      ? (["approach", "describe", "move", "stop", "goto"] as const)
      : (["approach", "describe", "move", "stop"] as const);
    const parsedMode = allowedModes.find((m) => m === parsed["mode"]);
    // goto without a valid place falls back to the visual approach (or a description).
    const mode = parsedMode === "goto" && !place ? "approach" : (parsedMode ?? "describe");
    const cocoRaw = typeof parsed["coco_class"] === "string" ? parsed["coco_class"] : null;
    const cocoClass = cocoRaw && resolveCocoClassId(cocoRaw) !== null ? cocoRaw : null;
    const description = typeof parsed["target_description"] === "string" ? parsed["target_description"] : null;
    const attributes = Array.isArray(parsed["attributes"])
      ? parsed["attributes"].filter((a): a is string => typeof a === "string" && a.trim().length > 0)
      : [];
    const stop = Number(parsed["stop_distance_m"]);
    const moves: Action[] = (Array.isArray(parsed["moves"]) ? parsed["moves"] : []).flatMap((m): Action[] => {
      const move = (m ?? {}) as Record<string, unknown>;
      if (Number.isFinite(Number(move["rotate_deg"]))) return [{ kind: "rotate", degrees: Number(move["rotate_deg"]) }];
      if (Number.isFinite(Number(move["advance_m"]))) return [{ kind: "advance", meters: Number(move["advance_m"]) }];
      return [];
    });
    return {
      // "go to the bedroom" sometimes comes back as move without motions: it means reaching the target.
      mode: mode === "approach" && !description ? "describe" : mode === "move" && moves.length === 0 && description ? "approach" : mode,
      target: description ? { description, cocoClass, attributes } : null,
      place,
      stopDistanceM: Number.isFinite(stop) && stop > 0 ? stop : null,
      searchDirection: parsed["search_direction"] === "left" ? 1 : -1,
      moves,
    };
  }

  private async observe(step: number, goal: string, target: TargetSpec | null): Promise<Observation> {
    const startedAt = performance.now();
    const snapshot = await this.robot.snapshot();
    log(`[Timing] AgenticROS snapshot: ${elapsed(startedAt)}s`);

    // A failed reading is "unknown" (null), never "clear": the policy then limits advances.
    const clearanceP = this.robot.depth(FRONT_ROI).catch(() => null);
    // COCO targets are localized by YOLO first; the vision model still reports a coarse column
    // as a fallback (YOLO misses many rendered objects in simulation), so its failure must not
    // abort the run.
    const useYolo = target?.cocoClass != null;
    const [scene, candidates] = await Promise.all([
      useYolo
        ? this.perception.describe(snapshot.image, goal, target).catch((error: unknown) => ({
            summary: `Vision model unavailable: ${error instanceof Error ? error.message : String(error)}`,
            target: null,
            doorwayBearingsDeg: [],
          }))
        : this.perception.describe(snapshot.image, goal, target),
      useYolo && target ? this.perception.detectCoco(snapshot.image, target) : Promise.resolve([]),
    ]);

    let observed: TargetObservation | null = useYolo ? { visible: false, source: "yolo" } : scene.target;
    if (useYolo && target) {
      // Highest confidence first; skip instances that fail the vision-model check
      // (requested attributes, or the class itself for low-confidence detections).
      for (const candidate of candidates) {
        const verified = await this.perception.verifyCandidate(snapshot.image, candidate, target, candidates.length).catch((error: unknown) => {
          log(`[Verify] failed, candidate discarded: ${error instanceof Error ? error.message : String(error)}`);
          return false;
        });
        if (verified) {
          observed = candidate;
          break;
        }
      }
      if (!observed?.visible && scene.target?.visible) {
        log(`[Perception] YOLO missed "${target.description}"; using the vision-model column`);
        observed = scene.target;
      }
    }
    if (observed?.visible && observed.bbox) {
      const distance = await this.robot
        .depth(observed.source === "yolo" ? innerBox(observed.bbox) : observed.bbox)
        .catch(() => null);
      observed.distance_m = distance != null ? round(distance) : null;
    }
    const clearance = await clearanceP;
    log(`[Timing] Perception: ${elapsed(startedAt)}s`);
    return {
      step,
      target: observed,
      front_clearance_m: clearance != null ? round(clearance) : null,
      scene: scene.summary,
      doorway_bearings_deg: scene.doorwayBearingsDeg,
    };
  }

  private async decide(
    goal: string,
    parsed: ParsedGoal,
    obs: Observation,
    history: HistoryEntry[],
    search: SearchState,
    limits: PolicyLimits,
  ): Promise<Intent | null> {
    const message = await this.chat(
      [
        {
          role: "system",
          content:
            "You are the decision module of a quadruped robot controlled through AgenticROS. Each turn you receive the goal, " +
            "the previous steps and the latest observation computed from the camera and depth sensor. Call exactly one tool. " +
            "Use search while the target is not visible, approach_target while it is visible and goal_reached is false, and " +
            "finish(success=true) as soon as goal_reached is true. There is no map: if searched_deg exceeds 360 without " +
            "seeing the target, call explore to look from another spot (another room may hold the target), and keep calling " +
            "explore while exploring is true. finish(success=false) only when explore is rejected. Use rotate/advance only for explicit user instructions or to get around an obstacle. " +
            "The controller validates every call; rejected calls appear in the history with the reason, so do not repeat them. " +
            "Write the finish message in Italian, stating only what was observed (distance from the depth sensor).",
        },
        {
          role: "user",
          content: JSON.stringify({
            goal,
            target: parsed.target?.description ?? null,
            stop_distance_m: limits.stopDistanceM,
            search_direction: search.direction > 0 ? "left" : "right",
            searched_deg: round(search.searchedDeg, 0),
            exploring: search.exploreHeadingDeg != null,
            explorations: `${search.explorations} of ${limits.maxExplorations}`,
            goal_reached: isGoalReached(obs, limits),
            history: history.slice(-8),
            observation: obs,
          }),
        },
      ],
      DECISION_TOOLS,
    );
    const call = message?.tool_calls?.find((c) => c.type === "function");
    if (!call || call.type !== "function") {
      log(`[Decision] no tool call: ${(message?.content ?? "").slice(0, 200)}`);
      return null;
    }
    let args: Record<string, unknown> = {};
    try {
      args = JSON.parse(call.function.arguments || "{}") as Record<string, unknown>;
    } catch {
      return null;
    }
    switch (call.function.name) {
      case "explore":
        return { kind: "explore" };
      case "search":
        return {
          kind: "search",
          ...(args["direction"] === "left" ? { direction: 1 as const } : args["direction"] === "right" ? { direction: -1 as const } : {}),
        };
      case "face_target":
        return { kind: "face_target" };
      case "approach_target":
        return { kind: "approach_target", ...(args["meters"] != null ? { meters: Number(args["meters"]) } : {}) };
      case "rotate":
        return { kind: "rotate", degrees: Number(args["degrees"]) };
      case "advance":
        return { kind: "advance", meters: Number(args["meters"]) };
      case "finish":
        return { kind: "finish", success: args["success"] === true, message: String(args["message"] ?? "") };
      default:
        return null;
    }
  }

  private async execute(action: Action): Promise<void> {
    const startedAt = performance.now();
    if (action.kind === "rotate") await this.robot.rotate(action.degrees);
    else if (action.kind === "advance") await this.robot.advance(action.meters);
    log(`[Timing] AgenticROS ${describeAction(action)}: ${elapsed(startedAt)}s`);
  }

  private async approach(goal: string, parsed: ParsedGoal): Promise<string> {
    const limits: PolicyLimits = {
      ...this.cfg.limits,
      ...(parsed.stopDistanceM != null ? { stopDistanceM: parsed.stopDistanceM } : {}),
      frontierExploration: this.navigation.navigate && this.navigation.explore,
    };
    if (limits.frontierExploration) limits.maxExplorations = Math.max(limits.maxExplorations, FRONTIER_MAX_EXPLORATIONS);
    const search: SearchState = newSearchState(parsed.searchDirection);
    const history: HistoryEntry[] = [];
    let frontierFailures = 0;
    const startedAt = performance.now();

    for (let step = 1; step <= this.cfg.maxSteps; step++) {
      if ((performance.now() - startedAt) / 1000 > this.cfg.maxSeconds) {
        return `Tempo massimo (${this.cfg.maxSeconds} s) esaurito prima di raggiungere l'obiettivo.`;
      }
      const obs = await this.observe(step, goal, parsed.target);
      log(`[Step ${step}] ${summarizeObservation(obs)}`);
      log(`[Scene] ${obs.scene}`);
      recordObservation(search, obs, limits);

      let proposed: Intent | null = null;
      if (this.cfg.policy === "llm") {
        // A Nebius failure must not abort the task: the deterministic policy takes over for this step.
        proposed = await this.decide(goal, parsed, obs, history, search, limits).catch((error: unknown) => {
          log(`[Decision] Nebius error, using fallback: ${error instanceof Error ? error.message : String(error)}`);
          return null;
        });
      }
      let action: Action | null = null;
      let plan: Plan | null = null;
      let outcome = "";
      if (proposed) {
        plan = planIntent(proposed, obs, limits, search);
        if (plan.ok) {
          action = plan.action;
          outcome = plan.note ?? "executed";
        } else {
          log(`[Decision] ${describeIntent(proposed)} rejected: ${plan.reason}`);
          outcome = `rejected: ${plan.reason}; fallback executed`;
        }
      }
      if (!action) {
        const intent = autopilot(obs, limits, search);
        plan = planIntent(intent, obs, limits, search);
        action = plan.ok ? plan.action : { kind: "finish", success: false, message: "Nessuna azione sicura disponibile." };
        outcome ||= "fallback policy";
        log(`[Decision] fallback ${describeIntent(intent)} -> ${describeAction(action)}`);
      } else {
        log(`[Decision] Nemotron ${proposed ? describeIntent(proposed) : ""} -> ${describeAction(action)}${outcome !== "executed" ? ` (${outcome})` : ""}`);
      }

      if (action.kind === "finish") {
        return action.message || (action.success ? "Obiettivo raggiunto." : "Obiettivo non raggiunto.");
      }
      // Perception and reasoning can be slow: never act on a stale observation past the deadline.
      if ((performance.now() - startedAt) / 1000 > this.cfg.maxSeconds) {
        return `Tempo massimo (${this.cfg.maxSeconds} s) esaurito prima di raggiungere l'obiettivo.`;
      }
      if (plan?.ok && plan.explore) {
        const scan = search.scan.map((x) => `${round(x.heading_deg, 0)}:${x.clearance_m}`).join(" ");
        log(
          `[Explore] ${plan.explore.phase} toward heading ${round(plan.explore.headingDeg, 0)} deg ` +
            `(robot ${round(search.headingDeg, 0)} deg, exploration ${search.explorations + 1}/${limits.maxExplorations}` +
            `${scan ? `, scan heading:clearance ${scan}` : ""})`,
        );
      }
      if (action.kind === "frontier") {
        const startedAt = performance.now();
        const frontier = await this.robot.exploreFrontier();
        log(`[Timing] AgenticROS explore(frontier): ${elapsed(startedAt)}s -> ${frontier.ok ? "reached" : "failed"}`);
        frontierFailures = frontier.ok ? 0 : frontierFailures + 1;
        if (frontierFailures >= MAX_FRONTIER_FAILURES) {
          // No reachable frontier left (or Nav2 keeps failing): explore by open directions.
          log(`[Explore] frontier exploration unavailable: ${frontier.text.slice(0, 160)}`);
          limits.frontierExploration = false;
        }
      } else {
        await this.execute(action);
      }
      recordAction(search, obs, action, plan, limits);
      history.push({
        step,
        observation: summarizeObservation(obs),
        action: `${proposed ? describeIntent(proposed) : "none"} -> ${describeAction(action)}`,
        outcome,
      });
    }
    return `Numero massimo di passi (${this.cfg.maxSteps}) raggiunto senza completare l'obiettivo.`;
  }

  private async describeScene(goal: string): Promise<string> {
    const snapshot = await this.robot.snapshot();
    const scene = await this.perception.describe(snapshot.image, goal, null);
    log(`[Scene] ${scene.summary}`);
    const message = await this.chat([
      {
        role: "system",
        content: "Answer the user's request in Italian using only the robot's visual observation. Do not invent details.",
      },
      { role: "user", content: `Richiesta: ${goal}\nOsservazione visiva: ${scene.summary}` },
    ]);
    return message?.content?.trim() || scene.summary;
  }

  private async runMoves(moves: Action[]): Promise<string> {
    const done: string[] = [];
    const startedAt = performance.now();
    const stopped = (reason: string) => `Movimento interrotto: ${reason}. Eseguiti: ${done.join(", ") || "nessuno"}.`;
    if (moves.length > MAX_EXPLICIT_MOVES) {
      return stopped(`troppi movimenti richiesti (${moves.length}, massimo ${MAX_EXPLICIT_MOVES})`);
    }
    for (const move of moves) {
      if ((performance.now() - startedAt) / 1000 > this.cfg.maxSeconds) return stopped("tempo massimo esaurito");
      if (move.kind === "advance") {
        const clearance = await this.robot.depth(FRONT_ROI).catch(() => null);
        const search: SearchState = newSearchState(1);
        const check = planIntent(move, { step: 0, target: null, front_clearance_m: clearance }, this.cfg.limits, search);
        if (!check.ok) return stopped(check.reason);
        await this.execute(check.action);
        done.push(describeAction(check.action));
      } else if (move.kind === "rotate") {
        if (!Number.isFinite(move.degrees) || Math.abs(move.degrees) < 3 || Math.abs(move.degrees) > 360) {
          return stopped(`rotazione non valida (${move.degrees} gradi, ammessi 3–360)`);
        }
        await this.execute(move);
        done.push(describeAction(move));
      }
    }
    return `Movimenti comandati: ${done.join(", ") || "nessuno"}. Le distanze reali possono differire per slittamento.`;
  }

  /**
   * Places the agent can navigate to: only with Nav2 on a known map. With a live
   * SLAM map (frontier exploration running) the robot does not know the house
   * yet, so room coordinates are not used: it explores instead.
   */
  private async navigationPlaces(): Promise<Place[]> {
    const [places, caps] = await Promise.all([
      this.robot.listPlaces().catch(() => []),
      this.robot.navigationCapabilities().catch(() => ({ navigate: false, explore: false })),
    ]);
    this.navigation = caps;
    const mode = !caps.navigate ? "not running" : caps.explore ? "live map (SLAM), frontier exploration" : "known map";
    const usable = caps.navigate && !caps.explore ? places : [];
    log(`[Navigation] Nav2 ${mode}; ${usable.length} known places`);
    return usable;
  }

  /** Nav2 trip to a known place; the result line is logged and returned. */
  private async navigate(place: string, places: Place[]): Promise<{ ok: boolean; message: string }> {
    const startedAt = performance.now();
    log(`[Navigation] going to "${place}" with Nav2`);
    const nav = await this.robot.navigateToPlace(place);
    log(`[Timing] AgenticROS navigate_to_place(${place}): ${elapsed(startedAt)}s -> ${nav.status}`);
    // "Camera da letto (ovest): ..." -> "Camera da letto"
    const label = places.find((p) => p.name === place)?.description?.split(/[(:.]/)[0]?.trim() || place;
    return nav.ok
      ? { ok: true, message: `Arrivato con Nav2: ${label}.` }
      : { ok: false, message: `Navigazione con Nav2 verso ${label} non riuscita (${nav.status}).` };
  }

  async run(goal: string): Promise<string> {
    const places = await this.navigationPlaces();
    const parsed = await this.parseGoal(goal, places);
    switch (parsed.mode) {
      case "stop":
        await this.robot.estop();
        return "Robot fermato.";
      case "describe":
        return this.describeScene(goal);
      case "move":
        return this.runMoves(parsed.moves);
      case "goto": {
        const nav = await this.navigate(parsed.place!, places);
        if (!nav.ok) return nav.message;
        if (parsed.target) return `${nav.message} ${await this.approach(goal, parsed)}`;
        return `${nav.message} ${await this.describeScene(goal)}`;
      }
      case "approach": {
        if (!parsed.place) return this.approach(goal, parsed);
        // The target belongs to a known place: get there with Nav2, then find it visually.
        // A failed trip still leaves the visual search, from wherever the robot is.
        const nav = await this.navigate(parsed.place, places);
        return `${nav.message} ${await this.approach(goal, parsed)}`;
      }
    }
  }
}
