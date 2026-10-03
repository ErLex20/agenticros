/**
 * Observe → decide → act loop that runs until the goal is reached or a bound
 * (steps, time, full search turn) is hit.
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
  autopilot,
  isGoalReached,
  round,
  planIntent,
  type Action,
  type BBoxNorm,
  type Intent,
  type Observation,
  type PolicyLimits,
  type SearchState,
  type TargetObservation,
} from "./policy.js";
import type { RobotTools } from "./robot.js";

/**
 * Corridor straight ahead (about ±12°, roughly the body width at 1–2 m), above
 * the horizon so the floor in front of the robot is not read as an obstacle.
 * A wider band catches objects beside the path, e.g. pillars flanking the target.
 */
const FRONT_ROI = { x_min: 0.4, y_min: 0.25, x_max: 0.6, y_max: 0.5 };

/** Upper bound on explicit motions parsed from one request. */
const MAX_EXPLICIT_MOVES = 8;

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

type GoalMode = "approach" | "describe" | "move" | "stop";

interface ParsedGoal {
  mode: GoalMode;
  target: TargetSpec | null;
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
    case "finish":
      return `finish(${action.success ? "success" : "failed"})`;
  }
}

function describeIntent(intent: Intent): string {
  switch (intent.kind) {
    case "search":
      return intent.direction ? `search(${intent.direction > 0 ? "left" : "right"})` : "search()";
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
  return `${target}; front clearance ${obs.front_clearance_m != null ? `${obs.front_clearance_m} m` : "unknown"}`;
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

  async parseGoal(prompt: string): Promise<ParsedGoal> {
    const message = await this.chat([
      {
        role: "system",
        content:
          "You translate a user's request for a quadruped robot into a JSON task. Return JSON only with this schema: " +
          '{"mode": "approach"|"describe"|"move"|"stop", "target_description": string|null (English, e.g. "red fire hydrant"), ' +
          '"coco_class": string|null, "attributes": string[], "stop_distance_m": number|null, ' +
          '"search_direction": "left"|"right"|null, "moves": [{"rotate_deg": number} or {"advance_m": number}]}. ' +
          "mode=approach: find an object and go near it or reach it. mode=describe: only look and describe, no motion. " +
          "mode=move: explicit motions only (rotate_deg positive = left; advance_m forward). mode=stop: stop the robot. " +
          `coco_class must be exactly one of these COCO classes when the target is one of them, otherwise null: ${COCO_CLASSES.join(", ")}. ` +
          "attributes lists properties a generic detector cannot verify (colour, material, size), e.g. [\"red\"]. " +
          "stop_distance_m is the requested final distance from the target, or null if unspecified.",
      },
      { role: "user", content: prompt },
    ]);
    const parsed = parseJsonObject(message?.content ?? "") ?? {};
    log(`[Goal] ${JSON.stringify(parsed)}`);

    const mode = (["approach", "describe", "move", "stop"] as const).find((m) => m === parsed["mode"]) ?? "describe";
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
      mode: mode === "approach" && !description ? "describe" : mode,
      target: description ? { description, cocoClass, attributes } : null,
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
    // COCO targets are localized by YOLO; the vision model then only summarizes the scene,
    // so its failure must not abort the run.
    const useYolo = target?.cocoClass != null;
    const [scene, candidates] = await Promise.all([
      useYolo
        ? this.perception.describe(snapshot.image, goal, null).catch((error: unknown) => ({
            summary: `Vision model unavailable: ${error instanceof Error ? error.message : String(error)}`,
            target: null,
          }))
        : this.perception.describe(snapshot.image, goal, target),
      useYolo && target ? this.perception.detectCoco(snapshot.image, target) : Promise.resolve([]),
    ]);

    let observed: TargetObservation | null = useYolo ? { visible: false, source: "yolo" } : scene.target;
    if (useYolo && target) {
      // Highest confidence first; skip instances that do not match the requested attributes.
      for (const candidate of candidates) {
        if (candidate.bbox && (await this.perception.verifyAttributes(snapshot.image, candidate.bbox, target, candidates.length))) {
          observed = candidate;
          break;
        }
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
            "finish(success=true) as soon as goal_reached is true. If searched_deg exceeds 360 without seeing the target, " +
            "finish(success=false). Use rotate/advance only for explicit user instructions or to get around an obstacle. " +
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
    };
    const search: SearchState = { searchedDeg: 0, direction: parsed.searchDirection };
    const history: HistoryEntry[] = [];
    const startedAt = performance.now();

    for (let step = 1; step <= this.cfg.maxSteps; step++) {
      if ((performance.now() - startedAt) / 1000 > this.cfg.maxSeconds) {
        return `Tempo massimo (${this.cfg.maxSeconds} s) esaurito prima di raggiungere l'obiettivo.`;
      }
      const obs = await this.observe(step, goal, parsed.target);
      log(`[Step ${step}] ${summarizeObservation(obs)}`);
      log(`[Scene] ${obs.scene}`);
      if (obs.target?.visible) {
        search.searchedDeg = 0;
        // If the target is lost later, look for it on the side where it was last seen.
        if (Math.abs(obs.target.bearing_deg ?? 0) > 5) search.direction = (obs.target.bearing_deg ?? 0) > 0 ? 1 : -1;
      }

      let proposed: Intent | null = null;
      if (this.cfg.policy === "llm") {
        // A Nebius failure must not abort the task: the deterministic policy takes over for this step.
        proposed = await this.decide(goal, parsed, obs, history, search, limits).catch((error: unknown) => {
          log(`[Decision] Nebius error, using fallback: ${error instanceof Error ? error.message : String(error)}`);
          return null;
        });
      }
      let action: Action | null = null;
      let outcome = "";
      if (proposed) {
        const plan = planIntent(proposed, obs, limits, search);
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
        const plan = planIntent(intent, obs, limits, search);
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
      await this.execute(action);
      if (action.kind === "rotate" && !obs.target?.visible) search.searchedDeg += Math.abs(action.degrees);
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
        const search: SearchState = { searchedDeg: 0, direction: 1 };
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

  async run(goal: string): Promise<string> {
    const parsed = await this.parseGoal(goal);
    switch (parsed.mode) {
      case "stop":
        await this.robot.estop();
        return "Robot fermato.";
      case "describe":
        return this.describeScene(goal);
      case "move":
        return this.runMoves(parsed.moves);
      case "approach":
        return this.approach(goal, parsed);
    }
  }
}
