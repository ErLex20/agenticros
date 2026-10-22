/**
 * Observe → decide → act loop that runs until the goal is reached or a bound
 * (steps, time, exploration budget) is hit. Without a map, a target that is not
 * seen during a full search turn is looked for from a new spot: the robot walks
 * toward confirmed doorways or the most open direction measured during the turn
 * (with NEBIUS_EXPLORE_CHOICE=vision, the way the vision model picks from the
 * views of the turn) and searches again.
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
import { Perception, parseJsonObject, type PerceptionMode, type TargetSpec, type TurnView } from "./perception.js";
import {
  DEFAULT_LIMITS,
  FRONTIER_MAX_EXPLORATIONS,
  applyImuYaw,
  autopilot,
  bearingFromImageX,
  depthConsistentWithBox,
  enterPlace,
  fullTurnSearched,
  isGoalReached,
  isPhantomSighting,
  nav2ApproachGoal,
  newSearchState,
  recordAction,
  recordObservation,
  round,
  sightingConfirmed,
  planIntent,
  wrapDeg,
  type Action,
  type BBoxNorm,
  type CameraGeometry,
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
/** Half width of the 3D corridor in front of the body checked for low obstacles (Go2 ~0.31 m wide, plus margin). */
const CORRIDOR_HALF_WIDTH_M = 0.25;

/** Upper bound on explicit motions parsed from one request. */
const MAX_EXPLICIT_MOVES = 8;

/** Consecutive failed frontier trips before falling back to walking toward open space. */
const MAX_FRONTIER_FAILURES = 2;

/** A Nav2 trip that moved the robot less than this did not happen (frontier or walk). */
const MIN_NAV2_TRIP_M = 0.3;
/** Nav2 time budget per meter of a walk, plus a fixed allowance for planning and turning. */
const NAV2_WALK_S_PER_M = 20;
const NAV2_WALK_BASE_S = 30;

/** Reactive approach (no pose): stop this much short of the stop distance; the next frame finishes. */
const REACTIVE_APPROACH_MARGIN_M = 0.15;
/** Longest reactive drive between two observations of the target. */
const REACTIVE_MAX_STEP_M = 1.5;
const REACTIVE_MAX_SLICES = 3;
/** A slice that commanded this much but shortened the ranges ahead by less than the next is stuck. */
const STUCK_MIN_COMMANDED_M = 0.4;
const STUCK_MAX_PROGRESS_M = 0.1;
/** Consecutive slices without progress before the drive counts as stuck (one can be a misreading). */
const STUCK_SLICES = 2;
/** Stop clearance of the reactive drive from the base centre (m): default, and floor for close targets. */
const DRIVE_CLEARANCE_M = 0.7;
const DRIVE_MIN_CLEARANCE_M = 0.5;
/** Wait before retrying a blocked drive once (a person crossing usually clears in that time). */
const BLOCKED_WAIT_S = 2;
/** A slice that turned more than this went around something: look again before driving on. */
const REACTIVE_DEVIATION_DEG = 10;

/** Budget of the request parsing call (s): longer than a step decision, without retries. */
const GOAL_TIMEOUT_S = 30;
/** Calls served by the fallback reasoning model after a failure of the primary. */
const FALLBACK_CALLS = 5;

/** Semantic waypoint: walk this far beyond the opening to be through it. */
const WAYPOINT_PASS_M = 1.0;
const WAYPOINT_MAX_WALK_M = 5;
/** Frames in a row without the waypoint (outside a walk) before falling back to current doorways. */
const WAYPOINT_MAX_MISSED = 3;
/** Route waypoints per task, and blocked drives toward a visible target before asking for one. */
const MAX_ROUTE_ATTEMPTS = 2;
const ROUTE_AFTER_BLOCKED = 2;
/** Places kept in the text memory given to the vision model. */
const MAX_PLACE_LOG = 4;

/** Camera ahead of the base (Go2 head, ~0.33 m), for placing a sighting in the map frame. */
const CAMERA_FORWARD_M = 0.3;
/** A close sighting within this distance of the remembered target position is the same target. */
const TARGET_MATCH_M = 1.5;

/** Consecutive observations without a camera frame or a usable vision reply before giving up. */
const MAX_PERCEPTION_FAILURES = 3;

/** Views of a search turn closer than this in heading are the same view; the latest one is kept. */
const SAME_VIEW_DEG = 30;
/** A view within this angle of an explored heading is shown as explored. */
const EXPLORED_VIEW_DEG = 45;

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
  /** "vlm" (default): vision model only; "yolo": local detector for COCO targets plus the vision model. */
  perception: PerceptionMode;
  camera: CameraGeometry;
  /**
   * Also check a 3D corridor in front of the body (ros2_depth_corridor) for obstacles lower than
   * the camera: the band above the horizon missed a bin, a coffee table and a dumbbell, and the
   * robot got stuck against them.
   */
  depthCorridor: boolean;
  /**
   * Where to explore after a full turn without the target (no Nav2 frontiers): "vision" asks
   * the vision model with all the views of the turn; "open" uses confirmed doorways, then the
   * most open direction.
   */
  exploreChoice: "vision" | "open";
  /**
   * With Nav2 running and a pose estimate (ros2_robot_pose), the exploration walks are Nav2
   * goals along the chosen heading instead of timed advances (NEBIUS_NAV2_WALK=off disables it).
   */
  nav2Walk: boolean;
  /**
   * With Nav2 and a pose estimate, a confirmed target farther than NAV2_APPROACH_MIN_M beyond the
   * stop distance is approached with one Nav2 goal; the last meter stays visual
   * (NEBIUS_NAV2_APPROACH=off disables it).
   */
  nav2Approach: boolean;
  /**
   * Named places of this localization session (NEBIUS_SESSION_PLACES=on): saved with the pose
   * estimate ("remember this place as kitchen", and every target reached), usable with a live
   * map too, since the map frame of EKF + LIO-SAM holds for the whole session.
   */
  sessionPlaces: boolean;
  /** Time budget of one step decision of the reasoning model (s); beyond it the fallback policy decides. */
  decisionTimeoutS: number;
  /**
   * Reasoning model used when the primary one fails or times out (NEBIUS_FALLBACK_REASONING_MODEL,
   * default nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B; empty: none, the deterministic policy decides).
   */
  fallbackReasoningModel?: string;
  /** Wait after every motion before observing (s): the window of the uncompensated lidar depth. */
  settleS: number;
  /** false (NEBIUS_NAV2=off, branch mapless): do not look for Nav2 at all. */
  nav2?: boolean;
}

type GoalMode = "approach" | "describe" | "move" | "stop" | "goto" | "remember";

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
    case "backup":
      return `backup(${round(action.meters)} m)`;
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

/** One line for the place memory: the first sentence of a few scene summaries of the turn. */
function summarizeScenes(scenes: string[]): string {
  const firsts = scenes.map((s) => s.split(/(?<=\.)\s/)[0]!.trim()).filter((s) => s.length > 0);
  return firsts.slice(0, 3).join(" ").slice(0, 300) || "no description";
}

function summarizeObservation(obs: Observation): string {
  const t = obs.target;
  const target = !t
    ? "no target"
    : t.visible
      ? `target visible (${t.source}${t.localization === "column" ? " column" : ""}${t.confidence != null ? ` ${t.confidence}` : ""}) bearing ${t.bearing_deg} deg, ` +
        `distance ${t.distance_m != null ? `${t.distance_m} m` : "unknown"}, height ${t.height_frac}`
      : "target not visible";
  const doorways = obs.doorway_bearings_deg?.length ? `; doorways at ${obs.doorway_bearings_deg.join(", ")} deg` : "";
  const occluded = t?.occluded ? " (occluded: depth reads a closer object)" : "";
  return `${target}${occluded}; front clearance ${obs.front_clearance_m != null ? `${obs.front_clearance_m} m` : "unknown"}${doorways}`;
}

const DECISION_TOOLS: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "search",
      description: `The target is not visible: turn about ${DEFAULT_LIMITS.searchStepDeg} degrees to look elsewhere. Direction defaults to the side where the target was last seen.`,
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
        "of meters toward a direction chosen from the search turn (never straight back). Then search again from there. " +
        "Use it to reach other rooms. While exploring is true, keep calling explore.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "face_target",
      description:
        "Turn in place so the visible target is centred in the camera, when it is more than 12 degrees off-centre " +
        "(closer to the centre, use approach_target). The controller computes the angle.",
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
  /** Camera frame of the latest observation (crop checks of vision-model sightings). */
  private lastImage: Buffer | null = null;
  /** IMU heading read right after the latest camera frame (null without an IMU). */
  private imuYawAtObservation: number | null = null;
  /** Seconds of IMU data the MCP server could not integrate so far (message gaps). */
  private imuSkippedS = 0;
  /** Frames of the current search turn with their dead-reckoned headings, for the exploration choice. */
  private turnViews: { turnId: number; views: { headingDeg: number; image: Buffer }[] } = { turnId: -1, views: [] };
  /** Turn for which the vision model was already asked where to explore. */
  private askedTurnId = -1;
  /**
   * Semantic waypoint (no pose): the opening the vision model chose after a search turn, found
   * again in every frame by its description. pendingTurnDeg is the one timed turn toward the
   * chosen view; afterwards only what the camera sees steers the robot.
   */
  private waypoint: { description: string; pendingTurnDeg: number | null; missed: number; explorations: number; route?: boolean } | null = null;
  /** Route waypoints tried this task (the target was in view but not reachable straight). */
  private routeAttempts = 0;
  /** Consecutive reactive drives toward the target that ended blocked or stuck without moving. */
  private blockedTowardTarget = 0;
  /** The waypoint as seen in the latest frame (box, bearing, lidar distance). */
  private lastWaypoint: TargetObservation | null = null;
  /** What the robot saw at each place and which opening it took: text memory for the next choices. */
  private placeLog: string[] = [];
  private turnScenes: { turnId: number; scenes: string[] } = { turnId: -1, scenes: [] };
  /** Navigation stack detected at the start of the run. */
  private navigation: NavigationCapabilities = { navigate: false, explore: false };
  /** Calls left on the fallback reasoning model after a failure of the primary one. */
  private fallbackCalls = 0;
  /** Map position of the target from its last far sighting (with a pose estimate). */
  private targetMap: { x: number; y: number } | null = null;

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
    // A step decision has a short budget and no retry: when the endpoint hangs (Nemotron
    // Lightning timed out for hours on 21 October) the deterministic policy takes over at
    // once instead of after 2 x 60 s. Parsing the request keeps the client defaults.
    const options = tools
      ? { timeout: this.cfg.decisionTimeoutS * 1000, maxRetries: 0 }
      : { timeout: GOAL_TIMEOUT_S * 1000, maxRetries: 0 };
    const request = (model: string) =>
      this.api.chat.completions.create({
        model,
        temperature: 0,
        messages,
        ...(tools ? { tools, tool_choice: "auto" as const, parallel_tool_calls: false } : {}),
        // The fallback Nemotron (Nano) reasons by default: 33 s to parse a request, with thinking
        // off 0.75 s and the same JSON.
        ...(model === this.cfg.fallbackReasoningModel ? { chat_template_kwargs: { enable_thinking: false } } : {}),
      } as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming, options);
    // After a failure of the primary model the fallback (another NVIDIA Nemotron) serves the next
    // few calls, then the primary is tried again.
    const fallback = this.cfg.fallbackReasoningModel;
    const primary = this.fallbackCalls > 0 && fallback ? fallback : this.cfg.reasoningModel;
    if (this.fallbackCalls > 0) this.fallbackCalls -= 1;
    let response;
    try {
      response = await request(primary);
    } catch (error) {
      if (!fallback || primary === fallback) throw error;
      log(`[Decision] ${primary} failed (${error instanceof Error ? error.message : String(error)}): retrying with ${fallback}`);
      this.fallbackCalls = FALLBACK_CALLS;
      response = await request(fallback);
    }
    log(`[Timing] Nebius reasoning: ${elapsed(startedAt)}s (${response.model ?? primary})`);
    return response.choices[0]?.message;
  }

  async parseGoal(prompt: string, places: Place[] = []): Promise<ParsedGoal> {
    const remember = this.cfg.sessionPlaces;
    const rememberRules = remember
      ? "mode=remember: save where the robot is now as a named place, e.g. 'ricorda questo posto come cucina' -> " +
        "place 'cucina' (the name the user gives, lower case). "
      : "";
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
          `{"mode": "approach"|"describe"|"move"|"stop"${places.length ? '|"goto"' : ""}${remember ? '|"remember"' : ""}, ` +
          '"target_description": string|null (English, e.g. "red fire hydrant"), ' +
          '"coco_class": string|null, "attributes": string[], "stop_distance_m": number|null, ' +
          '"search_direction": "left"|"right"|null, "moves": [{"rotate_deg": number} or {"advance_m": number}]' +
          `${places.length || remember ? ', "place": string|null' : ""}}. ` +
          "mode=approach: find an object and go near it or reach it. mode=describe: only look and describe, no motion. " +
          placeRules +
          rememberRules +
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
    const rawPlace = typeof parsed["place"] === "string" ? parsed["place"].trim() : "";
    const allowedModes: readonly GoalMode[] = [
      "approach", "describe", "move", "stop",
      ...(places.length ? (["goto"] as const) : []),
      ...(remember ? (["remember"] as const) : []),
    ];
    const parsedMode = allowedModes.find((m) => m === parsed["mode"]);
    if (parsedMode === "remember" && rawPlace) {
      return { mode: "remember", target: null, place: rawPlace.toLowerCase(), stopDistanceM: null, searchDirection: -1, moves: [] };
    }
    const place = placeNames.has(rawPlace) ? rawPlace : null;
    // goto without a valid place falls back to the visual approach (or a description).
    const mode = parsedMode === "goto" && !place ? "approach" : parsedMode === "remember" ? "describe" : (parsedMode ?? "describe");
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

  /** Latest observation, or null when the vision model, the only localizer here, gave no usable reply. */
  /**
   * Free space ahead: the band above the horizon and, if enabled, the corridor in front of the
   * body, whichever is nearer. A failed reading is "unknown" (null), never "clear".
   */
  private async frontClearance(): Promise<number | null> {
    const [band, corridor] = await Promise.all([
      this.robot.depth(FRONT_ROI).catch(() => null),
      this.cfg.depthCorridor
        ? this.robot
            .depthCorridor({ cameraHeightM: this.cfg.camera.heightM, hfovDeg: this.cfg.camera.hfovDeg, halfWidthM: CORRIDOR_HALF_WIDTH_M })
            .catch(() => null)
        : Promise.resolve(null),
    ]);
    const low = corridor?.distance ?? null;
    if (low == null) return band;
    if (band == null || low < band - 0.3) {
      log(`[Perception] obstacle in the corridor at ${round(low)} m (band above the horizon: ${band != null ? `${round(band)} m` : "unknown"})`);
    }
    return band == null ? low : Math.min(band, low);
  }

  private async observe(step: number, goal: string, target: TargetSpec | null): Promise<Observation | null> {
    const startedAt = performance.now();
    // A camera timeout is a missed observation, not the end of the task (it ended a 57-step
    // run in simulation): no motion this step, and the loop gives up after a few in a row.
    const snapshot = await this.robot.snapshot().catch((error: unknown) => {
      log(`[Perception] camera snapshot failed: ${error instanceof Error ? error.message : String(error)}`);
      return null;
    });
    if (!snapshot) return null;
    this.lastImage = snapshot.image;
    log(`[Timing] AgenticROS snapshot: ${elapsed(startedAt)}s`);
    // The heading the frame was taken at: read with the frame, applied before deciding.
    const imuP = this.robot.imuYaw().catch(() => null);

    // A failed reading is "unknown" (null), never "clear": the policy then limits advances.
    const clearanceP = this.frontClearance();
    // In yolo mode COCO targets are localized by YOLO first; the vision model still localizes
    // them as a fallback (YOLO misses many rendered objects in simulation), so its failure must
    // not abort the run. In vlm mode the vision model is the only localizer.
    const useYolo = this.cfg.perception === "yolo" && target?.cocoClass != null;
    const [scene, candidates] = await Promise.all([
      this.perception.describe(snapshot.image, goal, target, this.waypoint?.description).catch((error: unknown) => {
        log(`[Perception] vision model failed: ${error instanceof Error ? error.message : String(error)}`);
        return null;
      }),
      useYolo && target ? this.perception.detectCoco(snapshot.image, target) : Promise.resolve([]),
    ]);
    // Without YOLO the vision model is the only localizer: no reply means no observation,
    // never "target not visible" (that would count as searched space).
    if (!scene && !useYolo) return null;
    const sceneSummary = scene?.summary ?? "Vision model unavailable.";

    let observed: TargetObservation | null = useYolo ? { visible: false, source: "yolo" } : (scene?.target ?? null);
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
      if (!observed?.visible && scene?.target?.visible) {
        log(`[Perception] YOLO missed "${target.description}"; using the vision-model sighting`);
        observed = scene.target;
      }
    }
    if (observed?.visible && observed.bbox) {
      // A tight box (detector or grounding model): sample its inner part; a column band as is.
      const distance = await this.robot
        .depth(observed.localization === "column" ? observed.bbox : innerBox(observed.bbox))
        .catch(() => null);
      observed.distance_m = distance != null ? round(distance) : null;
      if (distance != null && observed.localization !== "column" && !depthConsistentWithBox(observed.bbox, distance, this.cfg.camera)) {
        log(`[Perception] depth ${round(distance)} m does not match the target box (occluded by a closer object): distance unknown`);
        observed.occluded = true;
        observed.distance_m = null;
      }
    }
    this.lastWaypoint = scene?.waypoint ?? null;
    if (this.lastWaypoint?.visible && this.lastWaypoint.bbox) {
      const d = await this.robot.depth(innerBox(this.lastWaypoint.bbox)).catch(() => null);
      this.lastWaypoint.distance_m = d != null ? round(d) : null;
    }
    const clearance = await clearanceP;
    const imu = await imuP;
    this.imuYawAtObservation = imu?.yawDeg ?? null;
    if (imu && imu.skippedS > this.imuSkippedS + 0.05) {
      log(`[Heading] IMU: ${round(imu.skippedS - this.imuSkippedS)} s of gyroscope data missing (message gaps)`);
    }
    if (imu) this.imuSkippedS = imu.skippedS;
    log(`[Timing] Perception: ${elapsed(startedAt)}s`);
    return {
      step,
      target: observed,
      front_clearance_m: clearance != null ? round(clearance) : null,
      scene: sceneSummary,
      doorway_bearings_deg: scene?.doorwayBearingsDeg ?? [],
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
    else if (action.kind === "backup") await this.robot.backUp(action.meters);
    log(`[Timing] AgenticROS ${describeAction(action)}: ${elapsed(startedAt)}s`);
  }

  private async approach(goal: string, parsed: ParsedGoal): Promise<string> {
    const limits: PolicyLimits = {
      ...this.cfg.limits,
      ...(parsed.stopDistanceM != null ? { stopDistanceM: parsed.stopDistanceM } : {}),
      frontierExploration: this.navigation.navigate && this.navigation.explore,
      // No pose and no gyroscope (reactive drive on the scan): explore from the current view.
      currentViewExplore: !!this.robot.opts.scanTopic && !this.robot.opts.imuTopic,
      reactiveDrive: !!this.robot.opts.scanTopic,
    };
    if (limits.frontierExploration) limits.maxExplorations = Math.max(limits.maxExplorations, FRONTIER_MAX_EXPLORATIONS);
    const search: SearchState = newSearchState(parsed.searchDirection);
    this.targetMap = null;
    this.waypoint = null;
    this.lastWaypoint = null;
    this.placeLog = [];
    this.routeAttempts = 0;
    this.blockedTowardTarget = 0;
    const history: HistoryEntry[] = [];
    let frontierFailures = 0;
    let perceptionFailures = 0;
    const startedAt = performance.now();

    for (let step = 1; step <= this.cfg.maxSteps; step++) {
      if ((performance.now() - startedAt) / 1000 > this.cfg.maxSeconds) {
        return `Tempo massimo (${this.cfg.maxSeconds} s) esaurito prima di raggiungere l'obiettivo.`;
      }
      const obs = await this.observe(step, goal, parsed.target);
      if (!obs) {
        // No motion on a missing observation: look again, give up after a few failures.
        perceptionFailures += 1;
        log(`[Step ${step}] no observation (camera or vision model failure ${perceptionFailures}/${MAX_PERCEPTION_FAILURES})`);
        if (perceptionFailures >= MAX_PERCEPTION_FAILURES) {
          return "Camera o modello di visione non disponibili: il robot è fermo e il compito è interrotto.";
        }
        continue;
      }
      perceptionFailures = 0;
      if (this.imuYawAtObservation != null) {
        const turn = applyImuYaw(search, this.imuYawAtObservation);
        if (turn && Math.abs(turn.measuredDeg - turn.commandedDeg) >= 5) {
          log(`[Heading] IMU: turned ${round(turn.measuredDeg, 0)} deg, commanded ${round(turn.commandedDeg, 0)} deg`);
        }
      }
      if (obs.target?.visible && isPhantomSighting(obs, search, limits)) {
        log(`[Perception] ignoring the sighting at ${obs.target.bearing_deg} deg: it vanished twice when faced from here`);
        obs.target = { visible: false, source: obs.target.source, localization: obs.target.localization ?? "box" };
      }
      await this.checkVisionSighting(obs, parsed.target, search, limits);
      await this.rememberTarget(obs, search, limits);
      log(`[Step ${step}] ${summarizeObservation(obs)}`);
      log(`[Scene] ${obs.scene}`);
      if (this.activeWaypoint()?.route && obs.target?.visible && !isGoalReached(obs, limits)) {
        // Going around: the target seen through the glass would pull the robot back to it.
        log("[Waypoint] target in view, but going through the route opening first");
        obs.target = { visible: false, source: obs.target.source, localization: obs.target.localization ?? "box" };
        if (!fullTurnSearched(search, limits)) search.searchedDeg = 360 + limits.searchStepDeg;
      }
      recordObservation(search, obs, limits);
      await this.suggestExploration(obs, parsed.target, search, limits);
      if (limits.currentViewExplore) this.applyWaypoint(obs, search, limits);

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
        if (!plan.ok && isGoalReached(obs, limits) && !sightingConfirmed(obs, search, limits)) {
          // Within the stop distance but seen in one frame only (vision-model sightings need two):
          // stay and look again instead of giving up (it ended a run 1.2 m from the target).
          log(`[Decision] target within the stop distance, not confirmed yet: looking again without moving`);
          history.push({ step, observation: summarizeObservation(obs), action: "none (look again)", outcome: "waiting for a second sighting" });
          continue;
        }
        action = plan.ok ? plan.action : { kind: "finish", success: false, message: "Nessuna azione sicura disponibile." };
        outcome ||= "fallback policy";
        log(`[Decision] fallback ${describeIntent(intent)} -> ${describeAction(action)}`);
      } else {
        log(`[Decision] Nemotron ${proposed ? describeIntent(proposed) : ""} -> ${describeAction(action)}${outcome !== "executed" ? ` (${outcome})` : ""}`);
      }

      if (
        limits.currentViewExplore && obs.target?.visible && !isGoalReached(obs, limits) && !this.activeWaypoint() &&
        this.routeAttempts < MAX_ROUTE_ATTEMPTS && parsed.target &&
        ((action.kind === "finish" && !action.success) || this.blockedTowardTarget >= ROUTE_AFTER_BLOCKED)
      ) {
        if (await this.chooseRoute(obs, parsed.target, search)) {
          history.push({ step, observation: summarizeObservation(obs), action: "route around the obstacle", outcome: "route waypoint set" });
          continue;
        }
      }
      const wp = this.activeWaypoint();
      if (wp?.pendingTurnDeg != null && !obs.target?.visible && action.kind !== "finish") {
        // The one timed turn toward the opening the vision model chose; the next frames find it.
        const degrees = round(wp.pendingTurnDeg, 1);
        wp.pendingTurnDeg = null;
        if (Math.abs(degrees) >= 5) {
          action = { kind: "rotate", degrees };
          plan = { ok: true, action, note: `turning to "${wp.description}"` };
          log(`[Waypoint] turning ${round(degrees, 0)} deg toward "${wp.description}"`);
        }
      }
      if (action.kind === "finish") {
        const message = action.message || (action.success ? "Obiettivo raggiunto." : "Obiettivo non raggiunto.");
        if (action.success && parsed.target && this.cfg.sessionPlaces && this.navigation.navigate) {
          // Where the robot stands now, facing the target: "torna dal frigorifero" goes back here.
          const name = parsed.target.description.toLowerCase();
          const saved = await this.robot.savePlace(name, `${name}: raggiunto con "${goal}", il robot guardava il bersaglio`);
          log(`[Memory] place "${name}" ${saved.ok ? "saved" : `not saved: ${saved.text.slice(0, 120)}`}`);
        }
        return message;
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
        const before = await this.robot.robotPose().catch(() => null);
        const frontier = await this.robot.exploreFrontier();
        const after = before ? await this.robot.robotPose().catch(() => null) : null;
        // With a pose estimate, a trip counts when the robot moved (not by its duration).
        const movedM = before && after ? Math.hypot(after.x - before.x, after.y - before.y) : null;
        const ok = frontier.ok && (movedM == null || movedM >= MIN_NAV2_TRIP_M);
        log(
          `[Timing] AgenticROS explore(frontier): ${elapsed(startedAt)}s -> ${ok ? "reached" : "failed"}` +
            (movedM != null ? `, moved ${round(movedM)} m` : ""),
        );
        frontierFailures = ok ? 0 : frontierFailures + 1;
        if (frontierFailures >= MAX_FRONTIER_FAILURES) {
          // No reachable frontier left (or Nav2 keeps failing): explore by open directions.
          log(`[Explore] frontier exploration unavailable: ${frontier.text.slice(0, 160)}`);
          limits.frontierExploration = false;
        }
      } else if (action.kind === "advance" && plan?.ok && plan.explore?.phase === "walk" && this.cfg.nav2Walk && this.navigation.navigate) {
        action = await this.walkWithNav2(action);
      } else if (
        action.kind === "advance" && plan?.ok && !plan.explore && !plan.detour && this.cfg.nav2Approach &&
        this.navigation.navigate && sightingConfirmed(obs, search, limits)
      ) {
        action = await this.approachWithNav2(action, obs, limits);
      } else if (action.kind === "advance" && this.robot.opts.scanTopic) {
        action = await this.driveReactive(action, plan, obs, limits);
      } else {
        await this.execute(action);
      }
      recordAction(search, obs, action, plan, limits);
      if (this.cfg.settleS > 0) {
        // Without a pose the lidar depth is not motion-compensated: let its window refill with
        // points taken standing still before the next observation (smeared walls read 0.3-0.6 m).
        await new Promise<void>((resolve) => setTimeout(resolve, this.cfg.settleS * 1000));
      }
      history.push({
        step,
        observation: summarizeObservation(obs),
        action: `${proposed ? describeIntent(proposed) : "none"} -> ${describeAction(action)}`,
        outcome,
      });
    }
    return `Numero massimo di passi (${this.cfg.maxSteps}) raggiunto senza completare l'obiettivo.`;
  }

  /**
   * Forward motion as reactive drive slices (ros2_drive_toward) on the instantaneous obstacle
   * scan, with no pose estimate (branch mapless). Toward a confirmed target the bearing is the
   * target's and the distance is what is left to the stop distance (at most 1.5 m per step);
   * exploration walks and detours drive straight ahead. Slices continue while the drive keeps
   * near the requested bearing; a deviation around an obstacle, a block or an abort ends the step,
   * and the next observation (vision model) aims again. The commanded distance, not a measured
   * one, goes to the search state.
   */
  private async driveReactive(
    action: Extract<Action, { kind: "advance" }>,
    plan: Plan | null,
    obs: Observation,
    limits: PolicyLimits,
  ): Promise<Action> {
    const t = obs.target;
    const toTarget = !!plan?.ok && !plan.explore && !plan.detour && !!t?.visible && (t.localization ?? "box") === "box";
    let bearing = toTarget ? (t!.bearing_deg ?? 0) : 0;
    let remaining = action.meters;
    if (toTarget && typeof t!.distance_m === "number" && !t!.occluded) {
      remaining = Math.max(0.1, Math.min(t!.distance_m - limits.stopDistanceM - REACTIVE_APPROACH_MARGIN_M, REACTIVE_MAX_STEP_M));
    }
    const startedAt = performance.now();
    const speed = this.robot.opts.maxLinear * this.robot.opts.forwardEfficiency;
    // The drive stops 0.7 m from the base centre; a shorter requested distance to the target
    // (camera ~0.33 m ahead of the base) lowers it, never below 0.5 m: closer, the scan drops
    // points as the robot's own body.
    const clearance = toTarget
      ? Math.max(DRIVE_MIN_CLEARANCE_M, Math.min(DRIVE_CLEARANCE_M, limits.stopDistanceM + CAMERA_FORWARD_M - 0.1))
      : DRIVE_CLEARANCE_M;
    let advanced = 0;
    let status = "";
    let waited = false;
    let stalled = 0;
    for (let slice = 0; slice < REACTIVE_MAX_SLICES && remaining > 0.05; slice++) {
      let r = await this.robot.driveToward(bearing, remaining, remaining / speed + 0.6, clearance);
      if (r.status === "blocked" && !waited) {
        // Often a person or a pet crossing: wait once, then try the same slice again.
        waited = true;
        log(`[Drive] blocked (free ahead ${r.frontFreeM} m): waiting ${BLOCKED_WAIT_S} s before trying again`);
        await new Promise<void>((resolve) => setTimeout(resolve, BLOCKED_WAIT_S * 1000));
        advanced += r.advancedM;
        remaining -= r.advancedM;
        bearing -= r.turnedDeg;
        r = await this.robot.driveToward(bearing, remaining, remaining / speed + 0.6, clearance);
      }
      status = r.status;
      log(
        `[Drive] ${toTarget ? "toward the target" : "ahead"}: ${r.status}, ${round(r.advancedM)} m commanded` +
          `${r.progressM != null ? ` (${round(r.progressM)} m from the scan)` : ""}, turned ${round(r.turnedDeg, 0)} deg, ` +
          `heading ${r.headingDeg ?? "none"} deg, free ahead ${r.frontFreeM} m`,
      );
      // A large negative "progress" is the scene ahead changing (something left the view), not a
      // measurement: it neither proves nor rules out motion.
      const noProgress =
        r.advancedM >= STUCK_MIN_COMMANDED_M && r.progressM != null && r.progressM < STUCK_MAX_PROGRESS_M && r.progressM > -0.2;
      stalled = noProgress ? stalled + 1 : 0;
      if (stalled >= STUCK_SLICES) {
        // Commanded forward twice, but the surfaces ahead did not get closer: the body is stuck.
        log(`[Drive] no progress measured on the scan in ${stalled} slices: the robot is stuck, the step ends here`);
        advanced += Math.max(0, r.progressM ?? 0);
        status = "stuck";
        break;
      }
      advanced += r.advancedM;
      remaining -= r.advancedM;
      bearing -= r.turnedDeg;
      if (!r.ok || Math.abs(r.turnedDeg) > REACTIVE_DEVIATION_DEG) break;
    }
    log(`[Timing] AgenticROS drive_toward(${round(action.meters)} m): ${elapsed(startedAt)}s -> ${status}, ${round(advanced)} m`);
    if (toTarget) {
      this.blockedTowardTarget = (status === "blocked" || status === "stuck") && advanced < 0.3 ? this.blockedTowardTarget + 1 : 0;
    }
    return { kind: "advance", meters: round(advanced, 2) };
  }

  /**
   * Exploration walk as a Nav2 goal: the robot already faces the heading chosen by the
   * controller, so the goal is `meters` straight ahead in the map frame. Nav2 plans around
   * obstacles (also the low ones the camera misses) and stops at the goal in closed loop. The
   * distance actually covered, from the pose estimate, replaces the commanded one. Without a
   * pose, or when Nav2 does not move the robot, the timed advance runs as before.
   */
  private async walkWithNav2(action: Extract<Action, { kind: "advance" }>): Promise<Action> {
    const startedAt = performance.now();
    const start = await this.robot.robotPose().catch(() => null);
    if (!start) {
      log("[Navigation] no pose estimate (ros2_robot_pose): timed exploration walk");
      await this.execute(action);
      return action;
    }
    const goal = {
      x: start.x + action.meters * Math.cos(start.yaw),
      y: start.y + action.meters * Math.sin(start.yaw),
      yaw: start.yaw,
      frame: start.frame,
    };
    log(`[Navigation] exploration walk with Nav2: ${round(action.meters)} m to (${round(goal.x)}, ${round(goal.y)}) in ${goal.frame}`);
    const nav = await this.robot.navigateToPose(goal, (NAV2_WALK_BASE_S + NAV2_WALK_S_PER_M * action.meters) * 1000).catch(
      (error: unknown) => ({ ok: false, status: error instanceof Error ? error.message : String(error), text: "" }),
    );
    const end = await this.robot.robotPose().catch(() => null);
    const movedM = end ? round(Math.hypot(end.x - start.x, end.y - start.y), 2) : 0;
    log(`[Timing] AgenticROS navigate_to_pose(walk): ${elapsed(startedAt)}s -> ${nav.status}, moved ${movedM} m`);
    if (movedM < MIN_NAV2_TRIP_M && !nav.ok) {
      log("[Navigation] Nav2 did not move the robot: timed exploration walk instead");
      await this.execute(action);
      return action;
    }
    return { kind: "advance", meters: movedM };
  }

  /**
   * Target position in the map frame, with a pose estimate and Nav2 running. A far sighting
   * stores it. A close sighting that falls there is the target the robot just drove to, not a
   * first sighting at arrival distance (closeFirstSighting, kept for unrelated close objects):
   * after a Nav2 approach the target is close and often off to a side when the robot stops, and
   * the guard turned the robot away from it at 1.0 m. While the target is not visible, the search
   * turns toward the stored position first.
   */
  private async rememberTarget(obs: Observation, search: SearchState, limits: PolicyLimits): Promise<void> {
    if (!this.navigation.navigate) return;
    const pose = await this.robot.robotPose().catch(() => null);
    if (!pose) return;
    const t = obs.target;
    const arriveM = limits.stopDistanceM + limits.arriveToleranceM;
    if (t?.visible) {
      const dir = pose.yaw + ((t.bearing_deg ?? 0) * Math.PI) / 180;
      const d = typeof t.distance_m === "number" ? t.distance_m : null;
      if (d != null && d > arriveM && !t.occluded && (t.localization ?? "box") === "box") {
        this.targetMap = {
          x: pose.x + CAMERA_FORWARD_M * Math.cos(pose.yaw) + d * Math.cos(dir),
          y: pose.y + CAMERA_FORWARD_M * Math.sin(pose.yaw) + d * Math.sin(dir),
        };
        return;
      }
      if (!this.targetMap) return;
      // Close (or unknown distance): where the sighting is, else where the robot is.
      const r = d ?? limits.stopDistanceM;
      const sx = pose.x + CAMERA_FORWARD_M * Math.cos(pose.yaw) + r * Math.cos(dir);
      const sy = pose.y + CAMERA_FORWARD_M * Math.sin(pose.yaw) + r * Math.sin(dir);
      const offM = Math.hypot(sx - this.targetMap.x, sy - this.targetMap.y);
      if (offM <= TARGET_MATCH_M) {
        search.farSighting = { headingDeg: wrapDeg(search.headingDeg + (t.bearing_deg ?? 0)), age: 0 };
        log(`[Memory] close sighting ${round(offM)} m from the remembered target position: approached, not a first sighting`);
      }
    } else if (this.targetMap && search.unsticks === 0) {
      // After a blocked rotation the controller keeps the direction it switched to.
      const rel = wrapDeg((Math.atan2(this.targetMap.y - pose.y, this.targetMap.x - pose.x) - pose.yaw) * (180 / Math.PI));
      search.direction = rel >= 0 ? 1 : -1;
    }
  }

  /**
   * Approach of a confirmed, far target as one Nav2 goal (nav2ApproachGoal): Nav2 plans around
   * what stands in between, so most of the way costs one step instead of one per meter. The
   * covered distance, from the pose estimate, replaces the commanded advance; the next frame
   * checks the target again and the visual approach finishes the last meter. Otherwise (no pose,
   * target near, Nav2 not moving the robot) the timed advance runs as before.
   */
  private async approachWithNav2(action: Extract<Action, { kind: "advance" }>, obs: Observation, limits: PolicyLimits): Promise<Action> {
    const start = await this.robot.robotPose().catch(() => null);
    const goal = start ? nav2ApproachGoal(start, obs, limits) : null;
    if (!start || !goal) {
      await this.execute(action);
      return action;
    }
    const startedAt = performance.now();
    log(
      `[Navigation] approach with Nav2: ${goal.meters} m toward the target (${round(obs.target?.distance_m ?? 0)} m away) ` +
        `to (${round(goal.x)}, ${round(goal.y)}) in ${start.frame}`,
    );
    const nav = await this.robot
      .navigateToPose({ ...goal, frame: start.frame }, (NAV2_WALK_BASE_S + NAV2_WALK_S_PER_M * goal.meters) * 1000)
      .catch((error: unknown) => ({ ok: false, status: error instanceof Error ? error.message : String(error), text: "" }));
    const end = await this.robot.robotPose().catch(() => null);
    const movedM = end ? round(Math.hypot(end.x - start.x, end.y - start.y), 2) : 0;
    log(`[Timing] AgenticROS navigate_to_pose(approach): ${elapsed(startedAt)}s -> ${nav.status}, moved ${movedM} m`);
    if (movedM < MIN_NAV2_TRIP_M && !nav.ok) {
      log("[Navigation] Nav2 did not move the robot: timed advance instead");
      await this.execute(action);
      return action;
    }
    return { kind: "advance", meters: movedM };
  }

  /**
   * Topological exploration without a map. Each place where the robot searches a full turn is
   * a node; the headings walked from it (and the way it came from) are its explored edges.
   * When a turn ends without the target, the vision model sees all the views of the turn and
   * picks the way to go; the controller then turns and walks that way as for any exploration.
   */
  private async suggestExploration(obs: Observation, target: TargetSpec | null, search: SearchState, limits: PolicyLimits): Promise<void> {
    const mapless = !!limits.currentViewExplore;
    if ((this.cfg.exploreChoice !== "vision" && !mapless) || !target || limits.frontierExploration || !this.lastImage) return;
    if (search.exploreHeadingDeg != null) return; // walking: not a view of the turn
    if (this.turnViews.turnId !== search.turnId) this.turnViews = { turnId: search.turnId, views: [] };
    this.turnViews.views.push({ headingDeg: search.headingDeg, image: this.lastImage });
    if (this.turnScenes.turnId !== search.turnId) this.turnScenes = { turnId: search.turnId, scenes: [] };
    if (obs.scene) this.turnScenes.scenes.push(obs.scene);
    if (obs.target?.visible || !fullTurnSearched(search, limits) || this.askedTurnId === search.turnId) return;
    if (search.explorations >= limits.maxExplorations) return;
    if (mapless && this.waypoint) return;
    this.askedTurnId = search.turnId;
    if (mapless) return this.chooseWaypoint(target, search);

    const place = enterPlace(search);
    const views: { headingDeg: number; image: Buffer }[] = [];
    for (const v of [...this.turnViews.views].reverse()) {
      if (!views.some((u) => Math.abs(wrapDeg(u.headingDeg - v.headingDeg)) < SAME_VIEW_DEG)) views.push(v);
    }
    views.sort((a, b) => wrapDeg(b.headingDeg - search.headingDeg) - wrapDeg(a.headingDeg - search.headingDeg));
    const tiles: TurnView[] = views.map((v) => ({
      image: v.image,
      relativeDeg: wrapDeg(v.headingDeg - search.headingDeg),
      explored: place.exploredDeg.some((e) => Math.abs(wrapDeg(v.headingDeg - e)) <= EXPLORED_VIEW_DEG),
    }));
    log(
      `[Graph] place ${(search.placeIndex ?? 0) + 1} of ${search.places.length} at (${round(place.x, 1)}, ${round(place.y, 1)}) m, ` +
        `explored headings [${place.exploredDeg.map((h) => round(h, 0)).join(", ")}] deg`,
    );
    const startedAt = performance.now();
    const choice = await this.perception.chooseExploreView(tiles, target).catch((error: unknown) => {
      log(`[Explore] vision model failed: ${error instanceof Error ? error.message : String(error)}`);
      return null;
    });
    log(`[Timing] Explore choice: ${elapsed(startedAt)}s`);
    if (!choice) {
      log("[Explore] no usable choice from the vision model: confirmed doorways, then the most open direction");
      return;
    }
    const view = views[choice.view]!;
    const heading = round(wrapDeg(view.headingDeg + bearingFromImageX(choice.x, this.cfg.camera.hfovDeg)), 1);
    search.suggestedHeadingDeg = heading;
    log(
      `[Explore] vision model: view ${choice.view} (${round(tiles[choice.view]!.relativeDeg, 0)} deg${tiles[choice.view]!.explored ? ", explored" : ""}), ` +
        `x ${round(choice.x)} -> heading ${round(heading, 0)} deg (robot ${round(search.headingDeg, 0)} deg): ${choice.reason}`,
    );
  }

  /**
   * Route waypoint: the target is in view but the straight way is closed (in simulation the bed
   * seen through a glass partition, the passage 2 m to the side). The vision model names the
   * opening that leads to it; the robot goes through it like an exploration waypoint, with the
   * target hidden from the policy until then. Returns false when the model sees no opening.
   */
  private async chooseRoute(obs: Observation, target: TargetSpec, search: SearchState): Promise<boolean> {
    if (!this.lastImage) return false;
    this.routeAttempts += 1;
    const startedAt = performance.now();
    const route = await this.perception.chooseRouteOpening(this.lastImage, target).catch((error: unknown) => {
      log(`[Waypoint] route choice failed: ${error instanceof Error ? error.message : String(error)}`);
      return null;
    });
    log(`[Timing] Route choice: ${elapsed(startedAt)}s`);
    if (!route) {
      log("[Waypoint] no opening toward the target in view: the path is blocked");
      return false;
    }
    const bearing = round(bearingFromImageX((route.bbox.x_min + route.bbox.x_max) / 2, this.cfg.camera.hfovDeg), 1);
    this.waypoint = { description: route.opening, pendingTurnDeg: bearing, missed: 0, explorations: search.explorations, route: true };
    this.blockedTowardTarget = 0;
    log(`[Waypoint] route to the target through "${route.opening}" at ${bearing} deg: ${route.reason}`);
    void obs;
    return true;
  }

  /** The waypoint set by chooseWaypoint (read through a call: other methods change it). */
  private activeWaypoint() {
    return this.waypoint;
  }

  /**
   * Mapless exploration choice: the views of the turn in one mosaic, with the text memory of the
   * places seen so far; the vision model picks the opening to go through and describes it. The
   * view headings come from timed turns within this one turn only (bounded error); the robot
   * turns once toward the chosen view and then follows the description, not the angle.
   */
  private async chooseWaypoint(target: TargetSpec, search: SearchState): Promise<void> {
    const views: { headingDeg: number; image: Buffer }[] = [];
    for (const v of [...this.turnViews.views].reverse()) {
      if (!views.some((u) => Math.abs(wrapDeg(u.headingDeg - v.headingDeg)) < SAME_VIEW_DEG)) views.push(v);
    }
    views.sort((a, b) => wrapDeg(b.headingDeg - search.headingDeg) - wrapDeg(a.headingDeg - search.headingDeg));
    const tiles: TurnView[] = views.map((v) => ({ image: v.image, relativeDeg: wrapDeg(v.headingDeg - search.headingDeg), explored: false }));
    const here = summarizeScenes(this.turnScenes.scenes);
    const memory = [...this.placeLog.slice(-MAX_PLACE_LOG), `now: ${here}`].join(" | ");
    const startedAt = performance.now();
    const choice = await this.perception.chooseExploreView(tiles, target, memory).catch((error: unknown) => {
      log(`[Explore] vision model failed: ${error instanceof Error ? error.message : String(error)}`);
      return null;
    });
    log(`[Timing] Explore choice: ${elapsed(startedAt)}s`);
    if (!choice) {
      log("[Explore] no usable choice from the vision model: doorways of the current view");
      this.placeLog.push(here);
      return;
    }
    const turnDeg = round(wrapDeg(tiles[choice.view]!.relativeDeg + bearingFromImageX(choice.x, this.cfg.camera.hfovDeg)), 1);
    const description = choice.opening || `the passage the robot chose in view ${choice.view}`;
    this.waypoint = { description, pendingTurnDeg: turnDeg, missed: 0, explorations: search.explorations };
    this.placeLog.push(`${here}; left through: ${description}`);
    log(`[Waypoint] "${description}" in view ${choice.view} (${round(turnDeg, 0)} deg from here): ${choice.reason}`);
  }

  /**
   * While a waypoint is active the policy sees one doorway: the waypoint, when the vision model
   * finds it in this frame. Its lidar distance stretches the walk so that it goes through the
   * opening. The waypoint ends when the exploration walk ends (a new search turn follows), when
   * the target shows up, or after a few frames without it.
   */
  private applyWaypoint(obs: Observation, search: SearchState, limits: PolicyLimits): void {
    const wp = this.waypoint;
    if (!wp) return;
    if (search.explorations > wp.explorations) {
      log(`[Waypoint] went through "${wp.description}"`);
      this.waypoint = null;
      return;
    }
    if (obs.target?.visible) return; // the target leads; the waypoint stays for a false sighting
    const seen = this.lastWaypoint;
    if (seen?.visible) {
      wp.missed = 0;
      obs.doorway_bearings_deg = [seen.bearing_deg ?? 0];
      if (search.exploreHeadingDeg != null && typeof seen.distance_m === "number") {
        search.exploreRemainingM = Math.max(1, Math.min(WAYPOINT_MAX_WALK_M, seen.distance_m + WAYPOINT_PASS_M));
      }
      log(`[Waypoint] in view at ${seen.bearing_deg} deg, ${seen.distance_m ?? "unknown"} m`);
      return;
    }
    obs.doorway_bearings_deg = [];
    if (wp.pendingTurnDeg != null) return;
    wp.missed += 1;
    if (wp.missed >= WAYPOINT_MAX_MISSED && search.exploreHeadingDeg == null) {
      log(`[Waypoint] "${wp.description}" not found in ${wp.missed} frames: doorways of the current view`);
      this.waypoint = null;
    }
  }

  /**
   * Vision-model sightings are checked once more on a zoomed crop when they would become
   * confirmed (second frame in a row) and when they would count as arrival. The full-frame
   * pass mistakes similar objects consistently (a closed folding door for a wardrobe); the
   * crop question rejected that case 3 of 3 times. A rejected sighting counts as not visible.
   */
  private async checkVisionSighting(obs: Observation, target: TargetSpec | null, search: SearchState, limits: PolicyLimits): Promise<void> {
    const t = obs.target;
    if (!t?.visible || t.source !== "vlm" || !target || !this.lastImage) return;
    const confirming = search.sightingStreak + 1 === 2;
    if (!confirming && !isGoalReached(obs, limits)) return;
    const match = await this.perception.verifyCandidate(this.lastImage, t, target, 1, true).catch((error: unknown) => {
      log(`[Verify] failed, sighting kept: ${error instanceof Error ? error.message : String(error)}`);
      return true;
    });
    if (!match) obs.target = { visible: false, source: "vlm", localization: t.localization ?? "box" };
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
        const clearance = await this.frontClearance();
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
    if (this.cfg.nav2 === false) {
      log("[Navigation] Nav2 disabled (NEBIUS_NAV2=off): reactive drive on the obstacle scan, no map, no places");
      return [];
    }
    const [places, caps] = await Promise.all([
      this.robot.listPlaces().catch(() => []),
      this.robot.navigationCapabilities().catch(() => ({ navigate: false, explore: false })),
    ]);
    this.navigation = caps;
    const mode = !caps.navigate ? "not running" : caps.explore ? "live map (SLAM), frontier exploration" : "known map";
    // With a live SLAM map (sim-zed nav:=slam) the room coordinates of a prior map do not hold;
    // places saved in this localization session (sessionPlaces) do.
    const usable = caps.navigate && (!caps.explore || this.cfg.sessionPlaces) ? places : [];
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

  /**
   * Sensors the mapless drive cannot do without, checked before anything moves: with the obstacle
   * scan missing every forward step would end at once (no_scan), and without depth the target
   * distance is unknown. On the robot the usual cause is go2_real.launch.py not running.
   */
  private async preflight(): Promise<string | null> {
    const scan = this.robot.opts.scanTopic;
    if (!scan) return null;
    const missing: string[] = [];
    if (!(await this.robot.topicAlive(scan, "sensor_msgs/msg/LaserScan"))) missing.push(scan);
    const depth = this.robot.opts.depthTopic;
    if (depth && !(await this.robot.topicAlive(depth, "sensor_msgs/msg/Image"))) missing.push(depth);
    if (!missing.length) return null;
    log(`[Preflight] no messages on ${missing.join(", ")}`);
    return `Sensori non disponibili (${missing.join(", ")}): avvia go2_real.launch.py sul robot (o la simulazione) prima dell'agente. Il robot non si è mosso.`;
  }

  async run(goal: string): Promise<string> {
    const notReady = await this.preflight();
    if (notReady) return notReady;
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
      case "remember": {
        const saved = await this.robot.savePlace(parsed.place!, `${parsed.place}: posto salvato con "${goal}"`);
        if (!saved.ok) return `Non riesco a salvare il posto "${parsed.place}": ${saved.text}`;
        const at = (parseJsonObject(saved.text)?.["place"] ?? {}) as { x?: number; y?: number; frame?: string };
        return Number.isFinite(at.x) && Number.isFinite(at.y)
          ? `Posto "${parsed.place}" salvato in (${round(at.x!)}, ${round(at.y!)}) m nel frame ${at.frame ?? "map"}.`
          : `Posto "${parsed.place}" salvato.`;
      }
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
