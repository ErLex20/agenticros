/**
 * Pure decision helpers for the Nebius goal loop: camera geometry, the
 * translation of model intents into bounded motions, the goal-reached test and
 * the deterministic fallback policy. Nothing here talks to ROS or to Nebius,
 * so it can be unit-tested offline.
 */

export interface BBoxNorm {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface TargetObservation {
  visible: boolean;
  source: "yolo" | "vlm";
  confidence?: number;
  /** Horizontal angle to the target centre in degrees; positive = left of the optical axis. */
  bearing_deg?: number;
  bbox?: BBoxNorm;
  /** Target bbox height as a fraction of the image height. */
  height_frac?: number;
  distance_m?: number | null;
}

export interface Observation {
  step: number;
  target: TargetObservation | null;
  /** Depth in front of the robot (central band above the horizon); null when unknown. */
  front_clearance_m: number | null;
  scene?: string;
  /** Doorways or passages to other rooms seen by the vision model (deg, positive = left). */
  doorway_bearings_deg?: number[];
}

/** Primitive motions executed through AgenticROS. */
export type Action =
  | { kind: "rotate"; degrees: number }
  | { kind: "advance"; meters: number }
  /** Drive with Nav2 to the nearest unexplored frontier of the live (SLAM) map. */
  | { kind: "frontier" }
  | { kind: "finish"; success: boolean; message: string };

/**
 * What the reasoning model may ask for. Semantic intents leave angle and sign
 * arithmetic to the controller, which small models get wrong.
 */
export type Intent =
  | Action
  | { kind: "search"; direction?: 1 | -1 }
  | { kind: "explore" }
  | { kind: "face_target" }
  | { kind: "approach_target"; meters?: number };

export interface PolicyLimits {
  /** Goal distance between camera and target surface. */
  stopDistanceM: number;
  /** Tolerance above stopDistanceM that still counts as arrived. */
  arriveToleranceM: number;
  /** Minimum free space to keep in front of the robot after an advance. */
  safetyMarginM: number;
  maxAdvanceM: number;
  /** Longest advance when front clearance could not be measured (fail safe, not fail open). */
  unknownClearanceAdvanceM: number;
  /** Bbox height fraction treated as "close" when no depth is available. */
  closeHeightFrac: number;
  /** Bearing above which the target must be centred before advancing. */
  alignToleranceDeg: number;
  /** The target must be within this bearing for the goal to count as reached. */
  arriveBearingDeg: number;
  searchStepDeg: number;
  /** Distance walked toward the most open direction before searching again. */
  exploreStepM: number;
  /** Exploration moves allowed before giving up (each one is followed by a new search turn). */
  maxExplorations: number;
  /** Directions with less front clearance are not worth exploring. */
  exploreMinClearanceM: number;
  /**
   * YOLO confidence trusted at once. Weaker detections (checked by the vision model on a
   * crop) and vision-model sightings must repeat in consecutive frames to count.
   */
  strongConfidence: number;
  /**
   * A live map with frontier exploration is available (Nav2 + SLAM): explore by
   * driving to the next frontier instead of walking toward the most open direction.
   */
  frontierExploration: boolean;
}

export const DEFAULT_LIMITS: PolicyLimits = {
  stopDistanceM: 1.0,
  arriveToleranceM: 0.35,
  safetyMarginM: 0.45,
  maxAdvanceM: 1.5,
  unknownClearanceAdvanceM: 0.3,
  closeHeightFrac: 0.45,
  alignToleranceDeg: 12,
  arriveBearingDeg: 30,
  searchStepDeg: 45,
  exploreStepM: 2.5,
  maxExplorations: 4,
  exploreMinClearanceM: 1.5,
  // The house's white sofa reaches 0.65 as "bed": below 0.8, check and confirm.
  strongConfidence: 0.8,
  frontierExploration: false,
};

/** Frontier trips are short and map-guided: a house needs more of them than blind walks. */
export const FRONTIER_MAX_EXPLORATIONS = 8;

/** Front clearance measured at one heading during a search turn. */
export interface ScanSample {
  heading_deg: number;
  clearance_m: number;
}

export interface SearchState {
  /** Cumulative rotation performed while the target was not visible. */
  searchedDeg: number;
  /** Search direction: +1 left, -1 right. Follows the side where the target was last seen. */
  direction: 1 | -1;
  /**
   * Heading estimated from the executed rotations (deg, positive = left, 0 = start).
   * Dead reckoning only: good enough to turn back to a direction seen a few steps earlier.
   */
  headingDeg: number;
  /** Front clearance per heading during the current search turn. */
  scan: ScanSample[];
  /** Doorway headings seen during the current search turn, one list per frame: the way to other rooms. */
  doorwayFrames: number[][];
  /** Completed exploration moves. */
  explorations: number;
  /** Heading of the exploration move in progress; null when not exploring. */
  exploreHeadingDeg: number | null;
  /** Distance still to walk in the exploration move in progress. */
  exploreRemainingM: number;
  /** Heading of the previous exploration move, so the robot does not walk straight back. */
  lastExploreHeadingDeg: number | null;
  /** Exploration headings found blocked from the current spot. */
  blockedHeadingsDeg: number[];
  /** Consecutive observations with the target visible. */
  sightingStreak: number;
}

export function newSearchState(direction: 1 | -1): SearchState {
  return {
    searchedDeg: 0,
    direction,
    headingDeg: 0,
    scan: [],
    doorwayFrames: [],
    explorations: 0,
    exploreHeadingDeg: null,
    exploreRemainingM: 0,
    lastExploreHeadingDeg: null,
    blockedHeadingsDeg: [],
    sightingStreak: 0,
  };
}

/** Angle wrapped to (-180, 180]. */
export function wrapDeg(deg: number): number {
  const w = ((((deg + 180) % 360) + 360) % 360) - 180;
  return w === -180 ? 180 : w;
}

export function fullTurnSearched(search: SearchState, limits: PolicyLimits): boolean {
  return search.searchedDeg >= 360 + limits.searchStepDeg;
}

/** Doorway reports within this angle of each other are the same doorway. */
const DOORWAY_MATCH_DEG = 20;

/**
 * Doorway headings reported by at least two frames of the search turn, with their
 * vote count. Single reports are too often the vision model guessing.
 */
export function confirmedDoorways(search: SearchState): { heading: number; votes: number }[] {
  const all = search.doorwayFrames.flat();
  return all
    .map((heading) => ({
      heading,
      votes: search.doorwayFrames.filter((frame) => frame.some((h) => Math.abs(wrapDeg(h - heading)) <= DOORWAY_MATCH_DEG)).length,
    }))
    .filter((d) => d.votes >= 2);
}

/**
 * Heading for the next exploration move. Doorways confirmed during the search turn
 * come first (other rooms are where an unseen target can be); otherwise the most
 * open direction, with very long readings capped (beyond a few meters every
 * direction is about as good). The way back to the previous spot is penalized and
 * headings found blocked are skipped; the walk itself is still clamped by the
 * clearance measured when facing the chosen heading.
 */
export function chooseExploreHeading(search: SearchState, limits: PolicyLimits): number | null {
  const back = search.lastExploreHeadingDeg != null ? wrapDeg(search.lastExploreHeadingDeg + 180) : null;
  const candidates: { heading: number; score: number }[] = [
    ...confirmedDoorways(search).map((d) => ({ heading: d.heading, score: 5 + d.votes })),
    ...search.scan
      .filter((sample) => sample.clearance_m >= limits.exploreMinClearanceM)
      .map((sample) => ({ heading: sample.heading_deg, score: Math.min(sample.clearance_m, 5) + sample.clearance_m / 100 })),
  ];
  let best: { heading: number; score: number } | null = null;
  for (const candidate of candidates) {
    if (search.blockedHeadingsDeg.some((h) => Math.abs(wrapDeg(candidate.heading - h)) < limits.searchStepDeg / 2)) continue;
    let score = candidate.score;
    if (back != null && Math.abs(wrapDeg(candidate.heading - back)) <= 60) score -= 4;
    if (!best || score > best.score) best = { heading: candidate.heading, score };
  }
  return best ? best.heading : null;
}

/** Bearing (deg, positive = left) of a normalized image x for a pinhole camera. */
export function bearingFromImageX(xNorm: number, hfovDeg: number): number {
  const halfTan = Math.tan((hfovDeg * Math.PI) / 360);
  const offset = (xNorm - 0.5) * 2; // -1 (left edge) .. +1 (right edge)
  // `0 -` rather than unary minus so the optical axis maps to +0, not -0.
  return 0 - (Math.atan(offset * halfTan) * 180) / Math.PI;
}

export function round(value: number, digits = 2): number {
  const f = 10 ** digits;
  return Math.round(value * f) / f;
}

export function isGoalReached(obs: Observation, limits: PolicyLimits): boolean {
  const t = obs.target;
  if (!t?.visible) return false;
  // Near the image edge depth sampling is unreliable, and "near it" implies facing it.
  if (Math.abs(t.bearing_deg ?? 0) > limits.arriveBearingDeg) return false;
  if (typeof t.distance_m === "number") {
    const near = t.distance_m <= limits.stopDistanceM + limits.arriveToleranceM;
    // A vision-model column band can contain nearer objects than the target:
    // also require the target to look reasonably large.
    return t.source === "vlm" ? near && (t.height_frac ?? 0) >= 0.25 : near;
  }
  return (t.height_frac ?? 0) >= limits.closeHeightFrac;
}

/** Longest forward step that keeps the stop distance to the target and the safety margin to obstacles. */
export function maxSafeAdvance(obs: Observation, limits: PolicyLimits): number {
  let max = limits.maxAdvanceM;
  const t = obs.target;
  if (t?.visible && typeof t.distance_m === "number") {
    max = Math.min(max, t.distance_m - limits.stopDistanceM);
  }
  max = Math.min(
    max,
    obs.front_clearance_m != null ? obs.front_clearance_m - limits.safetyMarginM : limits.unknownClearanceAdvanceM,
  );
  return Math.max(0, max);
}

export type Plan =
  | {
      ok: true;
      action: Action;
      note?: string;
      /**
       * Set for exploration moves: turning to the heading, walking along it, giving it up as
       * blocked, or a Nav2 trip to a frontier of the live map.
       */
      explore?: { phase: "turn" | "walk" | "blocked" | "frontier"; headingDeg: number };
    }
  | { ok: false; reason: string };

/** Turn a model intent into one bounded primitive, or explain why it is not allowed now. */
export function planIntent(intent: Intent, obs: Observation, limits: PolicyLimits, search: SearchState): Plan {
  const t = obs.target?.visible ? obs.target : null;
  const bearing = t?.bearing_deg ?? 0;
  // Searching on is allowed while a sighting is unconfirmed: it may be a similar object.
  const confirmed = t != null && sightingConfirmed(obs, search, limits);
  switch (intent.kind) {
    case "search": {
      if (confirmed) {
        return { ok: false, reason: `the target is already visible at ${round(bearing, 0)} deg; use face_target or approach_target` };
      }
      if (search.exploreHeadingDeg != null) {
        return { ok: false, reason: "an exploration move is in progress; use explore" };
      }
      if (fullTurnSearched(search, limits)) {
        return search.explorations < limits.maxExplorations
          ? { ok: false, reason: `already searched ${round(search.searchedDeg, 0)} deg here without seeing the target; use explore` }
          : { ok: false, reason: `already searched ${round(search.searchedDeg, 0)} deg without seeing the target; finish` };
      }
      const direction = intent.direction ?? search.direction;
      return { ok: true, action: { kind: "rotate", degrees: direction * limits.searchStepDeg } };
    }
    case "explore": {
      if (confirmed) return { ok: false, reason: `the target is visible at ${round(bearing, 0)} deg; use approach_target` };
      const exploring = search.exploreHeadingDeg != null;
      if (!exploring) {
        if (!fullTurnSearched(search, limits)) {
          return { ok: false, reason: `search a full turn here first (searched ${round(search.searchedDeg, 0)} of 360 deg)` };
        }
        if (search.explorations >= limits.maxExplorations) {
          return { ok: false, reason: `explored ${search.explorations} places without seeing the target; finish` };
        }
        if (limits.frontierExploration) {
          return {
            ok: true,
            action: { kind: "frontier" },
            note: "driving with Nav2 to the nearest unexplored frontier of the map",
            explore: { phase: "frontier", headingDeg: search.headingDeg },
          };
        }
      }
      const heading = search.exploreHeadingDeg ?? chooseExploreHeading(search, limits);
      if (heading == null) {
        return { ok: false, reason: `no direction with at least ${limits.exploreMinClearanceM} m of free space to explore; finish` };
      }
      const delta = wrapDeg(heading - search.headingDeg);
      if (Math.abs(delta) > limits.alignToleranceDeg) {
        return {
          ok: true,
          action: { kind: "rotate", degrees: delta },
          note: "turning toward the most open direction",
          explore: { phase: "turn", headingDeg: heading },
        };
      }
      const remaining = exploring ? search.exploreRemainingM : limits.exploreStepM;
      const meters = Math.min(remaining, maxSafeAdvance(obs, limits));
      if (meters < 0.3) {
        // Give the heading up and keep searching from here; the next exploration picks another one.
        return {
          ok: true,
          action: { kind: "rotate", degrees: search.direction * limits.searchStepDeg },
          note: "exploration direction blocked; searching from here",
          explore: { phase: "blocked", headingDeg: heading },
        };
      }
      return {
        ok: true,
        action: { kind: "advance", meters },
        note: "exploring toward the most open direction",
        explore: { phase: "walk", headingDeg: heading },
      };
    }
    case "face_target": {
      if (!t) return { ok: false, reason: "the target is not visible; use search" };
      if (Math.abs(bearing) < 3) return { ok: false, reason: "already facing the target" };
      return { ok: true, action: { kind: "rotate", degrees: bearing } };
    }
    case "approach_target": {
      if (!t) return { ok: false, reason: "the target is not visible; use search" };
      if (Math.abs(bearing) > limits.alignToleranceDeg) {
        return { ok: true, action: { kind: "rotate", degrees: bearing }, note: "aligning with the target before advancing" };
      }
      if (isGoalReached(obs, limits)) return { ok: false, reason: "the target is already within the stop distance; finish" };
      const allowed = maxSafeAdvance(obs, limits);
      if (allowed < 0.1) return { ok: false, reason: `no room to advance safely (allowed ${round(allowed)} m)` };
      // The model's step is a cap, floored at 0.5 m: copying a tiny value from the history
      // (approach_target(0.1 m)) crawled 10 cm per observation. Safety limits still apply.
      const requested = intent.meters != null && Number.isFinite(intent.meters) ? Math.max(0.5, intent.meters) : 1.0;
      return { ok: true, action: { kind: "advance", meters: Math.min(requested, allowed) } };
    }
    case "rotate": {
      if (!Number.isFinite(intent.degrees) || Math.abs(intent.degrees) < 3) {
        return { ok: false, reason: "rotation must be a finite angle of at least 3 degrees" };
      }
      return { ok: true, action: { kind: "rotate", degrees: Math.max(-180, Math.min(180, intent.degrees)) } };
    }
    case "advance": {
      if (!Number.isFinite(intent.meters) || intent.meters < 0.1) {
        return { ok: false, reason: "advance needs a distance of at least 0.1 m" };
      }
      if (t && Math.abs(bearing) > 2 * limits.alignToleranceDeg) {
        return { ok: false, reason: `the target is ${round(bearing, 0)} deg off-axis; use approach_target` };
      }
      const allowed = maxSafeAdvance(obs, limits);
      if (allowed < 0.1) return { ok: false, reason: `no room to advance safely (allowed ${round(allowed)} m)` };
      const meters = Math.min(intent.meters, allowed);
      return {
        ok: true,
        action: { kind: "advance", meters },
        ...(meters < intent.meters ? { note: `advance clamped from ${round(intent.meters)} m to ${round(meters)} m` } : {}),
      };
    }
    case "frontier":
      // Produced by the explore intent, never requested directly.
      return { ok: false, reason: "use explore" };
    case "finish": {
      if (intent.success && !isGoalReached(obs, limits)) {
        return {
          ok: false,
          reason: "goal not reached yet: the target must be visible, roughly centred and within the stop distance",
        };
      }
      if (intent.success && !sightingConfirmed(obs, search, limits)) {
        return { ok: false, reason: "the target was seen with low confidence only once; look again before finishing" };
      }
      return { ok: true, action: intent };
    }
  }
}

/** Deterministic fallback used when the reasoning model gives no usable intent. */
/** Confident YOLO sightings are trusted at once; weaker ones only when seen in two consecutive frames. */
export function sightingConfirmed(obs: Observation, search: SearchState, limits: PolicyLimits): boolean {
  const t = obs.target;
  if (!t?.visible) return false;
  const strong = t.source === "yolo" && (t.confidence ?? 0) >= limits.strongConfidence;
  return strong || search.sightingStreak >= 2;
}

export function autopilot(obs: Observation, limits: PolicyLimits, search: SearchState): Intent {
  if (isGoalReached(obs, limits) && sightingConfirmed(obs, search, limits)) {
    const d = obs.target?.distance_m;
    return {
      kind: "finish",
      success: true,
      message:
        typeof d === "number"
          ? `Obiettivo raggiunto: il bersaglio è davanti al robot a ${round(d)} m (sensore di profondità).`
          : "Obiettivo raggiunto: il bersaglio è davanti al robot e occupa gran parte dell'immagine.",
    };
  }
  const t = obs.target?.visible ? obs.target : null;
  const blockedAhead =
    t != null && Math.abs(t.bearing_deg ?? 0) <= limits.alignToleranceDeg && maxSafeAdvance(obs, limits) < 0.1;
  if (blockedAhead && sightingConfirmed(obs, search, limits)) {
    return { kind: "finish", success: false, message: "Il percorso verso il bersaglio è bloccato." };
  }
  // An unconfirmed sighting right in front of an obstacle is often the obstacle itself
  // (a sofa taken for a bed): keep searching as if nothing had been seen.
  if (!t || blockedAhead) {
    if (search.exploreHeadingDeg != null) return { kind: "explore" };
    if (fullTurnSearched(search, limits)) {
      const canExplore = limits.frontierExploration || chooseExploreHeading(search, limits) != null;
      if (search.explorations < limits.maxExplorations && canExplore) {
        return { kind: "explore" };
      }
      return {
        kind: "finish",
        success: false,
        message:
          search.explorations > 0
            ? `Bersaglio non trovato dopo ${search.explorations} esplorazioni, ognuna con un giro completo.`
            : "Bersaglio non trovato dopo un giro completo e nessuna direzione libera da esplorare.",
      };
    }
    return { kind: "search" };
  }
  return { kind: "approach_target" };
}

/** Update the search state with a new observation, before deciding. */
export function recordObservation(search: SearchState, obs: Observation, limits: PolicyLimits): void {
  const t = obs.target;
  search.sightingStreak = t?.visible ? search.sightingStreak + 1 : 0;
  // Weak sightings (vision model, low-confidence YOLO) are often similar objects, e.g. a
  // sofa for a bed. They count only once repeated, e.g. after turning toward them; until
  // then the search turn and any exploration move are kept, so a false alarm costs one step.
  if (sightingConfirmed(obs, search, limits) && t) {
    // A confirmed sighting restarts the search and cancels any exploration move.
    search.searchedDeg = 0;
    search.scan = [];
    search.doorwayFrames = [];
    search.exploreHeadingDeg = null;
    search.exploreRemainingM = 0;
    // If the target is lost later, look for it on the side where it was last seen.
    if (Math.abs(t.bearing_deg ?? 0) > 5) search.direction = (t.bearing_deg ?? 0) > 0 ? 1 : -1;
    return;
  }
  if (search.exploreHeadingDeg != null) return;
  if (obs.front_clearance_m != null) {
    search.scan.push({ heading_deg: round(search.headingDeg, 1), clearance_m: obs.front_clearance_m });
  }
  const doorways = (obs.doorway_bearings_deg ?? []).map((bearing) => round(wrapDeg(search.headingDeg + bearing), 1));
  if (doorways.length > 0) search.doorwayFrames.push(doorways);
}

/** Update the search state after an action has been executed. */
export function recordAction(search: SearchState, obs: Observation, action: Action, plan: Plan | null, limits: PolicyLimits): void {
  const explore = plan?.ok ? plan.explore : undefined;
  if (action.kind === "frontier") {
    // After a Nav2 trip the dead-reckoned heading is meaningless: restart it, and
    // search a full turn from the new spot.
    search.headingDeg = 0;
    search.exploreHeadingDeg = null;
    endExploration(search, 0);
    search.lastExploreHeadingDeg = null;
    return;
  }
  if (explore && search.exploreHeadingDeg == null) {
    search.exploreHeadingDeg = explore.headingDeg;
    search.exploreRemainingM = limits.exploreStepM;
  }
  if (action.kind === "rotate") {
    search.headingDeg = wrapDeg(search.headingDeg + action.degrees);
    if (!explore && !sightingConfirmed(obs, search, limits)) search.searchedDeg += Math.abs(action.degrees);
  }
  if (explore?.phase === "walk" && action.kind === "advance") {
    search.exploreRemainingM -= action.meters;
    if (search.exploreRemainingM < 0.3) endExploration(search, 0);
  } else if (explore?.phase === "blocked") {
    if (search.exploreRemainingM >= limits.exploreStepM) {
      // Blocked before walking: same spot, so the scan of this turn is still valid.
      // Skip this heading and let the next explore call pick another one.
      search.blockedHeadingsDeg.push(explore.headingDeg);
      search.exploreHeadingDeg = null;
      search.exploreRemainingM = 0;
    } else {
      // Blocked after walking part of the way: search a full turn from the new spot.
      // The search rotation that replaced the walk already counts toward it.
      endExploration(search, action.kind === "rotate" ? Math.abs(action.degrees) : 0);
    }
  }
}

function endExploration(search: SearchState, searchedDeg: number): void {
  search.explorations += 1;
  search.lastExploreHeadingDeg = search.exploreHeadingDeg;
  search.exploreHeadingDeg = null;
  search.exploreRemainingM = 0;
  search.scan = [];
  search.doorwayFrames = [];
  search.blockedHeadingsDeg = [];
  search.searchedDeg = searchedDeg;
}
