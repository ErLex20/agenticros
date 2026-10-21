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
  /**
   * "box": the bbox tightly bounds the target (detector, or a vision model that grounds).
   * "column": the bbox is a coarse image column band from the vision model, which may also
   * contain nearer objects. Missing means "box".
   */
  localization?: "box" | "column";
  confidence?: number;
  /** Horizontal angle to the target centre in degrees; positive = left of the optical axis. */
  bearing_deg?: number;
  bbox?: BBoxNorm;
  /** Target bbox height as a fraction of the image height. */
  height_frac?: number;
  distance_m?: number | null;
  /**
   * Set when the depth in the box is not the target's: a closer object hides it (the
   * reading does not match the box geometry, see depthConsistentWithBox). distance_m is null then.
   */
  occluded?: boolean;
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
  /** Walk backwards: only to free the body after a rotation the IMU found blocked. */
  | { kind: "backup"; meters: number }
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
  /**
   * Longest advance toward a visible target whose distance is unknown (depth failed, or it
   * read a closer object): the front corridor does not see obstacles lower than the camera.
   */
  unknownTargetAdvanceM: number;
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
  /**
   * Explore from the current view only (branch mapless, no pose and no gyroscope): headings
   * remembered from a search turn are not reliable once the robot turned back by timed
   * rotations, so explore goes through a doorway the vision model sees now, or ahead when
   * there is room, or turns one search step and looks again.
   */
  currentViewExplore?: boolean;
}

export const DEFAULT_LIMITS: PolicyLimits = {
  stopDistanceM: 1.0,
  arriveToleranceM: 0.35,
  safetyMarginM: 0.45,
  maxAdvanceM: 1.5,
  unknownClearanceAdvanceM: 0.3,
  unknownTargetAdvanceM: 0.5,
  closeHeightFrac: 0.45,
  alignToleranceDeg: 12,
  arriveBearingDeg: 30,
  // The camera sees ~110 deg and the vision model finds targets at the image edges as well as
  // in the centre (87% at 45-55 deg off-axis in simulation): quarter turns cover every direction
  // within 45 deg of some view, and a full search turn takes 5 rotations instead of 9.
  searchStepDeg: 90,
  exploreStepM: 2.5,
  // Sized with the 90 deg search step: ~10 steps per spot fit 7 spots in the 80-step budget.
  maxExplorations: 7,
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

/** A way around an obstacle between the robot and a visible target (see SearchState.detour). */
export interface Detour {
  headingDeg: number;
  remainingM: number;
  side: 1 | -1;
  reason: "stuck" | "blocked";
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
  /**
   * The current sighting streak began already within arrival distance with the box filling the
   * frame: from that close only a fragment shows, and in simulation the vision model and its crop
   * check both took a sofa 0.37 m ahead for a bed. Such a streak is never confirmed by repetition.
   */
  closeFirstSighting: boolean;
  /** Latest sighting farther than the arrival distance: absolute heading and observations since. */
  farSighting: { headingDeg: number; age: number } | null;
  /** Last executed motion, with the target as seen just before it: did the motion happen? */
  lastMotion: { kind: "rotate" | "advance"; amount: number; bearingDeg: number; distanceM: number | null } | null;
  /**
   * Consecutive motions after which a visible target stayed where it was: the body is blocked.
   * In simulation the robot pressed against the edge of a glass folding door for 45 steps,
   * turning and walking toward the bed with no effect while the depth saw free space past it.
   */
  ineffectiveMotions: number;
  /**
   * Way around an obstacle between the robot and a confirmed target: heading to walk along,
   * distance left, side of the turn (+1 left) and why ("stuck": motions without effect;
   * "blocked": no room to advance toward the target).
   */
  detour: Detour | null;
  /** Detours started from this spot, and the side of the latest one. */
  detours: number;
  lastDetourSide: 1 | -1 | null;
  /**
   * IMU heading (integrated gyroscope, deg) at the last correction, and the commanded turn added
   * to headingDeg since then (of which searchSinceImuDeg counted as searched). With an IMU the
   * measured turn replaces the commanded one: in simulation rotations commanded next to low
   * furniture turned 40% of the command (median) and the dead-reckoned heading drifted by 180 deg.
   */
  imuYawDeg: number | null;
  turnSinceImuDeg: number;
  searchSinceImuDeg: number;
  /**
   * The last rotation turned less than half the command (IMU): the body touches something at
   * its side or back. In simulation the robot pivoted around the corner of a low dumbbell rack
   * (turns of 30-47%) until it was wedged against it and nothing freed it, and 6 rotations in a
   * row between a bin and a coffee table did not turn at all. The robot then tries the
   * UNSTICK_STEPS in turn from that spot (unsticks counts them) and the controller picks the
   * search direction; blockedTurnDeg is the commanded turn that failed.
   */
  blockedTurn: boolean;
  blockedTurnDeg: number;
  unsticks: number;
  /** Absolute heading of the latest unconfirmed sighting, to tell whether it vanishes once faced. */
  pendingSightingDeg: number | null;
  /**
   * Headings from the current spot where an unconfirmed sighting vanished when the robot
   * turned to it. Seen twice at the same heading it is a phantom: in simulation a sofa
   * passed for a bed only near the image edge, and the robot turned to it and back for
   * 20 steps.
   */
  vanishedDeg: { headingDeg: number; count: number }[];
  /**
   * Dead-reckoned position (m) in the frame of the start pose (x forward, y left), from the
   * executed advances along headingDeg. No ground truth and no odometry: it drifts, which a
   * graph of places tolerates (it only tells whether the robot is back at a known place).
   */
  xM: number;
  yM: number;
  /** Places where the robot searched a full turn: the nodes of the topological graph. */
  places: PlaceNode[];
  /** Index in places of the place of the current search turn; null before the first one. */
  placeIndex: number | null;
  /** Incremented whenever a new search turn starts (new spot, or the target was seen and lost). */
  turnId: number;
  /**
   * Heading chosen by the vision model for the next exploration from this turn (see
   * chooseExploreHeading); null when not asked yet or not usable.
   */
  suggestedHeadingDeg: number | null;
}

/**
 * A place of the topological graph built while exploring without a map: where the robot
 * searched a full turn, and the directions already walked from it (the edges taken).
 */
export interface PlaceNode {
  x: number;
  y: number;
  /** Headings walked from here, plus the reverse of the walk that led here. */
  exploredDeg: number[];
}

/** A search turn closer than this to a known place is at that place. */
export const PLACE_MATCH_M = 1.5;

/** A sighting within this angle of a recorded heading is the same object. */
const PHANTOM_MATCH_DEG = 20;
/** Box height (fraction of the image) above which a close target fills the frame. */
const CLOSE_FILL_FRAC = 0.6;
/** A close first sighting is the end of an approach if the target was seen farther away this recently... */
const FAR_SIGHTING_MAX_AGE = 3;
/** ...within this angle. */
const FAR_SIGHTING_MATCH_DEG = 25;
/** A turn of at least this much must move a visible target in the image... */
const STUCK_MIN_TURN_DEG = 8;
/** ...by at least this fraction of the turn, or the turn did not happen. */
const STUCK_TURN_FRACTION = 0.35;
/** Motions without effect in a row after which the robot goes around the obstacle. */
const STUCK_MOTIONS = 2;
/** Way around: turn this far, then walk this much (front clearance permitting). */
const DETOUR_TURN_DEG = 40;
const DETOUR_WALK_M = 0.7;
/** Detours from one spot before a blocked path ends the approach. */
const MAX_DETOURS = 3;
/** A rotation of at least this much (deg) that turns less than this fraction is blocked. */
const BLOCKED_TURN_MIN_DEG = 30;
const BLOCKED_TURN_FRACTION = 0.5;
/**
 * Ways out of a blocked rotation, tried in turn from one spot: step forward, away from what the
 * body touches at its side or back and where the depth sees (back instead without room ahead),
 * turn the other way, walk back. Then the robot stops: it is stuck.
 */
const UNSTICK_STEPS = ["forward", "reverse", "forward", "backup", "reverse"] as const;
const UNSTICK_M = 0.3;
const UNSTICK_TURN_DEG = 45;
/** Vanishings at one heading after which sightings there are ignored from this spot. */
const PHANTOM_VANISHINGS = 2;

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
    closeFirstSighting: false,
    farSighting: null,
    lastMotion: null,
    ineffectiveMotions: 0,
    detour: null,
    detours: 0,
    lastDetourSide: null,
    imuYawDeg: null,
    turnSinceImuDeg: 0,
    searchSinceImuDeg: 0,
    blockedTurn: false,
    blockedTurnDeg: 0,
    unsticks: 0,
    pendingSightingDeg: null,
    vanishedDeg: [],
    xM: 0,
    yM: 0,
    places: [],
    placeIndex: null,
    turnId: 0,
    suggestedHeadingDeg: null,
  };
}

/**
 * The place of the current search turn: the known place within PLACE_MATCH_M of the
 * dead-reckoned position, or a new one. A new place remembers the reverse of the walk that
 * led to it, so the way back counts as explored.
 */
export function enterPlace(search: SearchState): PlaceNode {
  let index = search.places.findIndex((p) => Math.hypot(p.x - search.xM, p.y - search.yM) <= PLACE_MATCH_M);
  if (index < 0) {
    const back = search.lastExploreHeadingDeg != null ? [round(wrapDeg(search.lastExploreHeadingDeg + 180), 1)] : [];
    search.places.push({ x: round(search.xM), y: round(search.yM), exploredDeg: back });
    index = search.places.length - 1;
  }
  search.placeIndex = index;
  return search.places[index]!;
}

/**
 * True when a new (not yet confirmed) sighting lies where unconfirmed sightings already
 * vanished twice when faced from this spot; the caller then treats the target as not visible.
 */
export function isPhantomSighting(obs: Observation, search: SearchState, limits: PolicyLimits): boolean {
  const t = obs.target;
  if (!t?.visible || search.sightingStreak > 0 || sightingConfirmed(obs, search, limits)) return false;
  const heading = wrapDeg(search.headingDeg + (t.bearing_deg ?? 0));
  return search.vanishedDeg.some(
    (v) => v.count >= PHANTOM_VANISHINGS && Math.abs(wrapDeg(heading - v.headingDeg)) <= PHANTOM_MATCH_DEG,
  );
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
  const isBlocked = (heading: number) =>
    search.blockedHeadingsDeg.some((h) => Math.abs(wrapDeg(heading - h)) < limits.searchStepDeg / 2);
  // The vision model's choice over the whole turn comes first: in simulation it moved the
  // robot toward the target's room in 58% of the decisions, the most open direction in 38%.
  if (search.suggestedHeadingDeg != null && !isBlocked(search.suggestedHeadingDeg)) return search.suggestedHeadingDeg;
  const back = search.lastExploreHeadingDeg != null ? wrapDeg(search.lastExploreHeadingDeg + 180) : null;
  const candidates: { heading: number; score: number }[] = [
    ...confirmedDoorways(search).map((d) => ({ heading: d.heading, score: 5 + d.votes })),
    ...search.scan
      .filter((sample) => sample.clearance_m >= limits.exploreMinClearanceM)
      .map((sample) => ({ heading: sample.heading_deg, score: Math.min(sample.clearance_m, 5) + sample.clearance_m / 100 })),
  ];
  let best: { heading: number; score: number } | null = null;
  for (const candidate of candidates) {
    if (isBlocked(candidate.heading)) continue;
    let score = candidate.score;
    if (back != null && Math.abs(wrapDeg(candidate.heading - back)) <= 60) score -= 4;
    if (!best || score > best.score) best = { heading: candidate.heading, score };
  }
  return best ? best.heading : null;
}

/** Camera mounting used to tell where a floor-standing object meets the floor in the image. */
export interface CameraGeometry {
  heightM: number;
  hfovDeg: number;
  /** Image width / height. */
  aspect: number;
}

/** Normalized image y where the floor is seen at the given planar depth (level camera). */
export function floorImageY(depthM: number, cam: CameraGeometry): number {
  const tanHalfV = Math.tan((cam.hfovDeg * Math.PI) / 360) / cam.aspect;
  return 0.5 + (0.5 * (cam.heightM / depthM)) / tanHalfV;
}

/** Box bottom this far above the expected floor contact means the depth hit something in front. */
const FLOOR_CONTACT_MARGIN = 0.08;

/**
 * False when a depth reading inside a target box cannot be the target's: the box
 * bottom is below the horizon (a floor-standing object) yet well above the image row
 * where the floor lies at that depth (or the image bottom, when that row is below the
 * frame), so a closer object produced the reading, e.g. a chair in front of a
 * refrigerator. Boxes ending above the horizon (wall-mounted, on a table) are not judged.
 * In simulation: 17 of 20 such readings within 1.5 m caught, no false alarm in 88.
 */
export function depthConsistentWithBox(box: BBoxNorm, depthM: number, cam: CameraGeometry): boolean {
  if (box.y_max <= 0.52) return true;
  return box.y_max >= Math.min(1, floorImageY(depthM, cam)) - FLOOR_CONTACT_MARGIN;
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
    return t.localization === "column" ? near && (t.height_frac ?? 0) >= 0.25 : near;
  }
  return (t.height_frac ?? 0) >= limits.closeHeightFrac;
}

/** Longest forward step that keeps the stop distance to the target and the safety margin to obstacles. */
export function maxSafeAdvance(obs: Observation, limits: PolicyLimits): number {
  let max = limits.maxAdvanceM;
  const t = obs.target;
  if (t?.visible) {
    max = Math.min(max, typeof t.distance_m === "number" ? t.distance_m - limits.stopDistanceM : limits.unknownTargetAdvanceM);
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
      /**
       * Set for the moves going around an obstacle between the robot and the target (see
       * SearchState.detour); "flip" turns to the other side when this one has no room.
       */
      detour?: "turn" | "walk" | "flip";
      /** A move to free the body after a rotation the IMU found blocked (UNSTICK_STEPS). */
      unstick?: true;
    }
  | { ok: false; reason: string };

/** Turn a model intent into one bounded primitive, or explain why it is not allowed now. */
export function planIntent(intent: Intent, obs: Observation, limits: PolicyLimits, search: SearchState): Plan {
  const t = obs.target?.visible ? obs.target : null;
  const bearing = t?.bearing_deg ?? 0;
  if (search.blockedTurn && intent.kind !== "finish") return planUnstick(obs, search, limits);
  if (search.detour && intent.kind === "finish" && !intent.success && planDetour(search.detour, obs, search, limits)) {
    return { ok: false, reason: "going around the obstacle first (detour in progress)" };
  }
  if (search.detour && intent.kind !== "finish") {
    // No room to go around this way: plan the intent as usual (recordAction drops the detour).
    const way = planDetour(search.detour, obs, search, limits);
    if (way) return way;
  }
  // Searching on is allowed while a sighting is unconfirmed: it may be a similar object.
  const confirmed = t != null && sightingConfirmed(obs, search, limits);
  switch (intent.kind) {
    case "search": {
      if (confirmed) {
        return { ok: false, reason: `the target is already visible at ${round(bearing, 0)} deg; use face_target or approach_target` };
      }
      // A first, unconfirmed sighting off to the side: turning toward it gives the second look
      // that confirms or rejects it. Searching away (as the model did in simulation, using
      // search(right) to "turn toward" a target at -46 deg) loses it and costs two steps.
      if (t && search.sightingStreak === 1 && Math.abs(bearing) > limits.alignToleranceDeg) {
        return { ok: false, reason: `the target may be at ${round(bearing, 0)} deg (seen once); use face_target to look at it again` };
      }
      if (search.exploreHeadingDeg != null) {
        return { ok: false, reason: "an exploration move is in progress; use explore" };
      }
      if (fullTurnSearched(search, limits)) {
        return search.explorations < limits.maxExplorations
          ? { ok: false, reason: `already searched ${round(search.searchedDeg, 0)} deg here without seeing the target; use explore` }
          : { ok: false, reason: `already searched ${round(search.searchedDeg, 0)} deg without seeing the target; finish` };
      }
      // After a blocked rotation the controller keeps the direction it switched to.
      const direction = search.unsticks > 0 ? search.direction : intent.direction ?? search.direction;
      return { ok: true, action: { kind: "rotate", degrees: direction * limits.searchStepDeg } };
    }
    case "explore": {
      if (confirmed) return { ok: false, reason: `the target is visible at ${round(bearing, 0)} deg; use approach_target` };
      if (limits.currentViewExplore) return planCurrentViewExplore(obs, search, limits);
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
          note: "turning toward the exploration heading",
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
        note: "walking toward the exploration heading",
        explore: { phase: "walk", headingDeg: heading },
      };
    }
    case "face_target": {
      if (!t) return { ok: false, reason: "the target is not visible; use search" };
      // Within the alignment tolerance approach_target walks straight on: re-centring by a few
      // degrees only costs a step, and a small far target can be missed in the next frame
      // (in simulation a bed at 7.4 m was faced twice by 5 deg, then lost).
      if (Math.abs(bearing) <= limits.alignToleranceDeg) {
        return { ok: false, reason: `already facing the target (${round(bearing, 0)} deg off); use approach_target` };
      }
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
      // The model's step is a cap, floored at 0.5 m (1 m while the target is far): copying a
      // small value from the history (approach_target(0.1 m), or a 0.5 m step clamped earlier)
      // crawled toward the target. Safety limits still apply.
      const far = typeof t.distance_m === "number" && t.distance_m - limits.stopDistanceM >= 1.5;
      const floor = far ? 1.0 : 0.5;
      const requested = intent.meters != null && Number.isFinite(intent.meters) ? Math.max(floor, intent.meters) : 1.0;
      const meters = Math.min(requested, allowed);
      const unknown = typeof t.distance_m !== "number" && meters >= limits.unknownTargetAdvanceM - 1e-9 && requested > meters;
      return {
        ok: true,
        action: { kind: "advance", meters },
        ...(unknown ? { note: `advance limited to ${round(meters)} m: target distance unknown` } : {}),
      };
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
    case "backup":
      // Produced after a rotation the IMU found blocked, never requested directly.
      return { ok: false, reason: "backing up is decided by the controller" };
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

/** Doorways farther off-axis than this are not taken while walking through one. */
const CURRENT_VIEW_DOOR_DEG = 30;

/**
 * Explore from what the camera sees now (currentViewExplore). Each decision uses the current
 * observation only: the doorway nearest to straight ahead (the vision model reports doorways in
 * every frame), else the free space ahead, else one more search step to bring another view.
 */
function planCurrentViewExplore(obs: Observation, search: SearchState, limits: PolicyLimits): Plan {
  const exploring = search.exploreHeadingDeg != null;
  if (!exploring) {
    if (!fullTurnSearched(search, limits)) {
      return { ok: false, reason: `search a full turn here first (searched ${round(search.searchedDeg, 0)} of 360 deg)` };
    }
    if (search.explorations >= limits.maxExplorations) {
      return { ok: false, reason: `explored ${search.explorations} places without seeing the target; finish` };
    }
  }
  const doors = (obs.doorway_bearings_deg ?? []).filter((d) => !exploring || Math.abs(d) <= CURRENT_VIEW_DOOR_DEG);
  const door = doors.length ? doors.reduce((a, b) => (Math.abs(b) < Math.abs(a) ? b : a)) : null;
  if (door != null && Math.abs(door) > limits.alignToleranceDeg) {
    return {
      ok: true,
      action: { kind: "rotate", degrees: round(door, 1) },
      note: `turning to the doorway at ${round(door, 0)} deg`,
      explore: { phase: "turn", headingDeg: round(wrapDeg(search.headingDeg + door), 1) },
    };
  }
  const room = maxSafeAdvance(obs, limits);
  if (door != null || exploring || (obs.front_clearance_m ?? 0) >= limits.exploreMinClearanceM) {
    // Through a doorway in view the reactive drive handles what stands near it (it stops if the
    // passage is really closed); elsewhere the free space measured ahead bounds the walk.
    const meters = door != null
      ? (exploring ? search.exploreRemainingM : limits.exploreStepM)
      : Math.min(exploring ? search.exploreRemainingM : limits.exploreStepM, room);
    if (meters >= 0.3) {
      return {
        ok: true,
        action: { kind: "advance", meters },
        note: door != null ? "walking through the doorway in view" : "walking ahead into the free space in view",
        explore: { phase: "walk", headingDeg: search.headingDeg },
      };
    }
    if (exploring) {
      return {
        ok: true,
        action: { kind: "rotate", degrees: search.direction * limits.searchStepDeg },
        note: "exploration blocked ahead; searching from here",
        explore: { phase: "blocked", headingDeg: search.headingDeg },
      };
    }
  }
  if (search.searchedDeg >= 3 * 360) {
    return { ok: false, reason: "no doorway or free space in any direction; finish" };
  }
  // A plain search step (not an exploration move): the next frame is another view to choose from.
  return { ok: true, action: { kind: "rotate", degrees: search.direction * limits.searchStepDeg }, note: "no opening in view; turning to look for one" };
}

/**
 * Going around the obstacle that blocked the approach: turn to the detour heading, then walk
 * as far as the front clearance allows. Overrides the model's intent until done.
 */
function planDetour(detour: Detour, obs: Observation, search: SearchState, limits: PolicyLimits): Plan | null {
  const side = detour.side > 0 ? "left" : "right";
  const delta = wrapDeg(detour.headingDeg - search.headingDeg);
  if (Math.abs(delta) > 5) {
    return { ok: true, action: { kind: "rotate", degrees: round(delta, 1) }, note: `${detour.reason}: turning ${side} to go around the obstacle`, detour: "turn" };
  }
  const clearance = obs.front_clearance_m != null ? obs.front_clearance_m - limits.safetyMarginM : limits.unknownClearanceAdvanceM;
  const meters = round(Math.min(detour.remainingM, clearance), 2);
  if (meters >= 0.2) {
    return { ok: true, action: { kind: "advance", meters }, note: `${detour.reason}: walking around the obstacle on the ${side}`, detour: "walk" };
  }
  // No room on this side (in simulation a sofa 0.43 m ahead after the turn): try the other
  // side at once, if detours are left; otherwise the plain intent is planned.
  if (detour.remainingM < DETOUR_WALK_M || search.detours >= MAX_DETOURS) return null;
  const other = detour.side > 0 ? "right" : "left";
  return { ok: true, action: { kind: "rotate", degrees: -detour.side * 2 * DETOUR_TURN_DEG }, note: `${detour.reason}: no room on the ${side}, turning ${other}`, detour: "flip" };
}

/** Next way out of a blocked rotation (UNSTICK_STEPS), or the end of the task when none is left. */
function planUnstick(obs: Observation, search: SearchState, limits: PolicyLimits): Plan {
  const step = UNSTICK_STEPS[search.unsticks];
  if (!step) {
    return {
      ok: true,
      action: { kind: "finish", success: false, message: "Il robot è bloccato: non riesce a girare e i tentativi di liberarsi non sono bastati." },
    };
  }
  const n = `${search.unsticks + 1}/${UNSTICK_STEPS.length}`;
  if (step === "reverse") {
    const degrees = (search.blockedTurnDeg >= 0 ? -1 : 1) * UNSTICK_TURN_DEG;
    return { ok: true, action: { kind: "rotate", degrees }, note: `rotation blocked (IMU), unstick ${n}: turning the other way`, unstick: true };
  }
  if (step === "forward" && maxSafeAdvance(obs, limits) >= UNSTICK_M) {
    return { ok: true, action: { kind: "advance", meters: UNSTICK_M }, note: `rotation blocked (IMU), unstick ${n}: stepping forward`, unstick: true };
  }
  return { ok: true, action: { kind: "backup", meters: UNSTICK_M }, note: `rotation blocked (IMU), unstick ${n}: backing up`, unstick: true };
}

function startDetour(search: SearchState, side: 1 | -1, reason: Detour["reason"]): void {
  search.detour = { headingDeg: round(wrapDeg(search.headingDeg + side * DETOUR_TURN_DEG), 1), remainingM: DETOUR_WALK_M, side, reason };
  search.detours += 1;
  search.lastDetourSide = side;
}

/** A visible target roughly ahead with no room to advance toward it. */
function blockedAhead(obs: Observation, limits: PolicyLimits): boolean {
  const t = obs.target?.visible ? obs.target : null;
  return t != null && Math.abs(t.bearing_deg ?? 0) <= limits.alignToleranceDeg && maxSafeAdvance(obs, limits) < 0.1;
}

/** Deterministic fallback used when the reasoning model gives no usable intent. */
/** Confident YOLO sightings are trusted at once; weaker ones only when seen in two consecutive frames. */
/**
 * Nav2 approach: below this many meters left to the stop distance the approach stays visual
 * (timed advances re-checked on every frame); above it one Nav2 trip covers most of the way.
 */
export const NAV2_APPROACH_MIN_M = 1.5;
/** A Nav2 approach stops this much short of the stop distance; the visual approach finishes. */
export const NAV2_APPROACH_MARGIN_M = 0.5;

/** Estimated pose of the robot in the map frame (yaw in rad, counter-clockwise). */
export interface MapPose2D {
  x: number;
  y: number;
  yaw: number;
}

/**
 * Nav2 goal for a confirmed target far ahead: on the line from the robot to the target, short of
 * it by the stop distance plus NAV2_APPROACH_MARGIN_M, facing it. Moving the base along that line
 * shortens the camera-to-target distance by the same amount, so the camera offset cancels out.
 * Null when the approach should stay visual: target not tightly localized (box), distance unknown
 * or occluded, off-axis, or already within NAV2_APPROACH_MIN_M of the stop distance.
 */
export function nav2ApproachGoal(
  pose: MapPose2D,
  obs: Observation,
  limits: PolicyLimits,
): { x: number; y: number; yaw: number; meters: number } | null {
  const t = obs.target;
  if (!t?.visible || (t.localization ?? "box") !== "box" || t.occluded) return null;
  if (typeof t.distance_m !== "number" || !Number.isFinite(t.distance_m)) return null;
  const bearing = t.bearing_deg ?? 0;
  if (Math.abs(bearing) > limits.alignToleranceDeg) return null;
  const meters = t.distance_m - limits.stopDistanceM - NAV2_APPROACH_MARGIN_M;
  if (meters < NAV2_APPROACH_MIN_M - NAV2_APPROACH_MARGIN_M) return null;
  const dir = pose.yaw + (bearing * Math.PI) / 180;
  return {
    x: round(pose.x + meters * Math.cos(dir), 3),
    y: round(pose.y + meters * Math.sin(dir), 3),
    yaw: round(Math.atan2(Math.sin(dir), Math.cos(dir)), 4),
    meters: round(meters, 2),
  };
}

export function sightingConfirmed(obs: Observation, search: SearchState, limits: PolicyLimits): boolean {
  const t = obs.target;
  if (!t?.visible) return false;
  const strong = t.source === "yolo" && (t.confidence ?? 0) >= limits.strongConfidence;
  if (strong) return true;
  return !search.closeFirstSighting && search.sightingStreak >= 2;
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
  const blocked = blockedAhead(obs, limits);
  if (blocked && sightingConfirmed(obs, search, limits)) {
    // recordObservation starts a detour while there are attempts left from this spot.
    if (search.detour && planDetour(search.detour, obs, search, limits)) return { kind: "approach_target" };
    return { kind: "finish", success: false, message: "Il percorso verso il bersaglio è bloccato." };
  }
  // An unconfirmed sighting right in front of an obstacle is often the obstacle itself
  // (a sofa taken for a bed): keep searching as if nothing had been seen.
  if (!t || blocked) {
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

/**
 * Replace the commanded turns since the previous IMU reading with the measured one: heading,
 * and the searched angle for the turns that counted as search. The first reading is the
 * baseline. Returns the commanded and measured turn, for logging.
 */
export function applyImuYaw(search: SearchState, imuYawDeg: number): { commandedDeg: number; measuredDeg: number } | null {
  const previous = search.imuYawDeg;
  const commanded = search.turnSinceImuDeg;
  const counted = search.searchSinceImuDeg;
  search.imuYawDeg = imuYawDeg;
  search.turnSinceImuDeg = 0;
  search.searchSinceImuDeg = 0;
  if (previous == null) return null;
  const measured = imuYawDeg - previous;
  search.blockedTurn = Math.abs(commanded) >= BLOCKED_TURN_MIN_DEG && Math.abs(measured) < BLOCKED_TURN_FRACTION * Math.abs(commanded);
  if (search.blockedTurn) search.blockedTurnDeg = commanded;
  search.headingDeg = round(wrapDeg(search.headingDeg + measured - commanded), 1);
  if (counted > 0 && Math.abs(commanded) > 1) {
    search.searchedDeg = Math.max(0, search.searchedDeg - counted + (Math.abs(measured) * counted) / Math.abs(commanded));
  }
  return { commandedDeg: commanded, measuredDeg: measured };
}

/** Update the search state with a new observation, before deciding. */
export function recordObservation(search: SearchState, obs: Observation, limits: PolicyLimits): void {
  const t = obs.target;
  checkLastMotion(search, t);
  const pending = search.pendingSightingDeg;
  if (pending != null && !t?.visible && Math.abs(wrapDeg(search.headingDeg - pending)) <= PHANTOM_MATCH_DEG) {
    // The robot turned to an unconfirmed sighting and it was gone (or failed the crop check).
    const v = search.vanishedDeg.find((x) => Math.abs(wrapDeg(x.headingDeg - pending)) <= PHANTOM_MATCH_DEG);
    if (v) v.count += 1;
    else search.vanishedDeg.push({ headingDeg: pending, count: 1 });
  }
  search.sightingStreak = t?.visible ? search.sightingStreak + 1 : 0;
  if (search.farSighting) search.farSighting.age += 1;
  if (t?.visible) {
    const heading = wrapDeg(search.headingDeg + (t.bearing_deg ?? 0));
    const arriveM = limits.stopDistanceM + limits.arriveToleranceM;
    if (search.sightingStreak === 1) {
      // An unknown distance counts as close too: the depth often reads a nearer part of a large
      // object (a sofa filling 70% of the frame, "occluded", was then confirmed as a bed at 0.77 m).
      const close = (typeof t.distance_m !== "number" || t.distance_m <= arriveM) && (t.height_frac ?? 0) >= CLOSE_FILL_FRAC;
      const far = search.farSighting;
      const approached = far != null && far.age <= FAR_SIGHTING_MAX_AGE && Math.abs(wrapDeg(heading - far.headingDeg)) <= FAR_SIGHTING_MATCH_DEG;
      search.closeFirstSighting = close && !approached;
    }
    if (typeof t.distance_m === "number" && t.distance_m > arriveM) search.farSighting = { headingDeg: heading, age: 0 };
  } else {
    search.closeFirstSighting = false;
  }
  search.pendingSightingDeg =
    t?.visible && !sightingConfirmed(obs, search, limits) ? round(wrapDeg(search.headingDeg + (t.bearing_deg ?? 0)), 1) : null;
  // Weak sightings (vision model, low-confidence YOLO) are often similar objects, e.g. a
  // sofa for a bed. They count only once repeated, e.g. after turning toward them; until
  // then the search turn and any exploration move are kept, so a false alarm costs one step.
  if (sightingConfirmed(obs, search, limits) && t) {
    // A confirmed sighting restarts the search and cancels any exploration move.
    if (search.searchedDeg > 0 || search.exploreHeadingDeg != null) search.turnId += 1;
    search.searchedDeg = 0;
    search.suggestedHeadingDeg = null;
    search.scan = [];
    search.doorwayFrames = [];
    search.exploreHeadingDeg = null;
    search.exploreRemainingM = 0;
    // If the target is lost later, look for it on the side where it was last seen.
    if (Math.abs(t.bearing_deg ?? 0) > 5) search.direction = (t.bearing_deg ?? 0) > 0 ? 1 : -1;
    if (!search.detour && search.detours < MAX_DETOURS && blockedAhead(obs, limits) && !isGoalReached(obs, limits)) {
      // Something closer than the target is in the way (in simulation the frame of a glass
      // door 0.48 m ahead, the bed 3.2 m beyond it, the free passage 60 deg to the left).
      // Go around it on the target's side first, then on the other side.
      const side = search.lastDetourSide != null ? (-search.lastDetourSide as 1 | -1) : (t.bearing_deg ?? 0) >= 0 ? 1 : -1;
      startDetour(search, side, "blocked");
    }
    return;
  }
  if (search.exploreHeadingDeg != null) return;
  if (obs.front_clearance_m != null) {
    search.scan.push({ heading_deg: round(search.headingDeg, 1), clearance_m: obs.front_clearance_m });
  }
  const doorways = (obs.doorway_bearings_deg ?? []).map((bearing) => round(wrapDeg(search.headingDeg + bearing), 1));
  if (doorways.length > 0) search.doorwayFrames.push(doorways);
}

/** Whether the last motion moved a visible target in the image; two misses in a row start a detour. */
function checkLastMotion(search: SearchState, t: TargetObservation | null): void {
  const m = search.lastMotion;
  search.lastMotion = null;
  if (!m || !t?.visible || typeof t.bearing_deg !== "number") {
    if (!t?.visible) search.ineffectiveMotions = 0;
    return;
  }
  let moved = true;
  if (m.kind === "rotate" && Math.abs(m.amount) >= STUCK_MIN_TURN_DEG) {
    moved = Math.abs(wrapDeg(t.bearing_deg - m.bearingDeg)) >= STUCK_TURN_FRACTION * Math.abs(m.amount);
  } else if (m.kind === "advance" && m.amount >= 0.5 && m.distanceM != null && typeof t.distance_m === "number") {
    moved = Math.abs(t.bearing_deg - m.bearingDeg) >= 3 || t.distance_m < m.distanceM - 0.25 * m.amount;
  }
  search.ineffectiveMotions = moved ? 0 : search.ineffectiveMotions + 1;
  if (search.ineffectiveMotions >= STUCK_MOTIONS && !search.detour && search.detours < MAX_DETOURS) {
    // The obstacle is usually on the target's side (the frame the robot tried to turn past).
    startDetour(search, t.bearing_deg >= 0 ? -1 : 1, "stuck");
    search.ineffectiveMotions = 0;
  }
}

/** Update the search state after an action has been executed. */
export function recordAction(search: SearchState, obs: Observation, action: Action, plan: Plan | null, limits: PolicyLimits): void {
  const explore = plan?.ok ? plan.explore : undefined;
  search.blockedTurn = false; // used for this step's plan; the next IMU reading sets it again
  if (plan?.ok && plan.unstick) {
    // Search the rest of the turn the other way round: the blocked way may stay blocked.
    if (search.unsticks === 0) search.direction = search.direction === 1 ? -1 : 1;
    search.unsticks += 1;
    search.lastMotion = null;
    const a = (search.headingDeg * Math.PI) / 180;
    if (action.kind === "backup") {
      search.xM -= action.meters * Math.cos(a);
      search.yM -= action.meters * Math.sin(a);
    } else if (action.kind === "advance") {
      search.xM += action.meters * Math.cos(a);
      search.yM += action.meters * Math.sin(a);
    } else if (action.kind === "rotate") {
      search.headingDeg = wrapDeg(search.headingDeg + action.degrees);
      search.turnSinceImuDeg += action.degrees;
    }
    return;
  }
  const seen = obs.target?.visible && typeof obs.target.bearing_deg === "number" ? obs.target : null;
  search.lastMotion =
    seen && (action.kind === "rotate" || action.kind === "advance")
      ? {
          kind: action.kind,
          amount: action.kind === "rotate" ? action.degrees : action.meters,
          bearingDeg: seen.bearing_deg!,
          distanceM: typeof seen.distance_m === "number" ? seen.distance_m : null,
        }
      : null;
  if (search.detour) {
    const way = plan?.ok ? plan.detour : undefined;
    if (!way) search.detour = null; // no room to go around: back to the plain intent
    else if (way === "flip") {
      const side = -search.detour.side as 1 | -1;
      search.detour = { ...search.detour, side, headingDeg: round(wrapDeg(search.detour.headingDeg + side * 2 * DETOUR_TURN_DEG), 1) };
      search.detours += 1;
      search.lastDetourSide = side;
    } else if (way === "walk" && action.kind === "advance") {
      search.detour.remainingM -= action.meters;
      if (search.detour.remainingM < 0.2) search.detour = null;
    }
  }
  if (action.kind === "frontier") {
    // After a Nav2 trip the dead-reckoned heading and position are meaningless: restart
    // them (and the graph of places), and search a full turn from the new spot.
    search.headingDeg = 0;
    search.imuYawDeg = null;
    search.xM = 0;
    search.yM = 0;
    search.places = [];
    search.placeIndex = null;
    search.exploreHeadingDeg = null;
    endExploration(search, 0);
    search.lastExploreHeadingDeg = null;
    return;
  }
  if (explore && search.exploreHeadingDeg == null) {
    search.exploreHeadingDeg = explore.headingDeg;
    search.exploreRemainingM = limits.exploreStepM;
    // The edge taken from this place: a later turn here will not be sent the same way again.
    const place = search.placeIndex != null ? search.places[search.placeIndex] : undefined;
    if (place && explore.phase !== "frontier") place.exploredDeg.push(round(wrapDeg(explore.headingDeg), 1));
    search.suggestedHeadingDeg = null;
  }
  if (action.kind === "advance") {
    const a = (search.headingDeg * Math.PI) / 180;
    search.xM += action.meters * Math.cos(a);
    search.yM += action.meters * Math.sin(a);
  }
  if (action.kind === "rotate") {
    search.headingDeg = wrapDeg(search.headingDeg + action.degrees);
    search.turnSinceImuDeg += action.degrees;
    if (!explore && !sightingConfirmed(obs, search, limits)) {
      search.searchedDeg += Math.abs(action.degrees);
      search.searchSinceImuDeg += Math.abs(action.degrees);
    }
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
  search.vanishedDeg = [];
  search.pendingSightingDeg = null;
  search.detour = null;
  search.detours = 0;
  search.lastDetourSide = null;
  search.unsticks = 0;
  search.searchedDeg = searchedDeg;
  search.searchSinceImuDeg = searchedDeg; // the rotation that replaced a blocked walk, if any
  search.suggestedHeadingDeg = null;
  search.placeIndex = null;
  search.turnId += 1;
}
