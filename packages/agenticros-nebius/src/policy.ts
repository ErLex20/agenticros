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
}

/** Primitive motions executed through AgenticROS. */
export type Action =
  | { kind: "rotate"; degrees: number }
  | { kind: "advance"; meters: number }
  | { kind: "finish"; success: boolean; message: string };

/**
 * What the reasoning model may ask for. Semantic intents leave angle and sign
 * arithmetic to the controller, which small models get wrong.
 */
export type Intent =
  | Action
  | { kind: "search"; direction?: 1 | -1 }
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
};

export interface SearchState {
  /** Cumulative rotation performed while the target was not visible. */
  searchedDeg: number;
  /** Search direction: +1 left, -1 right. Follows the side where the target was last seen. */
  direction: 1 | -1;
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

export type Plan = { ok: true; action: Action; note?: string } | { ok: false; reason: string };

/** Turn a model intent into one bounded primitive, or explain why it is not allowed now. */
export function planIntent(intent: Intent, obs: Observation, limits: PolicyLimits, search: SearchState): Plan {
  const t = obs.target?.visible ? obs.target : null;
  const bearing = t?.bearing_deg ?? 0;
  switch (intent.kind) {
    case "search": {
      if (t) {
        return { ok: false, reason: `the target is already visible at ${round(bearing, 0)} deg; use face_target or approach_target` };
      }
      const direction = intent.direction ?? search.direction;
      return { ok: true, action: { kind: "rotate", degrees: direction * limits.searchStepDeg } };
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
      const requested = intent.meters != null && Number.isFinite(intent.meters) && intent.meters >= 0.1 ? intent.meters : 1.0;
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
    case "finish": {
      if (intent.success && !isGoalReached(obs, limits)) {
        return {
          ok: false,
          reason: "goal not reached yet: the target must be visible, roughly centred and within the stop distance",
        };
      }
      return { ok: true, action: intent };
    }
  }
}

/** Deterministic fallback used when the reasoning model gives no usable intent. */
export function autopilot(obs: Observation, limits: PolicyLimits, search: SearchState): Intent {
  if (isGoalReached(obs, limits)) {
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
  if (!obs.target?.visible) {
    if (search.searchedDeg >= 360 + limits.searchStepDeg) {
      return { kind: "finish", success: false, message: "Bersaglio non trovato dopo un giro completo." };
    }
    return { kind: "search" };
  }
  if (Math.abs(obs.target.bearing_deg ?? 0) <= limits.alignToleranceDeg && maxSafeAdvance(obs, limits) < 0.1) {
    return { kind: "finish", success: false, message: "Il percorso verso il bersaglio è bloccato." };
  }
  return { kind: "approach_target" };
}
