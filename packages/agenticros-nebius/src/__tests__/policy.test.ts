import { test } from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_LIMITS,
  autopilot,
  bearingFromImageX,
  isGoalReached,
  maxSafeAdvance,
  planIntent,
  type Observation,
} from "../policy.js";

function obs(partial: Partial<Observation> = {}): Observation {
  return { step: 1, target: null, front_clearance_m: null, ...partial };
}

// Front clearance is measured (5 m) unless a test overrides it.
const visibleAt = (bearing: number, distance: number | null): Observation =>
  obs({
    target: { visible: true, source: "yolo", bearing_deg: bearing, distance_m: distance, height_frac: 0.1 },
    front_clearance_m: 5,
  });

test("bearing is zero at the image centre and positive to the left", () => {
  assert.equal(bearingFromImageX(0.5, 110), 0);
  assert.ok(bearingFromImageX(0.2, 110) > 0);
  assert.ok(bearingFromImageX(0.8, 110) < 0);
  assert.ok(Math.abs(bearingFromImageX(1, 110) + 55) < 1e-9);
  // Sim check: hydrant at x=1101/1920 was ~11 deg to the right of the axis.
  assert.ok(Math.abs(bearingFromImageX(1101 / 1920, 110) + 11.8) < 0.3);
});

test("goal is reached only with a visible target within the stop distance", () => {
  assert.equal(isGoalReached(visibleAt(0, 1.2), DEFAULT_LIMITS), true);
  assert.equal(isGoalReached(visibleAt(0, 2.0), DEFAULT_LIMITS), false);
  assert.equal(isGoalReached(obs({ target: { visible: false, source: "yolo" } }), DEFAULT_LIMITS), false);
  const tallNoDepth = obs({ target: { visible: true, source: "vlm", height_frac: 0.6, distance_m: null } });
  assert.equal(isGoalReached(tallNoDepth, DEFAULT_LIMITS), true);
});

const search = { searchedDeg: 0, direction: -1 as const };

test("advance is clamped by stop distance and front clearance", () => {
  assert.equal(maxSafeAdvance(visibleAt(0, 3), DEFAULT_LIMITS), 1.5);
  assert.equal(maxSafeAdvance(visibleAt(0, 1.8), DEFAULT_LIMITS), 0.8);
  const blocked = { ...visibleAt(0, 5), front_clearance_m: 0.9 };
  assert.ok(Math.abs(maxSafeAdvance(blocked, DEFAULT_LIMITS) - 0.45) < 1e-9);
  // Unknown clearance (depth failure, obstacle closer than the stereo minimum) fails safe.
  const unknown = { ...visibleAt(0, 5), front_clearance_m: null };
  assert.equal(maxSafeAdvance(unknown, DEFAULT_LIMITS), DEFAULT_LIMITS.unknownClearanceAdvanceM);

  const res = planIntent({ kind: "advance", meters: 2 }, visibleAt(0, 1.8), DEFAULT_LIMITS, search);
  assert.ok(res.ok && res.action.kind === "advance" && Math.abs(res.action.meters - 0.8) < 1e-9);
});

test("semantic intents compute angle and sign in the controller", () => {
  assert.deepEqual(planIntent({ kind: "face_target" }, visibleAt(49, 3), DEFAULT_LIMITS, search), {
    ok: true,
    action: { kind: "rotate", degrees: 49 },
  });
  const align = planIntent({ kind: "approach_target" }, visibleAt(-30, 4), DEFAULT_LIMITS, search);
  assert.ok(align.ok && align.action.kind === "rotate" && align.action.degrees === -30);
  const walk = planIntent({ kind: "approach_target" }, visibleAt(5, 4), DEFAULT_LIMITS, search);
  assert.ok(walk.ok && walk.action.kind === "advance" && walk.action.meters === 1);
  const hidden = obs({ target: { visible: false, source: "yolo" } });
  assert.deepEqual(planIntent({ kind: "search" }, hidden, DEFAULT_LIMITS, { searchedDeg: 0, direction: 1 }), {
    ok: true,
    action: { kind: "rotate", degrees: 45 },
  });
  assert.equal(planIntent({ kind: "search" }, visibleAt(10, 3), DEFAULT_LIMITS, search).ok, false);
  assert.equal(planIntent({ kind: "approach_target" }, hidden, DEFAULT_LIMITS, search).ok, false);
});

test("off-axis advances and premature success are rejected", () => {
  assert.equal(planIntent({ kind: "advance", meters: 1 }, visibleAt(40, 4), DEFAULT_LIMITS, search).ok, false);
  assert.equal(planIntent({ kind: "finish", success: true, message: "" }, visibleAt(0, 3), DEFAULT_LIMITS, search).ok, false);
  assert.equal(planIntent({ kind: "finish", success: false, message: "" }, visibleAt(0, 3), DEFAULT_LIMITS, search).ok, true);
  assert.equal(planIntent({ kind: "rotate", degrees: 0 }, visibleAt(0, 3), DEFAULT_LIMITS, search).ok, false);
  // A near reading at the image edge is not "arrived".
  assert.equal(isGoalReached(visibleAt(-50, 1.1), DEFAULT_LIMITS), false);
  // A vision-model column band may contain a nearer object: a small target is not "arrived".
  const vlmSmall = obs({ target: { visible: true, source: "vlm", bearing_deg: 0, distance_m: 1.1, height_frac: 0.12 } });
  assert.equal(isGoalReached(vlmSmall, DEFAULT_LIMITS), false);
  assert.equal(planIntent({ kind: "advance", meters: 0.01 }, visibleAt(0, 3), DEFAULT_LIMITS, search).ok, false);
});

test("autopilot searches, approaches and stops", () => {
  const hidden = obs({ target: { visible: false, source: "yolo" } });
  assert.deepEqual(autopilot(hidden, DEFAULT_LIMITS, search), { kind: "search" });
  assert.equal(autopilot(hidden, DEFAULT_LIMITS, { searchedDeg: 420, direction: -1 }).kind, "finish");
  assert.deepEqual(autopilot(visibleAt(30, 4), DEFAULT_LIMITS, search), { kind: "approach_target" });
  const done = autopilot(visibleAt(2, 1.1), DEFAULT_LIMITS, search);
  assert.equal(done.kind === "finish" && done.success, true);
});
