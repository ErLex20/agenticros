import { test } from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_LIMITS,
  autopilot,
  bearingFromImageX,
  isGoalReached,
  isPhantomSighting,
  chooseExploreHeading,
  maxSafeAdvance,
  newSearchState,
  planIntent,
  recordAction,
  recordObservation,
  type Observation,
  type SearchState,
} from "../policy.js";

// The search logic is tested with 45 deg steps; the default step has its own test.
const LIMITS = { ...DEFAULT_LIMITS, searchStepDeg: 45 };

function obs(partial: Partial<Observation> = {}): Observation {
  return { step: 1, target: null, front_clearance_m: null, ...partial };
}

// Front clearance is measured (5 m) unless a test overrides it.
const visibleAt = (bearing: number, distance: number | null): Observation =>
  obs({
    target: { visible: true, source: "yolo", confidence: 0.9, bearing_deg: bearing, distance_m: distance, height_frac: 0.1 },
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
  assert.equal(isGoalReached(visibleAt(0, 1.2), LIMITS), true);
  assert.equal(isGoalReached(visibleAt(0, 2.0), LIMITS), false);
  assert.equal(isGoalReached(obs({ target: { visible: false, source: "yolo" } }), LIMITS), false);
  const tallNoDepth = obs({ target: { visible: true, source: "vlm", height_frac: 0.6, distance_m: null } });
  assert.equal(isGoalReached(tallNoDepth, LIMITS), true);
});

const search = newSearchState(-1);
const searchWith = (partial: Partial<SearchState>): SearchState => ({ ...newSearchState(-1), ...partial });

test("advance is clamped by stop distance and front clearance", () => {
  assert.equal(maxSafeAdvance(visibleAt(0, 3), LIMITS), 1.5);
  assert.equal(maxSafeAdvance(visibleAt(0, 1.8), LIMITS), 0.8);
  const blocked = { ...visibleAt(0, 5), front_clearance_m: 0.9 };
  assert.ok(Math.abs(maxSafeAdvance(blocked, LIMITS) - 0.45) < 1e-9);
  // Unknown clearance (depth failure, obstacle closer than the stereo minimum) fails safe.
  const unknown = { ...visibleAt(0, 5), front_clearance_m: null };
  assert.equal(maxSafeAdvance(unknown, LIMITS), LIMITS.unknownClearanceAdvanceM);

  const res = planIntent({ kind: "advance", meters: 2 }, visibleAt(0, 1.8), LIMITS, search);
  assert.ok(res.ok && res.action.kind === "advance" && Math.abs(res.action.meters - 0.8) < 1e-9);
});

test("semantic intents compute angle and sign in the controller", () => {
  assert.deepEqual(planIntent({ kind: "face_target" }, visibleAt(49, 3), LIMITS, search), {
    ok: true,
    action: { kind: "rotate", degrees: 49 },
  });
  const align = planIntent({ kind: "approach_target" }, visibleAt(-30, 4), LIMITS, search);
  assert.ok(align.ok && align.action.kind === "rotate" && align.action.degrees === -30);
  const walk = planIntent({ kind: "approach_target" }, visibleAt(5, 4), LIMITS, search);
  assert.ok(walk.ok && walk.action.kind === "advance" && walk.action.meters === 1);
  const hidden = obs({ target: { visible: false, source: "yolo" } });
  assert.deepEqual(planIntent({ kind: "search" }, hidden, LIMITS, searchWith({ direction: 1 })), {
    ok: true,
    action: { kind: "rotate", degrees: 45 },
  });
  assert.equal(planIntent({ kind: "search" }, visibleAt(10, 3), LIMITS, search).ok, false);
  assert.equal(planIntent({ kind: "search" }, hidden, LIMITS, searchWith({ searchedDeg: 405 })).ok, false);
  assert.equal(planIntent({ kind: "approach_target" }, hidden, LIMITS, search).ok, false);
});

test("off-axis advances and premature success are rejected", () => {
  assert.equal(planIntent({ kind: "advance", meters: 1 }, visibleAt(40, 4), LIMITS, search).ok, false);
  assert.equal(planIntent({ kind: "finish", success: true, message: "" }, visibleAt(0, 3), LIMITS, search).ok, false);
  assert.equal(planIntent({ kind: "finish", success: false, message: "" }, visibleAt(0, 3), LIMITS, search).ok, true);
  assert.equal(planIntent({ kind: "rotate", degrees: 0 }, visibleAt(0, 3), LIMITS, search).ok, false);
  // A near reading at the image edge is not "arrived".
  assert.equal(isGoalReached(visibleAt(-50, 1.1), LIMITS), false);
  // A vision-model column band may contain a nearer object: a small target is not "arrived".
  const vlmSmall = obs({ target: { visible: true, source: "vlm", localization: "column", bearing_deg: 0, distance_m: 1.1, height_frac: 0.12 } });
  assert.equal(isGoalReached(vlmSmall, LIMITS), false);
  // A grounding model's box bounds the target itself: depth in it is the target's, like a detector box.
  const vlmBox = obs({ target: { visible: true, source: "vlm", localization: "box", bearing_deg: 0, distance_m: 1.1, height_frac: 0.12 } });
  assert.equal(isGoalReached(vlmBox, LIMITS), true);
  assert.equal(planIntent({ kind: "advance", meters: 0.01 }, visibleAt(0, 3), LIMITS, search).ok, false);
});

test("autopilot searches, approaches and stops", () => {
  const hidden = obs({ target: { visible: false, source: "yolo" } });
  assert.deepEqual(autopilot(hidden, LIMITS, search), { kind: "search" });
  // Full turn with nowhere open to go, or with the exploration budget spent: give up.
  assert.equal(autopilot(hidden, LIMITS, searchWith({ searchedDeg: 420 })).kind, "finish");
  const open = [{ heading_deg: 90, clearance_m: 4 }];
  assert.equal(autopilot(hidden, LIMITS, searchWith({ searchedDeg: 420, scan: open })).kind, "explore");
  const spent = searchWith({ searchedDeg: 420, scan: open, explorations: LIMITS.maxExplorations });
  assert.equal(autopilot(hidden, LIMITS, spent).kind, "finish");
  assert.deepEqual(autopilot(visibleAt(30, 4), LIMITS, search), { kind: "approach_target" });
  const done = autopilot(visibleAt(2, 1.1), LIMITS, search);
  assert.equal(done.kind === "finish" && done.success, true);
});

const hiddenWith = (clearance: number | null): Observation =>
  obs({ target: { visible: false, source: "yolo" }, front_clearance_m: clearance });

test("explore picks the most open heading and avoids walking back", () => {
  const scan = [
    { heading_deg: 0, clearance_m: 2.2 },
    { heading_deg: -45, clearance_m: 4 },
    { heading_deg: -90, clearance_m: 1.0 },
    { heading_deg: 180, clearance_m: 8.0 },
  ];
  assert.equal(chooseExploreHeading(searchWith({ scan }), LIMITS), 180);
  // Coming from heading 0, 180 leads back: the open heading -45 wins.
  assert.equal(chooseExploreHeading(searchWith({ scan, lastExploreHeadingDeg: 0 }), LIMITS), -45);
  assert.equal(chooseExploreHeading(searchWith({ scan, blockedHeadingsDeg: [180, -45] }), LIMITS), 0);
  assert.equal(chooseExploreHeading(searchWith({ scan: [{ heading_deg: 0, clearance_m: 1 }] }), LIMITS), null);
});

test("explore is allowed only after a full turn and within the budget", () => {
  const scan = [{ heading_deg: 90, clearance_m: 4 }];
  assert.equal(planIntent({ kind: "explore" }, hiddenWith(4), LIMITS, searchWith({ searchedDeg: 90, scan })).ok, false);
  assert.equal(planIntent({ kind: "explore" }, visibleAt(0, 3), LIMITS, searchWith({ searchedDeg: 405, scan })).ok, false);
  const spent = searchWith({ searchedDeg: 405, scan, explorations: LIMITS.maxExplorations });
  assert.equal(planIntent({ kind: "explore" }, hiddenWith(4), LIMITS, spent).ok, false);
  // During an exploration move plain search is rejected so the move is completed.
  const searched = planIntent({ kind: "search" }, hiddenWith(4), LIMITS, searchWith({ searchedDeg: 405, scan }));
  assert.ok(!searched.ok && searched.reason.endsWith("use explore"));
  const exploring = searchWith({ exploreHeadingDeg: 90, exploreRemainingM: 1 });
  assert.equal(planIntent({ kind: "search" }, hiddenWith(4), LIMITS, exploring).ok, false);
});

test("an exploration turns, walks, then searches a full turn from the new spot", () => {
  const s = newSearchState(-1);
  // Search turn: clearance recorded at every heading.
  for (const clearance of [2, 1, 6, 1.2, 1, 1, 1, 1, 2]) {
    const o = hiddenWith(clearance);
    recordObservation(s, o, LIMITS);
    const plan = planIntent({ kind: "search" }, o, LIMITS, s);
    assert.ok(plan.ok);
    recordAction(s, o, plan.action, plan, LIMITS);
  }
  assert.equal(s.searchedDeg, 405);
  assert.equal(s.headingDeg, -45);
  // Most open reading (6 m) was the third one, at heading -90: turn there.
  let o = hiddenWith(2);
  recordObservation(s, o, LIMITS);
  let plan = planIntent({ kind: "explore" }, o, LIMITS, s);
  assert.ok(plan.ok && plan.action.kind === "rotate" && plan.action.degrees === -45 && plan.explore?.phase === "turn");
  recordAction(s, o, plan.action, plan, LIMITS);
  assert.equal(s.headingDeg, -90);
  assert.equal(s.exploreHeadingDeg, -90);
  // Walk in steps bounded by the front clearance until exploreStepM is covered.
  o = hiddenWith(6);
  recordObservation(s, o, LIMITS);
  plan = planIntent({ kind: "explore" }, o, LIMITS, s);
  assert.ok(plan.ok && plan.action.kind === "advance" && plan.action.meters === 1.5);
  recordAction(s, o, plan.action, plan, LIMITS);
  o = hiddenWith(6);
  plan = planIntent({ kind: "explore" }, o, LIMITS, s);
  assert.ok(plan.ok && plan.action.kind === "advance" && plan.action.meters === 1);
  recordAction(s, o, plan.action, plan, LIMITS);
  // Done: a fresh search turn starts from the new spot.
  assert.equal(s.explorations, 1);
  assert.equal(s.exploreHeadingDeg, null);
  assert.equal(s.lastExploreHeadingDeg, -90);
  assert.equal(s.searchedDeg, 0);
  assert.deepEqual(s.scan, []);
  assert.equal(planIntent({ kind: "search" }, hiddenWith(3), LIMITS, s).ok, true);
});

test("a blocked exploration heading is skipped without a new turn", () => {
  const scan = [
    { heading_deg: 0, clearance_m: 5 },
    { heading_deg: 90, clearance_m: 3 },
  ];
  const s = searchWith({ searchedDeg: 405, scan });
  const blocked = hiddenWith(0.5); // something appeared in front since the scan
  const plan = planIntent({ kind: "explore" }, blocked, LIMITS, s);
  assert.ok(plan.ok && plan.explore?.phase === "blocked");
  recordAction(s, blocked, plan.action, plan, LIMITS);
  assert.equal(s.explorations, 0);
  assert.deepEqual(s.blockedHeadingsDeg, [0]);
  assert.equal(chooseExploreHeading(s, LIMITS), 90);
});

test("seeing the target cancels the exploration", () => {
  const s = searchWith({ searchedDeg: 405, exploreHeadingDeg: 90, exploreRemainingM: 1 });
  recordObservation(s, visibleAt(20, 3), LIMITS);
  assert.equal(s.exploreHeadingDeg, null);
  assert.equal(s.searchedDeg, 0);
  assert.equal(s.direction, 1);
});

test("a single vision-model sighting keeps the search and cannot finish the task", () => {
  const s = searchWith({ searchedDeg: 405, exploreHeadingDeg: 90, exploreRemainingM: 1 });
  const vlm = obs({
    target: { visible: true, source: "vlm", bearing_deg: 0, distance_m: 1.2, height_frac: 0.45 },
    front_clearance_m: 1.2,
  });
  recordObservation(s, vlm, LIMITS);
  assert.equal(s.exploreHeadingDeg, 90);
  assert.equal(s.searchedDeg, 405);
  assert.equal(planIntent({ kind: "finish", success: true, message: "" }, vlm, LIMITS, s).ok, false);
  assert.equal(autopilot(vlm, LIMITS, s).kind, "approach_target");
  // Seen again in the next frame: confirmed.
  recordObservation(s, vlm, LIMITS);
  assert.equal(s.exploreHeadingDeg, null);
  assert.equal(planIntent({ kind: "finish", success: true, message: "" }, vlm, LIMITS, s).ok, true);
  // A miss in between resets the streak.
  recordObservation(s, hiddenWith(3), LIMITS);
  assert.equal(s.sightingStreak, 0);
});

test("a low-confidence YOLO sighting needs a second frame, like the vision model", () => {
  const s = searchWith({ searchedDeg: 405, exploreHeadingDeg: 90, exploreRemainingM: 1 });
  const weak = obs({
    target: { visible: true, source: "yolo", confidence: 0.4, bearing_deg: 0, distance_m: 1.2, height_frac: 0.5 },
    front_clearance_m: 1.2,
  });
  recordObservation(s, weak, LIMITS);
  assert.equal(s.exploreHeadingDeg, 90);
  assert.equal(autopilot(weak, LIMITS, s).kind, "approach_target");
  recordObservation(s, weak, LIMITS);
  assert.equal(autopilot(weak, LIMITS, s).kind, "finish");
});

test("doorways confirmed by two frames of the turn are explored first", () => {
  const s = newSearchState(-1);
  const seeing = (bearings: number[]) =>
    obs({ target: { visible: false, source: "vlm" }, front_clearance_m: 2, doorway_bearings_deg: bearings });
  s.headingDeg = 90;
  recordObservation(s, seeing([-30]), LIMITS);
  s.headingDeg = 0;
  recordObservation(s, hiddenWith(8), LIMITS);
  // A single report is not enough...
  assert.equal(chooseExploreHeading(s, LIMITS), 0);
  // ...a second frame seeing the same doorway (from another heading) confirms it.
  s.headingDeg = 45;
  recordObservation(s, seeing([17.6, -170]), LIMITS);
  assert.deepEqual(s.doorwayFrames, [[60], [62.6, -125]]);
  // The doorway beats an 8 m open direction...
  assert.equal(chooseExploreHeading(s, LIMITS), 60);
  // ...unless it leads back where the robot came from, or was found blocked.
  assert.equal(chooseExploreHeading({ ...s, lastExploreHeadingDeg: -100 }, LIMITS), 0);
  assert.equal(chooseExploreHeading({ ...s, blockedHeadingsDeg: [60] }, LIMITS), 0);
});

test("with a live map, explore drives to a frontier and a new turn starts there", () => {
  const limits = { ...LIMITS, frontierExploration: true };
  const s = searchWith({ searchedDeg: 405, headingDeg: 135, scan: [{ heading_deg: 90, clearance_m: 4 }] });
  // Not before a full turn
  assert.equal(planIntent({ kind: "explore" }, hiddenWith(3), limits, searchWith({ searchedDeg: 90 })).ok, false);
  const plan = planIntent({ kind: "explore" }, hiddenWith(3), limits, s);
  assert.ok(plan.ok && plan.action.kind === "frontier" && plan.explore?.phase === "frontier");
  recordAction(s, hiddenWith(3), plan.action, plan, limits);
  assert.equal(s.explorations, 1);
  assert.equal(s.searchedDeg, 0);
  assert.equal(s.headingDeg, 0);
  assert.equal(s.exploreHeadingDeg, null);
  assert.deepEqual(s.scan, []);
  assert.equal(planIntent({ kind: "search" }, hiddenWith(3), limits, s).ok, true);
  // The exploration budget still applies.
  const spent = searchWith({ searchedDeg: 405, explorations: LIMITS.maxExplorations });
  assert.equal(planIntent({ kind: "explore" }, hiddenWith(3), limits, spent).ok, false);
  // The fallback policy explores frontiers even when no direction looked open in the turn.
  assert.equal(autopilot(hiddenWith(1), limits, searchWith({ searchedDeg: 405 })).kind, "explore");
  // The model cannot ask for a frontier trip directly.
  assert.equal(planIntent({ kind: "frontier" }, hiddenWith(3), limits, s).ok, false);
});

test("an unconfirmed sighting blocked by an obstacle does not end the search", () => {
  // The vision model "sees" a bed 0.44 m ahead: it is the sofa the robot faces.
  const sofa = obs({
    target: { visible: true, source: "vlm", bearing_deg: 0, distance_m: 0.44, height_frac: 0.45 },
    front_clearance_m: 0.44,
  });
  const s = searchWith({ sightingStreak: 1 });
  assert.deepEqual(autopilot(sofa, LIMITS, s), { kind: "search" });
  const plan = planIntent({ kind: "search" }, sofa, LIMITS, s);
  assert.ok(plan.ok && plan.action.kind === "rotate");
  recordAction(s, sofa, plan.action, plan, LIMITS);
  assert.equal(s.searchedDeg, 45);
  // After a full turn it explores, it does not give up.
  const turned = searchWith({ sightingStreak: 1, searchedDeg: 405 });
  assert.equal(autopilot(sofa, { ...LIMITS, frontierExploration: true }, turned).kind, "explore");
  // Confirmed (seen again) and still blocked: the path really is blocked.
  const confirmed = searchWith({ sightingStreak: 2 });
  assert.equal(autopilot(sofa, LIMITS, confirmed).kind, "finish");
});

test("approach steps are not crawled 10 cm at a time", () => {
  const plan = planIntent({ kind: "approach_target", meters: 0.1 }, visibleAt(2, 2.2), LIMITS, search);
  assert.ok(plan.ok && plan.action.kind === "advance" && plan.action.meters === 0.5);
  const far = planIntent({ kind: "approach_target", meters: 0.1 }, visibleAt(2, 4), LIMITS, search);
  assert.ok(far.ok && far.action.kind === "advance" && far.action.meters === 1);
  // Close to the stop distance the safety clamp still wins.
  const near = planIntent({ kind: "approach_target", meters: 1 }, visibleAt(0, 1.4), LIMITS, search);
  assert.ok(near.ok && near.action.kind === "advance" && Math.abs(near.action.meters - 0.4) < 1e-9);
});

test("a depth reading that does not match a floor-standing box is an occluder", async () => {
  const { depthConsistentWithBox } = await import("../policy.js");
  const cam = { heightM: 0.385, hfovDeg: 110, aspect: 16 / 9 };
  // Refrigerator ~3.4 m away behind a chair: box bottom at 0.56, depth reads the chair at 0.91 m.
  assert.equal(depthConsistentWithBox({ x_min: 0.49, y_min: 0.21, x_max: 0.61, y_max: 0.56 }, 0.91, cam), false);
  // Same box, depth of the refrigerator itself: consistent.
  assert.equal(depthConsistentWithBox({ x_min: 0.49, y_min: 0.21, x_max: 0.61, y_max: 0.56 }, 3.5, cam), true);
  // A table 0.34 m away would fill the image bottom: a box ending at 0.56 is something farther.
  assert.equal(depthConsistentWithBox({ x_min: 0.49, y_min: 0.21, x_max: 0.61, y_max: 0.56 }, 0.34, cam), false);
  // Wall-mounted TV (box above the horizon) is not judged; a close sofa reaching the image bottom is fine.
  assert.equal(depthConsistentWithBox({ x_min: 0.4, y_min: 0.1, x_max: 0.6, y_max: 0.4 }, 1.0, cam), true);
  assert.equal(depthConsistentWithBox({ x_min: 0.1, y_min: 0.3, x_max: 0.9, y_max: 1.0 }, 0.3, cam), true);
});

test("a visible target with unknown distance is approached in short steps", () => {
  const blind = obs({ target: { visible: true, source: "vlm", localization: "box", bearing_deg: 0, distance_m: null, height_frac: 0.2 }, front_clearance_m: 5 });
  assert.equal(maxSafeAdvance(blind, LIMITS), LIMITS.unknownTargetAdvanceM);
});

test("a first sighting off to the side is looked at again, not searched away from", () => {
  const once = obs({ target: { visible: true, source: "vlm", localization: "box", bearing_deg: -46, distance_m: 2.1, height_frac: 0.1 }, front_clearance_m: 5 });
  const s = searchWith({ sightingStreak: 1 });
  const plan = planIntent({ kind: "search" }, once, LIMITS, s);
  assert.ok(!plan.ok && plan.reason.includes("face_target"));
  assert.deepEqual(planIntent({ kind: "face_target" }, once, LIMITS, s), { ok: true, action: { kind: "rotate", degrees: -46 } });
  // Seen once straight ahead (e.g. the sofa taken for a bed): searching on stays allowed.
  const ahead = obs({ target: { visible: true, source: "vlm", localization: "box", bearing_deg: 2, distance_m: 0.44, height_frac: 0.45 }, front_clearance_m: 0.44 });
  assert.equal(planIntent({ kind: "search" }, ahead, LIMITS, s).ok, true);
});

test("far targets are approached in full steps even if the model copies a short one", () => {
  const plan = planIntent({ kind: "approach_target", meters: 0.5 }, visibleAt(0, 5), LIMITS, search);
  assert.ok(plan.ok && plan.action.kind === "advance" && plan.action.meters === 1);
});

test("the default search turn covers every direction in 5 quarter turns", () => {
  const s = newSearchState(-1);
  let rotations = 0;
  for (;;) {
    const o = hiddenWith(3);
    recordObservation(s, o, DEFAULT_LIMITS);
    const plan = planIntent({ kind: "search" }, o, DEFAULT_LIMITS, s);
    if (!plan.ok) break;
    assert.ok(plan.action.kind === "rotate" && Math.abs(plan.action.degrees) === 90);
    recordAction(s, o, plan.action, plan, DEFAULT_LIMITS);
    rotations += 1;
  }
  assert.equal(rotations, 5);
});

test("a sighting that vanishes twice when faced is ignored from that spot", () => {
  // Simulation: a sofa passed for a bed at ~45 deg, near the image edge, and was gone once faced.
  const edge = obs({ target: { visible: true, source: "vlm", localization: "box", bearing_deg: 45, distance_m: 3.4, height_frac: 0.07 }, front_clearance_m: 5 });
  const gone = obs({ target: { visible: false, source: "vlm", localization: "box" }, front_clearance_m: 5 });
  const s = newSearchState(-1);
  for (let round = 1; round <= 2; round++) {
    assert.equal(isPhantomSighting(edge, s, LIMITS), false);
    recordObservation(s, edge, LIMITS);
    const face = planIntent({ kind: "face_target" }, edge, LIMITS, s);
    assert.ok(face.ok);
    recordAction(s, edge, face.action, face, LIMITS);
    recordObservation(s, gone, LIMITS);
    const back = planIntent({ kind: "search" }, gone, LIMITS, s);
    assert.ok(back.ok && back.action.kind === "rotate" && back.action.degrees === -45);
    recordAction(s, gone, back.action, back, LIMITS);
  }
  assert.equal(isPhantomSighting(edge, s, LIMITS), true);
  // Elsewhere, or confirmed by YOLO, a sighting still counts.
  const other = obs({ target: { visible: true, source: "vlm", localization: "box", bearing_deg: -40, distance_m: 3, height_frac: 0.1 }, front_clearance_m: 5 });
  assert.equal(isPhantomSighting(other, s, LIMITS), false);
  assert.equal(isPhantomSighting(visibleAt(45, 3.4), s, LIMITS), false);
  // From a new spot (after an exploration walk) the slate is clean.
  const walked = { ...s, exploreHeadingDeg: 0, exploreRemainingM: 0.2 };
  recordAction(walked, gone, { kind: "advance", meters: 0.2 }, { ok: true, action: { kind: "advance", meters: 0.2 }, explore: { phase: "walk", headingDeg: 0 } }, LIMITS);
  assert.equal(isPhantomSighting(edge, walked, LIMITS), false);
});

test("a sighting that vanishes once when faced is not a phantom", () => {
  const edge = obs({ target: { visible: true, source: "vlm", localization: "box", bearing_deg: 45, distance_m: 3.4, height_frac: 0.07 }, front_clearance_m: 5 });
  const s = newSearchState(-1);
  recordObservation(s, edge, LIMITS);
  const face = planIntent({ kind: "face_target" }, edge, LIMITS, s);
  assert.ok(face.ok);
  recordAction(s, edge, face.action, face, LIMITS);
  recordObservation(s, obs({ target: { visible: false, source: "vlm" }, front_clearance_m: 5 }), LIMITS);
  assert.equal(isPhantomSighting(obs({ ...edge, target: { ...edge.target!, bearing_deg: 0 } }), s, LIMITS), false);
});
