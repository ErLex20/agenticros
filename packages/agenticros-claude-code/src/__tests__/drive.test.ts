import { test } from "node:test";
import assert from "node:assert/strict";
import { chooseHeading, driveCommand, freeAlong, scanPoints, type ScanPoint } from "../drive.js";

const PARAMS = { halfWidthM: 0.25, maxHeadingDeg: 60, stepDeg: 5, lookaheadM: 3, minFreeM: 0.45, wantedFreeM: 1.5 };
const COMMAND = { maxLinear: 0.25, maxAngular: 0.5, turnGain: 1.5, turnInPlaceDeg: 35, stopClearanceM: 0.35, slowDownM: 0.6 };

/** A wall across the path at x = distance, from y = yMin to y = yMax. */
function wall(distance: number, yMin: number, yMax: number): ScanPoint[] {
  const pts: ScanPoint[] = [];
  for (let y = yMin; y <= yMax + 1e-9; y += 0.05) pts.push({ x: distance, y });
  return pts;
}

test("scanPoints drops non-finite and out-of-range returns", () => {
  const pts = scanPoints({ ranges: [1, Infinity, NaN, 0, 9], angle_min: 0, angle_increment: Math.PI / 2, range_max: 8 });
  assert.equal(pts.length, 1);
  assert.ok(Math.abs(pts[0]!.x - 1) < 1e-9);
});

test("freeAlong sees only points inside the body corridor", () => {
  const pts = [{ x: 1, y: 0.4 }, { x: 2, y: 0.1 }];
  assert.ok(Math.abs(freeAlong(pts, 0, 0.25, 3) - 2) < 1e-9);
  assert.equal(freeAlong([], 0, 0.25, 3), 3);
});

test("chooseHeading keeps the bearing when it is free", () => {
  const choice = chooseHeading(wall(1.0, 0.6, 2), 0, PARAMS);
  assert.equal(choice.headingDeg, 0);
  assert.equal(choice.frontFreeM, 3);
});

test("chooseHeading goes around an obstacle on the bearing, on the free side", () => {
  // Box 1 m ahead from y = -0.4 to 0.3: the right side (negative) is blocked farther out.
  const pts = [...wall(1.0, -0.4, 0.3), ...wall(1.2, -2, -0.4)];
  const choice = chooseHeading(pts, 0, PARAMS);
  assert.ok(choice.headingDeg != null && choice.headingDeg > 0, `heading ${choice.headingDeg}`);
  assert.ok(choice.freeM >= 1.5);
});

test("chooseHeading reports blocked when every direction is closed", () => {
  const ring: ScanPoint[] = [];
  for (let a = -90; a <= 90; a += 2) ring.push({ x: 0.3 * Math.cos((a * Math.PI) / 180), y: 0.3 * Math.sin((a * Math.PI) / 180) });
  assert.equal(chooseHeading(ring, 0, PARAMS).headingDeg, null);
});

test("driveCommand turns in place for large errors and stops before obstacles", () => {
  assert.equal(driveCommand(50, 3, 1, COMMAND).linear, 0);
  assert.ok(driveCommand(50, 3, 1, COMMAND).angular > 0);
  assert.equal(driveCommand(0, 0.3, 1, COMMAND).linear, 0);
  assert.equal(driveCommand(0, 3, 1, COMMAND).linear, 0.25);
  assert.ok(driveCommand(0, 0.65, 1, COMMAND).linear < 0.25);
  assert.equal(driveCommand(0, 3, 0, COMMAND).linear, 0);
});
