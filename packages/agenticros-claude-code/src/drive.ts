/**
 * Reactive drive toward a direction seen by the camera (ros2_drive_toward), without any pose
 * estimate: no odometry, no IMU, no map. Each call is short (at most 3 s): the caller looks
 * again (vision model) and sends the next one. Obstacles come from the latest LaserScan in the
 * body frame (go2_real lidar_obstacle_scan: Mid-360 points 10-70 cm above the floor), with no
 * memory between scans.
 *
 * Within one call the remaining bearing and distance are the requested ones minus the motion
 * commanded so far (scaled by the measured locomotion efficiencies): a short feed-forward that
 * the next observation replaces, never accumulated across calls.
 */

/** One obstacle point of the scan in the body frame (x forward, y left), metres. */
export interface ScanPoint {
  x: number;
  y: number;
}

export interface HeadingChoice {
  /** Chosen heading in the body frame, degrees, positive = left. Null when every direction is blocked. */
  headingDeg: number | null;
  /** Free distance along the chosen heading (m), capped at lookaheadM. */
  freeM: number;
  /** Free distance straight ahead (m), capped at lookaheadM. */
  frontFreeM: number;
}

export interface HeadingParams {
  /** Half width of the corridor swept by the body (m). */
  halfWidthM: number;
  /** Candidate headings span +/- this many degrees around straight ahead. */
  maxHeadingDeg: number;
  stepDeg: number;
  /** Free distances are capped here (m). */
  lookaheadM: number;
  /** A heading is usable only with at least this much free distance (m). */
  minFreeM: number;
  /** Free distance wanted along the heading (m): shorter ones are penalized. */
  wantedFreeM: number;
}

/**
 * The ranges of a LaserScan as numbers. Through the AgenticROS transport a float32[] field can
 * arrive as its raw little-endian bytes (a Uint8Array or a plain array of 0-255 values, four per
 * range): read as metres they gave a drive without obstacles. `count` is the number of beams
 * from the scan angles.
 */
export function decodeRanges(raw: ArrayLike<number>, count: number): number[] {
  const bytesPerElement = (raw as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT;
  const isFloatArray = bytesPerElement === 4 || bytesPerElement === 8;
  if (!isFloatArray && count > 0 && raw.length === 4 * count) {
    const bytes = Uint8Array.from(raw as ArrayLike<number>);
    const view = new DataView(bytes.buffer);
    return Array.from({ length: count }, (_, i) => view.getFloat32(4 * i, true));
  }
  return Array.from(raw);
}

/** Beams of a LaserScan from its angles (angle_max is inclusive). */
export function beamCount(angleMin: number, angleMax: number, increment: number): number {
  return increment > 0 ? Math.round((angleMax - angleMin) / increment) + 1 : 0;
}

export function scanPoints(scan: { ranges: number[]; angle_min: number; angle_increment: number; range_max?: number }): ScanPoint[] {
  const pts: ScanPoint[] = [];
  const max = Number.isFinite(scan.range_max) && scan.range_max! > 0 ? scan.range_max! : Infinity;
  for (let i = 0; i < scan.ranges.length; i++) {
    const r = scan.ranges[i]!;
    if (!Number.isFinite(r) || r <= 0 || r > max) continue;
    const a = scan.angle_min + i * scan.angle_increment;
    pts.push({ x: r * Math.cos(a), y: r * Math.sin(a) });
  }
  return pts;
}

/** Free distance along a heading: nearest obstacle point inside the corridor swept by the body. */
export function freeAlong(points: ScanPoint[], headingDeg: number, halfWidthM: number, lookaheadM: number): number {
  const h = (headingDeg * Math.PI) / 180;
  const c = Math.cos(h);
  const s = Math.sin(h);
  let free = lookaheadM;
  for (const p of points) {
    const along = p.x * c + p.y * s;
    if (along <= 0 || along >= free) continue;
    if (Math.abs(-p.x * s + p.y * c) <= halfWidthM) free = along;
  }
  return free;
}

/**
 * Heading closest to the wanted bearing among those with room to move (a vector field histogram
 * reduced to one row): cost = angular deviation + a penalty for each metre of missing room.
 */
export function chooseHeading(points: ScanPoint[], bearingDeg: number, p: HeadingParams): HeadingChoice {
  const frontFreeM = freeAlong(points, 0, p.halfWidthM, p.lookaheadM);
  let best: { h: number; free: number; cost: number } | null = null;
  for (let h = -p.maxHeadingDeg; h <= p.maxHeadingDeg + 1e-9; h += p.stepDeg) {
    const free = freeAlong(points, h, p.halfWidthM, p.lookaheadM);
    if (free < p.minFreeM) continue;
    // 100 deg per missing metre: with an obstacle 1 m ahead and 1.5 m wanted, a 45 deg detour with room wins.
    const cost = Math.abs(h - bearingDeg) + 100 * Math.max(0, p.wantedFreeM - free);
    if (!best || cost < best.cost) best = { h, free, cost };
  }
  return best ? { headingDeg: best.h, freeM: best.free, frontFreeM } : { headingDeg: null, freeM: 0, frontFreeM };
}

export interface DriveCommandParams {
  maxLinear: number;
  maxAngular: number;
  /** Proportional gain from heading error (rad) to angular speed (rad/s). */
  turnGain: number;
  /** Beyond this heading error the robot turns in place. */
  turnInPlaceDeg: number;
  /** Stop advancing when the free distance ahead drops below this (m). */
  stopClearanceM: number;
  /** Slow down over this distance above stopClearanceM (m). */
  slowDownM: number;
}

/** Velocity command toward a chosen heading, given the room ahead and the distance left. */
export function driveCommand(
  headingDeg: number,
  frontFreeM: number,
  remainingM: number,
  p: DriveCommandParams,
): { linear: number; angular: number } {
  const h = (headingDeg * Math.PI) / 180;
  const angular = Math.max(-p.maxAngular, Math.min(p.maxAngular, p.turnGain * h));
  if (remainingM <= 0 || Math.abs(headingDeg) > p.turnInPlaceDeg || frontFreeM <= p.stopClearanceM) {
    return { linear: 0, angular };
  }
  const room = Math.min(1, (frontFreeM - p.stopClearanceM) / p.slowDownM);
  const align = Math.cos(h) ** 2;
  // Do not overshoot the distance left within one 20 Hz cycle at full speed.
  const linear = Math.min(p.maxLinear * room * align, remainingM * 4);
  return { linear: Math.max(0, linear), angular };
}

/**
 * Forward progress between two scans of the same drive, from the scan alone (no odometry): the
 * median shortening of the ranges within +/- halfAngleDeg ahead (30 deg: wider beams also see
 * surfaces beside the path, which shorten less; one slice of 0.6 m read 0 with 45 deg). Moving forward shortens the
 * ranges to the surfaces ahead by the distance covered. Null with too few comparable beams
 * (open space beyond the scan range) or after a turn, which shifts the beams.
 */
export function scanProgress(
  before: { ranges: number[]; angle_min: number; angle_increment: number },
  after: { ranges: number[]; angle_min: number; angle_increment: number },
  halfAngleDeg = 30,
  maxRangeM = 6,
  minBeams = 20,
): number | null {
  if (before.ranges.length !== after.ranges.length || before.angle_increment !== after.angle_increment) return null;
  const diffs: number[] = [];
  const half = (halfAngleDeg * Math.PI) / 180;
  for (let i = 0; i < before.ranges.length; i++) {
    const a = before.angle_min + i * before.angle_increment;
    if (Math.abs(a) > half) continue;
    const r0 = before.ranges[i]!;
    const r1 = after.ranges[i]!;
    if (!Number.isFinite(r0) || !Number.isFinite(r1) || r0 > maxRangeM || r1 > maxRangeM) continue;
    diffs.push(r0 - r1);
  }
  if (diffs.length < minBeams) return null;
  diffs.sort((x, y) => x - y);
  return diffs[Math.floor(diffs.length / 2)]!;
}
