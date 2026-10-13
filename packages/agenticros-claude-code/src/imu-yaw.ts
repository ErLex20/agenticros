/**
 * Heading change integrated from an IMU gyroscope, for agents that dead-reckon the robot's
 * heading: a legged base pressed against furniture does not turn as commanded.
 *
 * Only the angular velocity is integrated. The orientation is used for roll and pitch alone
 * (observable from gravity on a real IMU) to project the rate on the vertical axis, so a
 * tilted camera IMU still measures yaw; its absolute yaw is never used.
 */

import type { RosTransport } from "@agenticros/core";

export interface ImuYawReading {
  valid: boolean;
  topic: string;
  /** Integrated heading change since the first reading on this topic (deg, counter-clockwise positive, unwrapped). */
  yaw_deg: number;
  samples: number;
  /** Message gaps longer than MAX_GAP_S, and their total duration (s): rotation there is not measured. */
  gaps: number;
  skipped_s: number;
  /** Wall-clock seconds since the last IMU message; null before the first one. */
  age_s: number | null;
}

/** Gaps longer than this (s, message stamps) are not integrated: the rate before them is unknown. */
const MAX_GAP_S = 0.2;
/** Messages kept while the server's JS thread is busy (e.g. converting a camera frame): 5 s at 100 Hz. */
const IMU_QUEUE_LENGTH = 500;

interface Vec3 {
  x: number;
  y: number;
  z: number;
}

class ImuYawIntegrator {
  private yawRad = 0;
  private samples = 0;
  private gaps = 0;
  private skippedS = 0;
  private lastStampS: number | null = null;
  private lastWallMs: number | null = null;

  constructor(transport: RosTransport, private readonly topic: string) {
    transport.subscribe({ topic, type: "sensor_msgs/msg/Imu", queueLength: IMU_QUEUE_LENGTH }, (msg) => this.onMessage(msg));
  }

  private onMessage(msg: Record<string, unknown>): void {
    const header = msg.header as { stamp?: { sec?: number; nanosec?: number } } | undefined;
    const w = msg.angular_velocity as Vec3 | undefined;
    const q = msg.orientation as (Vec3 & { w: number }) | undefined;
    if (!w || ![w.x, w.y, w.z].every(Number.isFinite)) return;
    const stampS = Number(header?.stamp?.sec ?? 0) + Number(header?.stamp?.nanosec ?? 0) / 1e9;
    // Vertical component of the body rate: third row of the rotation matrix of the orientation
    // (depends on roll and pitch only); without a valid orientation the IMU is assumed level.
    let rate = w.z;
    if (q && [q.x, q.y, q.z, q.w].every(Number.isFinite) && Math.hypot(q.x, q.y, q.z, q.w) > 0.5) {
      rate =
        2 * (q.x * q.z - q.w * q.y) * w.x +
        2 * (q.y * q.z + q.w * q.x) * w.y +
        (1 - 2 * (q.x * q.x + q.y * q.y)) * w.z;
    }
    if (this.lastStampS != null) {
      const dt = stampS - this.lastStampS;
      if (dt > 0 && dt <= MAX_GAP_S) this.yawRad += rate * dt;
      else if (dt > MAX_GAP_S) {
        this.gaps++;
        this.skippedS += dt;
      }
    }
    this.lastStampS = stampS;
    this.lastWallMs = Date.now();
    this.samples++;
  }

  reading(): ImuYawReading {
    const age = this.lastWallMs != null ? (Date.now() - this.lastWallMs) / 1000 : null;
    return {
      valid: this.samples > 1 && age != null && age < 1,
      topic: this.topic,
      yaw_deg: Math.round(((this.yawRad * 180) / Math.PI) * 100) / 100,
      samples: this.samples,
      gaps: this.gaps,
      skipped_s: Math.round(this.skippedS * 100) / 100,
      age_s: age != null ? Math.round(age * 100) / 100 : null,
    };
  }
}

const integrators = new Map<string, ImuYawIntegrator>();

/**
 * Integrated heading change on the IMU topic. The first call subscribes and starts integrating
 * (its reading is the baseline); the subscription then stays open for the server's lifetime.
 */
export async function getImuYaw(transport: RosTransport, topic: string, waitMs = 1000): Promise<ImuYawReading> {
  let integrator = integrators.get(topic);
  if (!integrator) {
    integrator = new ImuYawIntegrator(transport, topic);
    integrators.set(topic, integrator);
    const deadline = Date.now() + waitMs;
    while (!integrator.reading().valid && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  }
  return integrator.reading();
}
