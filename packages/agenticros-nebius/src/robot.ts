/**
 * Thin wrappers around the AgenticROS MCP tools used by the goal loop.
 * Every robot interaction goes through AgenticROS, so its safety limits and
 * timed-stop behaviour always apply.
 */

import type { Client } from "@modelcontextprotocol/sdk/client/index.js";
import type { BBoxNorm } from "./policy.js";

type McpContent = { type: string; text?: string; data?: string; mimeType?: string };

/** ros2_move_for accepts 0.1–3 s per call. */
const MAX_MOVE_SECONDS = 3.0;
const MIN_MOVE_SECONDS = 0.1;
/** Hard cap for one rotate/advance, whatever the requested amount. */
const MAX_ACTION_SECONDS = 20;
const TOOL_TIMEOUT_MS = 30_000;

export interface RobotOptions {
  cameraTopic?: string;
  cameraMessageType: "Image" | "CompressedImage";
  depthTopic?: string;
  maxLinear: number;
  maxAngular: number;
  /** Fraction of the commanded yaw the base actually achieves (legged bases undershoot). */
  turnEfficiency: number;
  /** Fraction of the commanded forward distance the base actually achieves. */
  forwardEfficiency: number;
}

export interface Snapshot {
  image: Buffer;
  mimeType: string;
}

export class RobotTools {
  /** Set on Ctrl-C: no further motion command is sent. */
  aborted = false;

  constructor(
    private readonly mcp: Client,
    readonly opts: RobotOptions,
  ) {}

  private async call(name: string, args: Record<string, unknown>) {
    const result = await this.mcp.callTool({ name, arguments: args }, undefined, { timeout: TOOL_TIMEOUT_MS });
    const content = (result.content ?? []) as McpContent[];
    const text = content
      .filter((c) => c.type === "text" && typeof c.text === "string")
      .map((c) => c.text)
      .join("\n");
    return { content, text, isError: result.isError === true };
  }

  async snapshot(): Promise<Snapshot> {
    const { content, text, isError } = await this.call("ros2_camera_snapshot", {
      message_type: this.opts.cameraMessageType,
      ...(this.opts.cameraTopic ? { topic: this.opts.cameraTopic } : {}),
    });
    const image = content.find((c) => c.type === "image" && typeof c.data === "string");
    if (isError || !image?.data) throw new Error(`Camera snapshot failed: ${text || "no image returned"}`);
    return { image: Buffer.from(image.data, "base64"), mimeType: image.mimeType ?? "image/png" };
  }

  /** Near-surface distance (12th percentile) inside a normalized image region; null when unavailable. */
  async depth(roi: BBoxNorm): Promise<number | null> {
    if (!this.opts.depthTopic) return null;
    const { text, isError } = await this.call("ros2_depth_distance", {
      topic: this.opts.depthTopic,
      roi_x_min: roi.x_min,
      roi_y_min: roi.y_min,
      roi_x_max: roi.x_max,
      roi_y_max: roi.y_max,
    });
    if (isError) return null;
    const json = text.split("\n").reverse().find((line) => line.trim().startsWith("{"));
    if (!json) return null;
    try {
      const parsed = JSON.parse(json) as { valid?: boolean; distance_m?: number };
      return parsed.valid && typeof parsed.distance_m === "number" ? parsed.distance_m : null;
    } catch {
      return null;
    }
  }

  private async moveChunks(linear: number, angular: number, seconds: number): Promise<void> {
    if (!Number.isFinite(seconds) || seconds <= 0) throw new Error(`invalid motion duration: ${seconds}`);
    let remaining = Math.min(seconds, MAX_ACTION_SECONDS);
    // Round a short remainder up to the tool minimum instead of silently dropping it.
    while (remaining >= MIN_MOVE_SECONDS / 2) {
      if (this.aborted) throw new Error("motion aborted");
      const duration = Math.max(MIN_MOVE_SECONDS, Math.min(MAX_MOVE_SECONDS, remaining));
      const { text, isError } = await this.call("ros2_move_for", {
        linear_x: linear,
        angular_z: angular,
        duration_seconds: Math.round(duration * 100) / 100,
      });
      if (isError) throw new Error(`ros2_move_for failed: ${text}`);
      remaining -= duration;
    }
  }

  /** Rotate in place; positive degrees turn left (counter-clockwise). */
  async rotate(degrees: number): Promise<void> {
    const radians = (Math.abs(degrees) * Math.PI) / 180;
    const seconds = radians / (this.opts.maxAngular * this.opts.turnEfficiency);
    await this.moveChunks(0, Math.sign(degrees) * this.opts.maxAngular, seconds);
  }

  async advance(meters: number): Promise<void> {
    const seconds = meters / (this.opts.maxLinear * this.opts.forwardEfficiency);
    await this.moveChunks(this.opts.maxLinear, 0, seconds);
  }

  async estop(): Promise<void> {
    const { text, isError } = await this.call("ros2_estop", {});
    if (isError) throw new Error(`ros2_estop failed: ${text}`);
  }
}
