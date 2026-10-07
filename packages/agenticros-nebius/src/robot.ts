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
/** A Nav2 trip across a house takes minutes at the Go2's walking speed. */
const NAVIGATION_TIMEOUT_MS = 660_000;

/** A named pose of the map (ros2_list_places), e.g. a room. */
export interface Place {
  name: string;
  description?: string;
  x: number;
  y: number;
  yaw: number;
}

export interface NavigationResult {
  ok: boolean;
  /** succeeded, aborted, canceled, rejected or failed. */
  status: string;
  text: string;
}

/** Navigation stack found on the ROS graph. */
export interface NavigationCapabilities {
  /** Nav2 navigate_to_pose is running. */
  navigate: boolean;
  /** Frontier exploration (agenticros_explore) is running: the map is being built live. */
  explore: boolean;
}

/** ROS action of agenticros_explore (go2_sim.launch.py nav:=slam). */
const EXPLORE_ACTION = "/explore";
const EXPLORE_ACTION_TYPE = "agenticros_msgs/action/Explore";
/** A frontier trip that ends sooner than this did not move the robot. */
const MIN_FRONTIER_TRIP_MS = 3000;

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

  private async call(name: string, args: Record<string, unknown>, timeoutMs = TOOL_TIMEOUT_MS) {
    const result = await this.mcp.callTool({ name, arguments: args }, undefined, { timeout: timeoutMs });
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

  /** Named places of the map; empty when none are configured (AGENTICROS_PLACES_PATH). */
  async listPlaces(): Promise<Place[]> {
    const { text, isError } = await this.call("ros2_list_places", {});
    if (isError) return [];
    try {
      const parsed = JSON.parse(text) as { places?: unknown[] };
      return (parsed.places ?? []).filter(
        (p): p is Place =>
          !!p && typeof p === "object" && typeof (p as Place).name === "string" &&
          Number.isFinite((p as Place).x) && Number.isFinite((p as Place).y),
      );
    } catch {
      return [];
    }
  }

  /** Which navigation action servers (Nav2, frontier exploration) are running. */
  async navigationCapabilities(): Promise<NavigationCapabilities> {
    const none = { navigate: false, explore: false };
    const { text, isError } = await this.call("ros2_list_actions", {});
    if (isError) return none;
    try {
      const names = ((JSON.parse(text) as { actions?: { name?: string }[] }).actions ?? []).map((a) => a.name ?? "");
      return {
        navigate: names.some((n) => n.endsWith("/navigate_to_pose")),
        explore: names.some((n) => n === EXPLORE_ACTION),
      };
    } catch {
      return none;
    }
  }

  /**
   * Drive with Nav2 to the nearest unexplored frontier of the live map
   * (agenticros_explore, one goal). ros2_estop cancels it.
   */
  async exploreFrontier(timeoutS = 150): Promise<{ ok: boolean; text: string }> {
    if (this.aborted) throw new Error("motion aborted");
    const startedAt = performance.now();
    const { text, isError } = await this.call(
      "ros2_action_goal",
      {
        action: EXPLORE_ACTION,
        actionType: EXPLORE_ACTION_TYPE,
        goal: { mode: "explore", timeout_s: timeoutS, min_frontier_m: 0.5, max_goals: 1 },
      },
      (timeoutS + 30) * 1000,
    );
    if (isError) return { ok: false, text };
    try {
      const parsed = JSON.parse(text) as { success?: boolean; result?: { goals_sent?: number; message?: string } };
      // agenticros_explore reports success even when Nav2 rejected or aborted its goal:
      // require a goal sent and a trip long enough to have moved (a frontier is at
      // least ~0.5 m away, more than 3 s at the Go2's speed).
      const moved = performance.now() - startedAt > MIN_FRONTIER_TRIP_MS;
      return { ok: parsed.success === true && (parsed.result?.goals_sent ?? 0) > 0 && moved, text };
    } catch {
      return { ok: false, text };
    }
  }

  /**
   * Drive to a named place with Nav2 (ros2_navigate_to_place). Planning,
   * obstacle avoidance and speed limits are Nav2's; ros2_estop cancels it.
   */
  async navigateToPlace(name: string): Promise<NavigationResult> {
    if (this.aborted) throw new Error("motion aborted");
    const { text, isError } = await this.call("ros2_navigate_to_place", { name }, NAVIGATION_TIMEOUT_MS);
    let status = isError ? "failed" : "succeeded";
    const json = text.split("\n").reverse().find((line) => line.trim().startsWith("{"));
    if (json) {
      try {
        const parsed = JSON.parse(json) as { status?: unknown };
        if (typeof parsed.status === "string") status = parsed.status;
      } catch {
        // keep the status derived from isError
      }
    }
    return { ok: !isError && status === "succeeded", status, text };
  }

  async estop(): Promise<void> {
    const { text, isError } = await this.call("ros2_estop", {});
    if (isError) throw new Error(`ros2_estop failed: ${text}`);
  }
}
