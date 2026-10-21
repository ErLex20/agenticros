/**
 * Runtime helpers for named-place tools (pose snapshot + Nav2 dispatch).
 */

import type { AgenticROSConfig } from "./config.js";
import { listCapabilitiesForRobot, type Capability } from "./capabilities.js";
import { executeExternalCapability } from "./external-capability.js";
import { getPlace, poseFromLocalizationMessage, savePlace, type SavedPlace } from "./places.js";
import { resolveSafetyForRobot } from "./safety.js";
import { toNamespacedTopicFull } from "./topic-utils.js";
import type { RosTransport } from "./transport/transport.js";
import type { ResolvedRobot } from "./robots.js";

const POSE_CANDIDATES = [
  { topic: "/amcl_pose", type: "geometry_msgs/msg/PoseWithCovarianceStamped" },
  { topic: "/pose", type: "geometry_msgs/msg/PoseWithCovarianceStamped" },
];

/**
 * Pose topics tried in order: AGENTICROS_POSE_TOPIC first when set (type
 * AGENTICROS_POSE_TYPE, default nav_msgs/msg/Odometry, e.g. the EKF of a robot
 * without AMCL), then the AMCL ones.
 */
function poseCandidates(): { topic: string; type: string }[] {
  const topic = process.env.AGENTICROS_POSE_TOPIC?.trim();
  if (!topic) return POSE_CANDIDATES;
  const type = process.env.AGENTICROS_POSE_TYPE?.trim() || "nav_msgs/msg/Odometry";
  return [{ topic, type }, ...POSE_CANDIDATES];
}

/** A localization pose in its frame (header.frame_id, "map" when missing). */
export interface MapPose {
  x: number;
  y: number;
  yaw: number;
  frame: string;
  topic: string;
}

export async function readCurrentMapPose(
  transport: RosTransport,
  namespace = "",
  timeoutMs = 2500,
): Promise<MapPose | undefined> {
  for (const cand of poseCandidates()) {
    const topic = toNamespacedTopicFull(namespace, cand.topic);
    const pose = await new Promise<MapPose | undefined>((resolve) => {
      const timer = setTimeout(() => {
        try {
          sub.unsubscribe();
        } catch {
          /* ignore */
        }
        resolve(undefined);
      }, timeoutMs);
      const sub = transport.subscribe({ topic, type: cand.type }, (msg) => {
        clearTimeout(timer);
        try {
          sub.unsubscribe();
        } catch {
          /* ignore */
        }
        const p = poseFromLocalizationMessage(msg);
        const header = (msg as { header?: { frame_id?: unknown } } | undefined)?.header;
        const frame = typeof header?.frame_id === "string" && header.frame_id ? header.frame_id : "map";
        resolve(p ? { ...p, frame, topic } : undefined);
      });
    });
    if (pose) return pose;
  }
  return undefined;
}

export async function savePlaceFromArgs(
  args: Record<string, unknown>,
  opts: { robot: ResolvedRobot; transport?: RosTransport },
): Promise<SavedPlace> {
  const name = String(args["name"] ?? "").trim();
  if (!name) throw new Error("ros2_save_place requires 'name'.");
  let x = typeof args["x"] === "number" ? args["x"] : Number.NaN;
  let y = typeof args["y"] === "number" ? args["y"] : Number.NaN;
  let yaw = typeof args["yaw"] === "number" ? args["yaw"] : 0;
  let frame = typeof args["frame"] === "string" ? args["frame"] : "map";
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    if (!opts.transport) {
      throw new Error("Pass x and y, or connect to ROS so the current localization pose can be read.");
    }
    const pose = await readCurrentMapPose(opts.transport, opts.robot.namespace);
    if (!pose) {
      throw new Error(
        "Could not read AGENTICROS_POSE_TOPIC, /amcl_pose or /pose. Pass x and y, or localize first (EKF / AMCL / RTAB-Map).",
      );
    }
    x = pose.x;
    y = pose.y;
    yaw = Number.isFinite(args["yaw"] as number) ? (args["yaw"] as number) : pose.yaw;
    if (typeof args["frame"] !== "string") frame = pose.frame;
  }
  return savePlace({
    name,
    x,
    y,
    yaw,
    frame,
    ...(typeof args["description"] === "string" ? { description: args["description"] } : {}),
    robot_id: opts.robot.id,
  });
}

export function findNavigateToCapability(config: AgenticROSConfig, robotId?: string): Capability | undefined {
  return listCapabilitiesForRobot(config, robotId).find((c) => c.id === "navigate_to");
}

/**
 * Navigate to a pose with the navigate_to capability (Nav2 NavigateToPose).
 * Planning, obstacle avoidance and speed limits are Nav2's.
 */
export async function executeNavigateToPose(
  config: AgenticROSConfig,
  robot: ResolvedRobot,
  pose: { x: number; y: number; yaw?: number; frame_id?: string },
  transport: RosTransport,
  signal?: AbortSignal,
): Promise<{ text: string; isError?: boolean }> {
  if (!Number.isFinite(pose.x) || !Number.isFinite(pose.y)) {
    return { text: "navigate_to_pose requires finite x and y.", isError: true };
  }
  const cap = findNavigateToCapability(config, robot.id);
  if (!cap || cap.implementation?.kind !== "external_ros_node") {
    return {
      text:
        "navigate_to is not installed. Add a navigate_to skill to skillPaths " +
        "(or `agenticros skills install @agenticros/navigate-to`) and restart the agent.",
      isError: true,
    };
  }
  return executeExternalCapability(
    cap,
    { x: pose.x, y: pose.y, yaw: Number.isFinite(pose.yaw) ? pose.yaw : 0, frame_id: pose.frame_id ?? "map" },
    transport,
    {
      namespace: robot.namespace,
      signal,
      workspaceCheck: resolveSafetyForRobot(config, robot),
    },
  );
}

export async function executeNavigateToPlace(
  config: AgenticROSConfig,
  robot: ResolvedRobot,
  name: string,
  transport: RosTransport,
  signal?: AbortSignal,
): Promise<{ text: string; isError?: boolean; place?: SavedPlace }> {
  const place = getPlace(name);
  if (!place) {
    return {
      text: `Unknown place "${name}". Use ros2_list_places or ros2_save_place first.`,
      isError: true,
    };
  }
  const result = await executeNavigateToPose(
    config,
    robot,
    { x: place.x, y: place.y, yaw: place.yaw, frame_id: place.frame },
    transport,
    signal,
  );
  return { ...result, place };
}

/** action_msgs CancelGoal with a zero goal id and stamp: cancel every goal of the server. */
const CANCEL_ALL_GOALS = {
  goal_info: { goal_id: { uuid: new Array(16).fill(0) }, stamp: { sec: 0, nanosec: 0 } },
};

/**
 * Stop navigation for an emergency stop: cancel the goals this transport sent
 * and ask the navigate_to action server (e.g. Nav2) to cancel every goal, also
 * those sent by other clients. Otherwise the planner keeps commanding the base
 * right after a zero cmd_vel. Best effort and bounded in time; a no-op when no
 * navigate_to capability is configured.
 */
export async function cancelNavigation(
  config: AgenticROSConfig,
  robot: ResolvedRobot,
  transport: RosTransport,
  timeoutMs = 1500,
): Promise<boolean> {
  const cap = findNavigateToCapability(config, robot.id);
  const action = cap?.implementation?.kind === "external_ros_node" ? cap.implementation.action : undefined;
  if (!action) return false;
  const name = toNamespacedTopicFull(robot.namespace, action.startsWith("/") ? action : `/${action}`);
  const cancelAll = Promise.allSettled([
    transport.cancelAllActionGoals?.() ?? transport.cancelActionGoal(name),
    transport.callService({
      service: `${name}/_action/cancel_goal`,
      type: "action_msgs/srv/CancelGoal",
      args: CANCEL_ALL_GOALS,
    }),
  ]);
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<void>((resolve) => {
    timer = setTimeout(resolve, timeoutMs);
  });
  await Promise.race([cancelAll, timeout]);
  if (timer) clearTimeout(timer);
  return true;
}
