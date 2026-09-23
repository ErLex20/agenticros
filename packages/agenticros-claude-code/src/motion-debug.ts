import type { RosTransport } from "@agenticros/core";

/** Observe Gazebo ground truth independently of commanded Twist. */
export function startMotionDebug(transport: RosTransport, cmdTopic: string) {
  const topic = process.env.AGENTICROS_DEBUG_POSE_TOPIC;
  if (!topic) return { phase: (_name: string) => {}, close: () => {} };
  let phase = "baseline";
  let first: { x: number; y: number; yaw: number } | undefined;
  let lastPrint = 0;
  let samples = 0;
  const started = performance.now();
  const cmdSub = transport.subscribe({ topic: cmdTopic, type: "geometry_msgs/msg/Twist" }, (msg) => {
    const v = msg.linear as {x:number;y:number};
    const w = msg.angular as {z:number};
    process.stderr.write(`[MotionDebug wire] t=${((performance.now()-started)/1000).toFixed(2)}s vx=${v?.x} vy=${v?.y} wz=${w?.z}\n`);
  });
  const sub = transport.subscribe({ topic, type: "geometry_msgs/msg/PoseStamped" }, (msg) => {
    const pose = msg.pose as { position?: { x: number; y: number; z: number }; orientation?: { x: number; y: number; z: number; w: number } };
    const p = pose?.position;
    const q = pose?.orientation;
    if (!p || !q || ![p.x, p.y, p.z, q.x, q.y, q.z, q.w].every(Number.isFinite)) return;
    const yaw = Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
    first ??= { x: p.x, y: p.y, yaw };
    samples++;
    const now = performance.now();
    if (now - lastPrint < 200) return;
    lastPrint = now;
    const dx = p.x - first.x, dy = p.y - first.y;
    const forward = Math.cos(first.yaw) * dx + Math.sin(first.yaw) * dy;
    const left = -Math.sin(first.yaw) * dx + Math.cos(first.yaw) * dy;
    const angle = Math.atan2(Math.sin(yaw - first.yaw), Math.cos(yaw - first.yaw));
    process.stderr.write(`[MotionDebug] t=${((now-started)/1000).toFixed(2)}s phase=${phase} ` +
      `forward=${forward.toFixed(3)}m left=${left.toFixed(3)}m yaw_delta=${(angle*180/Math.PI).toFixed(1)}deg z=${p.z.toFixed(3)}m\n`);
  });
  process.stderr.write(`[MotionDebug] pose=${topic}; displacement in initial body frame; left positive\n`);
  return {
    phase(name: string) { phase = name; },
    close() {
      sub.unsubscribe();
      cmdSub.unsubscribe();
      process.stderr.write(`[MotionDebug] samples=${samples}${samples ? "" : " NO POSE RECEIVED: motion cannot be verified"}\n`);
    },
  };
}
