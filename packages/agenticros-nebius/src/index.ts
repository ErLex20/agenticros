#!/usr/bin/env node

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import OpenAI from "openai";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { GoalAgent } from "./agent.js";
import { Perception, type PerceptionMode } from "./perception.js";
import { DEFAULT_LIMITS } from "./policy.js";
import { RobotTools } from "./robot.js";
import { visionProfile, type Grounding, type VisionProfile } from "./vision.js";

const DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/";
const DEFAULT_REASONING_MODEL = "nvidia/Nemotron-3_5-Lightning";
// Grounds targets with boxes (vision.ts); MiniCPM-V-4_5 still works with column grounding.
const DEFAULT_VISION_MODEL = "moonshotai/Kimi-K2.6";

function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function envString(name: string): string | undefined {
  return process.env[name]?.trim() || undefined;
}

function envNumber(name: string, fallback: number): number {
  const raw = process.env[name]?.trim();
  if (!raw) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${name} must be a number, got "${raw}"`);
  return value;
}

/** Positive number in (0, max]; zero or negative speeds/efficiencies would break motion timing. */
function envPositive(name: string, fallback: number, max = Number.POSITIVE_INFINITY): number {
  const value = envNumber(name, fallback);
  if (!(value > 0) || value > max) throw new Error(`${name} must be in (0, ${max}], got ${value}`);
  return value;
}

/**
 * Fallback vision model (NEBIUS_FALLBACK_VISION_MODEL, default nvidia/Nemotron-3.5-Super-VL; empty:
 * none). Only a model with the same grounding as the primary one: the prompts depend on it.
 */
function fallbackVisionOptions(primary: string, vision: VisionProfile): { fallbackVisionModel?: string; fallbackVision?: VisionProfile } {
  const model = process.env.NEBIUS_FALLBACK_VISION_MODEL !== undefined
    ? envString("NEBIUS_FALLBACK_VISION_MODEL")
    : "nvidia/Nemotron-3.5-Super-VL";
  if (!model || model === primary) return {};
  const profile = visionProfile(model);
  if (profile.grounding !== vision.grounding) {
    process.stderr.write(`[Perception] fallback vision model ${model} ignored: ${profile.grounding} grounding, the primary uses ${vision.grounding}\n`);
    return {};
  }
  process.stderr.write(`[Perception] fallback vision model: ${model}\n`);
  return { fallbackVisionModel: model, fallbackVision: profile };
}

function envChoice<T extends string>(name: string, allowed: readonly T[]): T | undefined {
  const raw = envString(name);
  if (raw === undefined) return undefined;
  const value = allowed.find((a) => a === raw.toLowerCase());
  if (!value) throw new Error(`${name} must be one of ${allowed.join(", ")}, got "${raw}"`);
  return value;
}

function userPrompt(): string {
  const args = process.argv.slice(2);
  const normalized = args[0] === "--" ? args.slice(1) : args;
  return normalized.join(" ").trim();
}

async function main(): Promise<void> {
  const prompt = userPrompt();
  if (!prompt) {
    throw new Error('Usage: agenticros-nebius "<robot request>"');
  }

  const api = new OpenAI({
    apiKey: requiredEnv("NEBIUS_API_KEY"),
    baseURL: envString("NEBIUS_BASE_URL") ?? DEFAULT_BASE_URL,
    timeout: 60_000,
    maxRetries: 1,
  });

  const here = dirname(fileURLToPath(import.meta.url));
  const mcpServer = resolve(here, "../../agenticros-claude-code/dist/index.js");
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [mcpServer],
    env: { ...process.env } as Record<string, string>,
    stderr: "inherit",
  });
  const mcp = new Client({ name: "agenticros-nebius", version: "0.0.1" });
  const mcpStartedAt = performance.now();
  await mcp.connect(transport);
  process.stderr.write(`[Timing] AgenticROS MCP startup: ${((performance.now() - mcpStartedAt) / 1000).toFixed(2)}s\n`);

  const robot = new RobotTools(mcp, {
    cameraTopic: envString("NEBIUS_CAMERA_TOPIC"),
    cameraMessageType: envString("NEBIUS_CAMERA_MESSAGE_TYPE") === "CompressedImage" ? "CompressedImage" : "Image",
    depthTopic: envString("NEBIUS_DEPTH_TOPIC"),
    maxLinear: envPositive("NEBIUS_MAX_LINEAR", 0.25),
    maxAngular: envPositive("NEBIUS_MAX_ANGULAR", 0.5),
    turnEfficiency: envPositive("NEBIUS_TURN_EFFICIENCY", 1.0, 2),
    forwardEfficiency: envPositive("NEBIUS_FORWARD_EFFICIENCY", 1.0, 2),
    imuTopic: envString("NEBIUS_IMU_TOPIC"),
    exploreAction: envString("NEBIUS_EXPLORE_ACTION"),
    nav2Motion: (envChoice("NEBIUS_NAV2_MOTION", ["on", "off"] as const) ?? "off") === "on",
  });
  process.stderr.write(
    robot.opts.imuTopic
      ? `[Heading] IMU gyroscope on ${robot.opts.imuTopic}\n`
      : "[Heading] NEBIUS_IMU_TOPIC not set: the heading is the sum of the commanded turns\n",
  );
  if (!robot.opts.depthTopic) {
    process.stderr.write(
      "[AgenticROS] NEBIUS_DEPTH_TOPIC not set: no obstacle or target distance; advances are limited to short creeps\n",
    );
  }
  // YOLO confidence trusted without a vision-model check and without a second sighting.
  const strongConfidence = envNumber("NEBIUS_VERIFY_BELOW_CONFIDENCE", DEFAULT_LIMITS.strongConfidence);
  const perceptionMode: PerceptionMode = envChoice("NEBIUS_PERCEPTION", ["yolo", "vlm"] as const) ?? "vlm";
  const visionModel = envString("NEBIUS_VISION_MODEL") ?? DEFAULT_VISION_MODEL;
  const vision = visionProfile(visionModel, envChoice<Grounding>("NEBIUS_VISION_GROUNDING", ["box", "columns"] as const));
  process.stderr.write(
    `[Perception] mode=${perceptionMode} vision=${visionModel} grounding=${vision.grounding} width=${vision.width}\n`,
  );
  if (perceptionMode === "vlm" && vision.grounding === "columns") {
    process.stderr.write("[Perception] warning: without YOLO a column-grounding vision model localizes coarsely (±9 deg)\n");
  }
  // Without a map, the free-space rule picks where to explore by default. With "vision" the
  // vision model picks it from the views of each turn (box-grounding models only): better
  // offline (58-65% of the walks toward the target's room against 38%), but in 16 closed-loop
  // runs in the house both reached a west-wing target 1 time in 5 (parent docs/ESPLORAZIONE_GRAFO.md).
  const requestedChoice = envChoice("NEBIUS_EXPLORE_CHOICE", ["vision", "open"] as const) ?? "open";
  const exploreChoice = requestedChoice === "vision" && vision.grounding !== "box" ? "open" : requestedChoice;
  process.stderr.write(`[Explore] choice=${exploreChoice}\n`);
  // Obstacles lower than the camera, from a 3D corridor of the depth image (NEBIUS_DEPTH_CORRIDOR=off: band only).
  const depthCorridor = (envChoice("NEBIUS_DEPTH_CORRIDOR", ["on", "off"] as const) ?? "on") === "on" && !!robot.opts.depthTopic;
  const hfovDeg = envPositive("NEBIUS_CAMERA_HFOV_DEG", 110, 179);
  const perception = new Perception(api, {
    vision,
    visionModel,
    hfovDeg,
    minConfidence: envNumber("NEBIUS_DETECTION_MIN_CONFIDENCE", 0.35),
    verifyBelowConfidence: strongConfidence,
    ...fallbackVisionOptions(visionModel, vision),
  });
  const agent = new GoalAgent(api, robot, perception, {
    reasoningModel: envString("NEBIUS_REASONING_MODEL") ?? DEFAULT_REASONING_MODEL,
    maxSteps: envPositive("NEBIUS_MAX_STEPS", 80),
    maxSeconds: envPositive("NEBIUS_MAX_SECONDS", 600),
    policy: envString("NEBIUS_POLICY") === "auto" ? "auto" : "llm",
    perception: perceptionMode,
    exploreChoice,
    nav2Walk: (envChoice("NEBIUS_NAV2_WALK", ["on", "off"] as const) ?? "on") === "on",
    nav2Approach: (envChoice("NEBIUS_NAV2_APPROACH", ["on", "off"] as const) ?? "on") === "on",
    sessionPlaces: (envChoice("NEBIUS_SESSION_PLACES", ["on", "off"] as const) ?? "off") === "on",
    decisionTimeoutS: envPositive("NEBIUS_DECISION_TIMEOUT_S", 10),
    fallbackReasoningModel:
      process.env.NEBIUS_FALLBACK_REASONING_MODEL !== undefined
        ? envString("NEBIUS_FALLBACK_REASONING_MODEL")
        : "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    depthCorridor,
    camera: {
      heightM: envPositive("NEBIUS_CAMERA_HEIGHT_M", 0.385, 3),
      hfovDeg,
      aspect: 16 / 9,
    },
    limits: {
      ...DEFAULT_LIMITS,
      stopDistanceM: envPositive("NEBIUS_STOP_DISTANCE_M", DEFAULT_LIMITS.stopDistanceM),
      strongConfidence,
    },
  });

  // Ctrl-C must never leave the robot moving: block further motion commands,
  // stop, then exit. The MCP server also stops the base when it gets the same signal.
  let interrupted = false;
  const interrupt = (signal: NodeJS.Signals) => {
    if (interrupted) process.exit(130);
    interrupted = true;
    robot.aborted = true;
    process.stderr.write(`\n[AgenticROS] ${signal}: stopping the robot\n`);
    robot
      .estop()
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        // A terminal Ctrl-C reaches the MCP server too: it stops the robot (and cancels Nav2
        // goals) on its own and closes the connection before answering this estop.
        process.stderr.write(
          /connection closed/i.test(message)
            ? "[AgenticROS] robot stopped by the MCP server on the same signal\n"
            : `[AgenticROS] estop failed: ${message}\n`,
        );
      })
      .finally(() => process.exit(130));
  };
  process.on("SIGINT", interrupt);
  process.on("SIGTERM", interrupt);

  try {
    const answer = await agent.run(prompt);
    process.stdout.write(`${answer}\n`);
  } finally {
    await robot.estop().catch((error: unknown) => {
      process.stderr.write(`[AgenticROS] estop failed: ${error instanceof Error ? error.message : String(error)}\n`);
    });
    await perception.dispose();
    await mcp.close();
  }
}

main().catch((error) => {
  process.stderr.write(`error: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
