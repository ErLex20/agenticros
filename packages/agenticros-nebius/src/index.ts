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
import { visionProfile, type Grounding } from "./vision.js";

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
  });
  if (!robot.opts.depthTopic) {
    process.stderr.write(
      "[AgenticROS] NEBIUS_DEPTH_TOPIC not set: no obstacle or target distance; advances are limited to short creeps\n",
    );
  }
  // YOLO confidence trusted without a vision-model check and without a second sighting.
  const strongConfidence = envNumber("NEBIUS_VERIFY_BELOW_CONFIDENCE", DEFAULT_LIMITS.strongConfidence);
  const perceptionMode: PerceptionMode = envChoice("NEBIUS_PERCEPTION", ["yolo", "vlm"] as const) ?? "yolo";
  const visionModel = envString("NEBIUS_VISION_MODEL") ?? DEFAULT_VISION_MODEL;
  const vision = visionProfile(visionModel, envChoice<Grounding>("NEBIUS_VISION_GROUNDING", ["box", "columns"] as const));
  process.stderr.write(
    `[Perception] mode=${perceptionMode} vision=${visionModel} grounding=${vision.grounding} width=${vision.width}\n`,
  );
  if (perceptionMode === "vlm" && vision.grounding === "columns") {
    process.stderr.write("[Perception] warning: without YOLO a column-grounding vision model localizes coarsely (±9 deg)\n");
  }
  const hfovDeg = envPositive("NEBIUS_CAMERA_HFOV_DEG", 110, 179);
  const perception = new Perception(api, {
    vision,
    visionModel,
    hfovDeg,
    minConfidence: envNumber("NEBIUS_DETECTION_MIN_CONFIDENCE", 0.35),
    verifyBelowConfidence: strongConfidence,
  });
  const agent = new GoalAgent(api, robot, perception, {
    reasoningModel: envString("NEBIUS_REASONING_MODEL") ?? DEFAULT_REASONING_MODEL,
    maxSteps: envPositive("NEBIUS_MAX_STEPS", 80),
    maxSeconds: envPositive("NEBIUS_MAX_SECONDS", 600),
    policy: envString("NEBIUS_POLICY") === "auto" ? "auto" : "llm",
    perception: perceptionMode,
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
        process.stderr.write(`[AgenticROS] estop failed: ${error instanceof Error ? error.message : String(error)}\n`);
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
