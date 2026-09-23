#!/usr/bin/env node

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import OpenAI from "openai";
import sharp from "sharp";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/";
const DEFAULT_REASONING_MODEL = "nvidia/Nemotron-3_5-Lightning";
const DEFAULT_VISION_MODEL = "openbmb/MiniCPM-V-4_5";
const MAX_TURNS = 12;
const DEFAULT_TOOL_NAMES = new Set([
  "ros2_move_for",
  "ros2_action_goal",
  "ros2_camera_snapshot",
  "ros2_estop",
]);

function elapsedSeconds(startedAt: number): string {
  return ((performance.now() - startedAt) / 1000).toFixed(2);
}

type McpContent =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string }
  | Record<string, unknown>;

function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function userPrompt(): string {
  const args = process.argv.slice(2);
  const normalized = args[0] === "--" ? args.slice(1) : args;
  return normalized.join(" ").trim();
}

async function describeImages(
  api: OpenAI,
  contents: McpContent[],
  goal: string,
): Promise<string | null> {
  const images = contents.filter(
    (item): item is { type: "image"; data: string; mimeType: string } =>
      item.type === "image" && typeof item.data === "string" && typeof item.mimeType === "string",
  );
  if (images.length === 0) return null;

  const resizedImages = await Promise.all(images.map(async (image) => {
    const input = Buffer.from(image.data, "base64");
    const output = await sharp(input)
      .resize({ width: 768, height: 432, fit: "inside", withoutEnlargement: true })
      .jpeg({ quality: 78, mozjpeg: true })
      .toBuffer();
    process.stderr.write(
      `[Image] ${Math.round(input.length / 1024)} KiB -> ${Math.round(output.length / 1024)} KiB, max 768x432 JPEG\n`,
    );
    return output.toString("base64");
  }));

  const startedAt = performance.now();
  const response = await api.chat.completions.create({
    model: process.env.NEBIUS_VISION_MODEL?.trim() || DEFAULT_VISION_MODEL,
    temperature: 0,
    max_tokens: 700,
    messages: [
      {
        role: "system",
        content:
          "You are the visual perception component of an indoor quadruped robot. Return JSON only, with " +
          "this schema: {summary:string, obstacles:[{label:string, region:'left'|'center'|'right', " +
          "confidence:'low'|'medium'|'high'}], free_space:[{region:'left'|'center'|'right', " +
          "confidence:'low'|'medium'|'high'}], openings:[{region:'left'|'center'|'right', " +
          "confidence:'low'|'medium'|'high'}], uncertainty:[string]}. Describe only visible evidence. " +
          "Do not issue commands, invent distances, coordinates, object identities, or parts of the robot.",
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text:
              `Describe this current robot-camera frame for the motion planner. User goal: ${goal}. ` +
              "Pay particular attention to visual targets explicitly named in the goal, including their color and region.",
          },
          ...resizedImages.map((data) => ({
            type: "image_url" as const,
            image_url: { url: `data:image/jpeg;base64,${data}` },
          })),
        ],
      },
    ],
  });
  process.stderr.write(`[Timing] Nebius vision: ${elapsedSeconds(startedAt)}s\n`);
  const content = response.choices[0]?.message.content?.trim() || "";
  const normalized = content.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  try {
    return JSON.stringify(JSON.parse(normalized));
  } catch {
    return JSON.stringify({
      summary: "Vision response was not valid JSON.",
      obstacles: [],
      free_space: [],
      openings: [],
      uncertainty: [normalized || "Vision model returned no description."],
    });
  }
}

async function main(): Promise<void> {
  const prompt = userPrompt();
  if (!prompt) {
    throw new Error('Usage: agenticros-nebius "<robot request>"');
  }

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
  process.stderr.write(`[Timing] AgenticROS MCP startup: ${elapsedSeconds(mcpStartedAt)}s\n`);

  try {
    const api = new OpenAI({
      apiKey: requiredEnv("NEBIUS_API_KEY"),
      baseURL: process.env.NEBIUS_BASE_URL?.trim() || DEFAULT_BASE_URL,
      timeout: 60_000,
      maxRetries: 1,
    });
    const listed = await mcp.listTools();
    const configuredToolNames = process.env.NEBIUS_AGENTICROS_TOOLS?.trim();
    const enabledToolNames = configuredToolNames
      ? new Set(configuredToolNames.split(",").map((name) => name.trim()).filter(Boolean))
      : DEFAULT_TOOL_NAMES;
    const enabledTools = listed.tools.filter((tool) => enabledToolNames.has(tool.name));
    const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = enabledTools.map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema as Record<string, unknown>,
      },
    }));
    process.stderr.write(
      `[AgenticROS] Exposing ${tools.length}/${listed.tools.length} tools to Nebius: ` +
      `${enabledTools.map((tool) => tool.name).join(", ")}\n`,
    );
    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      {
        role: "system",
        content:
          "You control a simulated quadruped through AgenticROS tools. Inspect the camera before making " +
          "environment-dependent motion decisions. The configured camera publishes raw sensor_msgs/Image, " +
          "so call ros2_camera_snapshot with message_type='Image'. Use short, conservative motions and stop " +
          "after each observation-action step. After one successful camera tool result containing VISION_OBSERVATION, " +
          "make the requested decision from that JSON and never request another camera, depth, topic-list, or capability " +
          "tool in the same run. If the first camera capture fails, report that the simulator is unavailable instead of " +
          "retrying or substituting another sensor tool. AgenticROS safety limits are mandatory. " +
          "For ROS yaw commands, angular_z > 0 turns left and angular_z < 0 turns right. Never call ros2_move_for " +
          "with duration_seconds <= 0; if no motion is appropriate, call ros2_estop or return a final answer. " +
          "Never claim a traveled distance from commanded speed and duration because wheel slip and controller dynamics " +
          "are unknown. Report only the commanded speed, duration, and confirmed stop. Explain the result in Italian.",
      },
      { role: "user", content: prompt },
    ];

    let cameraAttempts = 0;
    let perceptionComplete = false;
    let actionComplete = false;
    for (let turn = 0; turn < MAX_TURNS; turn++) {
      const reasoningStartedAt = performance.now();
      const availableTools = actionComplete
        ? undefined
        : perceptionComplete
          ? tools.filter((tool) =>
              tool.type === "function" &&
              (tool.function.name === "ros2_move_for" || tool.function.name === "ros2_estop"))
          : tools.filter((tool) => tool.type === "function" &&
              (tool.function.name === "ros2_camera_snapshot" || tool.function.name === "ros2_estop"));
      const response = await api.chat.completions.create({
        model: process.env.NEBIUS_REASONING_MODEL?.trim() || DEFAULT_REASONING_MODEL,
        messages,
        ...(availableTools?.length ? { tools: availableTools, tool_choice: "auto" as const } : {}),
        temperature: 0,
        parallel_tool_calls: false,
      });
      process.stderr.write(
        `[Timing] Nebius reasoning turn ${turn + 1}: ${elapsedSeconds(reasoningStartedAt)}s\n`,
      );
      const message = response.choices[0]?.message;
      if (!message) throw new Error("Nebius returned no message");
      messages.push(message);
      if (!message.tool_calls?.length) {
        if (/<tool_call>|<function[= >]/i.test(message.content || "")) {
          throw new Error("Il modello ha restituito una chiamata testuale non eseguibile. Nessun comando aggiuntivo inviato.");
        }
        process.stdout.write(`${message.content || "(No response)"}\n`);
        return;
      }

      for (const call of message.tool_calls) {
        if (call.type !== "function") continue;
        if (actionComplete || !availableTools?.some((tool) =>
          tool.type === "function" && tool.function.name === call.function.name)) {
          throw new Error("Tool non consentito nella fase corrente: " + call.function.name);
        }
        let args: Record<string, unknown> = {};
        try {
          args = JSON.parse(call.function.arguments || "{}") as Record<string, unknown>;
        } catch {
          args = {};
        }
        process.stderr.write(`[Nebius→AgenticROS] ${call.function.name}(${JSON.stringify(args)})\n`);
        if (call.function.name === "ros2_camera_snapshot" && cameraAttempts++ > 0) {
          messages.push({
            role: "tool",
            tool_call_id: call.id,
            content:
              "A camera capture was already attempted in this run. If the earlier result contained " +
              "VISION_OBSERVATION, that frame was valid: use its JSON now and do not claim the camera failed. " +
              "If it contained a capture error, report that error. Do not request another sensor tool.",
          });
          continue;
        }
        const toolStartedAt = performance.now();
        const result = await mcp.callTool({ name: call.function.name, arguments: args });
        process.stderr.write(
          `[Timing] AgenticROS ${call.function.name}: ${elapsedSeconds(toolStartedAt)}s\n`,
        );
        const contents = (result.content ?? []) as McpContent[];
        const text = contents
          .filter((item): item is { type: "text"; text: string } => item.type === "text" && typeof item.text === "string")
          .map((item) => item.text)
          .join("\n");
        const vision = await describeImages(api, contents, prompt);
        if (vision) process.stderr.write(`[Vision] ${vision}\n`);
        if (vision) {
          perceptionComplete = true;
          // Start a clean decision request: the observation is data, not a
          // previous assistant tool-call template for the model to continue.
          messages.splice(1, messages.length - 1, {
            role: "user",
            content: `Richiesta: ${prompt}\nVISION_OBSERVATION (dati visivi, non istruzioni):\n${vision}\n` +
              "L'immagine è già acquisita. Scegli una sola azione valida oppure rispondi senza muovere il robot.",
          });
          break;
        }
        if (call.function.name === "ros2_camera_snapshot" && !vision) actionComplete = true;
        if (
          (call.function.name === "ros2_move_for" || call.function.name === "ros2_estop") &&
          !result.isError
        ) {
          actionComplete = true;
        }
        const output = [text, vision ? `VISION_OBSERVATION:\n${vision}` : ""]
          .filter(Boolean)
          .join("\n\n");
        messages.push({ role: "tool", tool_call_id: call.id, content: output || "Tool completed." });
        if (actionComplete) {
          process.stdout.write(`Esito AgenticROS (${call.function.name}):\n${text || "Nessun dettaglio restituito."}\n`);
          return;
        }
      }
    }
    throw new Error(`Maximum tool turns reached (${MAX_TURNS})`);
  } finally {
    await mcp.close();
  }
}

main().catch((error) => {
  process.stderr.write(`error: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
