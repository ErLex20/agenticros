/**
 * Pure helpers for the Nebius vision model: per-model profiles, robust JSON
 * extraction and box normalization. No network, no image decoding, so they
 * can be unit-tested offline.
 */

import type { BBoxNorm } from "./policy.js";

/**
 * How a vision model localizes the target.
 * - "box": the model returns a bounding box (fractions of the image), measured
 *   in simulation as accurate as a detector box (bearing error ~0.2 deg).
 * - "columns": the frame carries numbered columns and the model returns one;
 *   for models that cannot ground (MiniCPM-V boxes have median IoU 0.13).
 */
export type Grounding = "box" | "columns";

export interface VisionProfile {
  grounding: Grounding;
  /** Width of the JPEG sent to the model; the height keeps the aspect ratio. */
  width: number;
  /** Provider-specific body fields, e.g. disabling reasoning (Kimi thinks by default). */
  extraBody: Record<string, unknown>;
  /** The endpoint honours response_format json_schema for this model. */
  structuredOutput: boolean;
}

const PROFILES: Record<string, VisionProfile> = {
  "moonshotai/Kimi-K2.6": {
    grounding: "box",
    width: 1024,
    extraBody: { chat_template_kwargs: { thinking: false } },
    structuredOutput: true,
  },
  "openbmb/MiniCPM-V-4_5": { grounding: "columns", width: 768, extraBody: {}, structuredOutput: false },
  // NVIDIA open model. Reasons by default (~270 tokens, 2 s per frame): enable_thinking=false
  // replies in ~1 s with the same box. Boxes come on a 0-1000 grid even when asked for
  // fractions; normalizeBox detects the scale.
  "nvidia/Nemotron-3.5-Super-VL": {
    grounding: "box",
    width: 1024,
    extraBody: { chat_template_kwargs: { enable_thinking: false } },
    structuredOutput: false,
  },
};

/** Unknown models get the conservative column grounding unless overridden. */
export function visionProfile(model: string, grounding?: Grounding): VisionProfile {
  const base = PROFILES[model] ?? { grounding: "columns", width: 768, extraBody: {}, structuredOutput: false };
  return grounding ? { ...base, grounding } : base;
}

/**
 * First complete JSON object in a model reply that may include <think> blocks,
 * code fences, prose or a stray trailing brace (MiniCPM-V often ends compact
 * JSON with "}}}", which a first-brace-to-last-brace slice cannot parse).
 */
export function parseJsonObject(text: string): Record<string, unknown> | null {
  const cleaned = text
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/```(?:json)?/gi, "");
  for (let start = cleaned.indexOf("{"); start >= 0; start = cleaned.indexOf("{", start + 1)) {
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let i = start; i < cleaned.length; i++) {
      const ch = cleaned[i];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === "{") depth++;
      else if (ch === "}" && --depth === 0) {
        try {
          const parsed: unknown = JSON.parse(cleaned.slice(start, i + 1));
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed as Record<string, unknown>;
        } catch {
          // malformed candidate: try the next opening brace
        }
        break;
      }
    }
  }
  return null;
}

/** Smallest box side kept, as a fraction of the image: thinner boxes are noise. */
const MIN_BOX_SIDE = 0.004;

/**
 * Normalize a model box to fractions of the image. The prompt asks for
 * fractions; a reply in the 0-1000 convention (common in grounding models)
 * is accepted too. Corners are sorted and clamped; degenerate boxes are null.
 */
export function normalizeBox(raw: unknown): BBoxNorm | null {
  if (!Array.isArray(raw) || raw.length !== 4) return null;
  const v = raw.map(Number);
  if (v.some((x) => !Number.isFinite(x) || x < 0)) return null;
  const max = Math.max(...v);
  const scale = max <= 1 ? 1 : max <= 1000 ? 1000 : null;
  if (scale === null) return null;
  const [a, b, c, d] = v.map((x) => Math.min(1, x / scale)) as [number, number, number, number];
  const box = { x_min: Math.min(a, c), y_min: Math.min(b, d), x_max: Math.max(a, c), y_max: Math.max(b, d) };
  if (box.x_max - box.x_min < MIN_BOX_SIDE || box.y_max - box.y_min < MIN_BOX_SIDE) return null;
  return box;
}

/** JSON schema of the box-grounding reply (response_format json_schema). */
export function boxReplySchema(withTarget: boolean, withWaypoint = false): Record<string, unknown> {
  const box = { type: "array", items: { type: "number" }, minItems: 4, maxItems: 4 };
  const properties: Record<string, unknown> = { summary: { type: "string" } };
  const required = ["summary"];
  if (withTarget) {
    properties["target"] = {
      type: "object",
      properties: { visible: { type: "boolean" }, bbox: { anyOf: [box, { type: "null" }] } },
      required: ["visible", "bbox"],
      additionalProperties: false,
    };
    properties["doorways"] = { type: "array", items: box };
    required.push("target", "doorways");
  }
  if (withWaypoint) {
    properties["waypoint"] = {
      type: "object",
      properties: { visible: { type: "boolean" }, bbox: { anyOf: [box, { type: "null" }] } },
      required: ["visible", "bbox"],
      additionalProperties: false,
    };
    required.push("waypoint");
  }
  return { type: "object", properties, required, additionalProperties: false };
}
