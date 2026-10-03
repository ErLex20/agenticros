/**
 * Target perception for the goal loop.
 *
 * COCO targets use the local YOLOv8n detector of @agenticros/object-detection
 * with a tiled pass (precise bearing and bbox, ~1 s on CPU). Other targets,
 * and attribute checks such as colour, use the Nebius vision model on a frame
 * overlaid with numbered columns, which it localizes far more reliably than
 * free-form pixel boxes.
 */

import type OpenAI from "openai";
import sharp from "sharp";
import { PersonDetector, detectClassTiled, resolveCocoClassId } from "@agenticros/object-detection";
import { bearingFromImageX, round, type BBoxNorm, type TargetObservation } from "./policy.js";

const GRID_COLUMNS = 9;
/** Detections checked against the target's attributes per frame, highest confidence first. */
const MAX_CANDIDATES = 3;
const VLM_MAX_WIDTH = 768;
const VLM_MAX_HEIGHT = 432;
const SIZE_TO_HEIGHT_FRAC: Record<string, number> = {
  tiny: 0.05,
  small: 0.12,
  medium: 0.25,
  large: 0.45,
  very_large: 0.7,
};

export interface TargetSpec {
  description: string;
  cocoClass: string | null;
  /** Visual attributes YOLO cannot check (e.g. colour); verified once with the vision model. */
  attributes: string[];
}

export interface PerceptionOptions {
  visionModel: string;
  hfovDeg: number;
  minConfidence: number;
}

export interface SceneResult {
  summary: string;
  target: TargetObservation | null;
}

/** Extract the JSON object from a model reply that may include <think> blocks, fences or prose. */
export function parseJsonObject(text: string): Record<string, unknown> | null {
  const cleaned = text
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/```(?:json)?/gi, "")
    .trim();
  const end = cleaned.lastIndexOf("}");
  if (end < 0) return null;
  // Try the widest candidate first, then later opening braces (prose before the JSON may contain braces).
  for (let start = cleaned.indexOf("{"); start >= 0 && start < end; start = cleaned.indexOf("{", start + 1)) {
    try {
      const parsed: unknown = JSON.parse(cleaned.slice(start, end + 1));
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed as Record<string, unknown>;
    } catch {
      // try the next opening brace
    }
  }
  return null;
}

async function toVisionJpeg(image: Buffer, grid: boolean): Promise<string> {
  let pipeline = sharp(image).resize({
    width: VLM_MAX_WIDTH,
    height: VLM_MAX_HEIGHT,
    fit: "inside",
    withoutEnlargement: true,
  });
  if (grid) {
    const { data, info } = await pipeline.raw().toBuffer({ resolveWithObject: true });
    const w = info.width;
    const h = info.height;
    const cols = Array.from({ length: GRID_COLUMNS }, (_, i) => {
      const x = (i * w) / GRID_COLUMNS;
      const line = i > 0 ? `<line x1="${x}" y1="0" x2="${x}" y2="${h}" stroke="#00a0ff" stroke-width="2"/>` : "";
      const label = `<text x="${x + w / GRID_COLUMNS / 2}" y="26" font-size="22" font-family="sans-serif" font-weight="bold" fill="#0050ff" text-anchor="middle">${i + 1}</text>`;
      return line + label;
    }).join("");
    const svg = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">${cols}</svg>`);
    pipeline = sharp(data, { raw: { width: w, height: h, channels: info.channels } }).composite([{ input: svg }]);
  }
  const jpeg = await pipeline.jpeg({ quality: 80, mozjpeg: true }).toBuffer();
  return `data:image/jpeg;base64,${jpeg.toString("base64")}`;
}

export class Perception {
  private detector: PersonDetector | null = null;
  /** Set once a crop matched the attributes; reused only while a single candidate is in view. */
  private verifiedOnce = false;

  constructor(
    private readonly api: OpenAI,
    private readonly opts: PerceptionOptions,
  ) {}

  /** Visible detections of the target's COCO class, highest confidence first (at most MAX_CANDIDATES). */
  async detectCoco(image: Buffer, target: TargetSpec): Promise<TargetObservation[]> {
    const classId = target.cocoClass ? resolveCocoClassId(target.cocoClass) : null;
    if (classId === null) return [];
    this.detector ??= new PersonDetector({ scoreThreshold: this.opts.minConfidence });
    const result = await detectClassTiled(this.detector, image, classId);
    return result.detections.slice(0, MAX_CANDIDATES).map((d) => ({
      visible: true,
      source: "yolo" as const,
      confidence: round(d.confidence),
      bearing_deg: round(bearingFromImageX(d.cx / result.width, this.opts.hfovDeg), 1),
      bbox: {
        x_min: d.x / result.width,
        y_min: d.y / result.height,
        x_max: (d.x + d.width) / result.width,
        y_max: (d.y + d.height) / result.height,
      },
      height_frac: round(d.height / result.height),
    }));
  }

  /** Vision-model check that a detected crop matches attributes YOLO cannot see (colour, material...). */
  async verifyAttributes(image: Buffer, bbox: BBoxNorm, target: TargetSpec, candidates: number): Promise<boolean> {
    if (target.attributes.length === 0) return true;
    // With a single candidate in view after a positive check, assume it is the same object.
    if (this.verifiedOnce && candidates === 1) return true;
    const meta = await sharp(image).metadata();
    const w = meta.width ?? 0;
    const h = meta.height ?? 0;
    const padX = Math.max(48, (bbox.x_max - bbox.x_min) * w * 0.3);
    const padY = Math.max(48, (bbox.y_max - bbox.y_min) * h * 0.3);
    const left = Math.max(0, Math.floor(bbox.x_min * w - padX));
    const top = Math.max(0, Math.floor(bbox.y_min * h - padY));
    const width = Math.max(1, Math.min(w - left, Math.ceil((bbox.x_max - bbox.x_min) * w + 2 * padX)));
    const height = Math.max(1, Math.min(h - top, Math.ceil((bbox.y_max - bbox.y_min) * h + 2 * padY)));
    const crop = await sharp(image).extract({ left, top, width, height }).resize({ width: 384, height: 384, fit: "inside" }).jpeg({ quality: 85 }).toBuffer();
    const response = await this.api.chat.completions.create({
      model: this.opts.visionModel,
      temperature: 0,
      max_tokens: 120,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text:
                `Does the main object in this crop match "${target.description}" ` +
                `(attributes to check: ${target.attributes.join(", ")})? Return JSON only: {"match": boolean, "reason": string}`,
            },
            { type: "image_url", image_url: { url: `data:image/jpeg;base64,${crop.toString("base64")}` } },
          ],
        },
      ],
    });
    const parsed = parseJsonObject(response.choices[0]?.message.content ?? "");
    const match = parsed?.["match"] === true;
    process.stderr.write(`[Verify] ${target.description}: ${match ? "match" : "no match"} ${String(parsed?.["reason"] ?? "")}\n`);
    if (match) this.verifiedOnce = true;
    return match;
  }

  /**
   * Scene summary plus coarse target localization from the vision model. The
   * frame carries numbered columns so the model reports a column, not pixels.
   */
  async describe(image: Buffer, goal: string, target: TargetSpec | null): Promise<SceneResult> {
    const url = await toVisionJpeg(image, target !== null);
    const targetSchema = target
      ? `, "target": {"visible": boolean, "column": integer 1-${GRID_COLUMNS} or null, "apparent_size": "tiny"|"small"|"medium"|"large"|"very_large"|null}`
      : "";
    const response = await this.api.chat.completions.create({
      model: this.opts.visionModel,
      temperature: 0,
      max_tokens: 400,
      messages: [
        {
          role: "system",
          content:
            "You are the visual perception component of an indoor quadruped robot. Describe only visible evidence. " +
            "Do not issue commands or invent distances. Return JSON only.",
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text:
                `Robot goal: ${goal}. ` +
                (target
                  ? `The image is split by blue lines into ${GRID_COLUMNS} numbered columns (1 = far left, ${GRID_COLUMNS} = far right). ` +
                    `Look for: "${target.description}". Report the column containing its centre, or visible=false if it is not in view. `
                  : "") +
                `Schema: {"summary": string (one or two sentences: layout, obstacles, free space, openings)${targetSchema}}`,
            },
            { type: "image_url", image_url: { url } },
          ],
        },
      ],
    });
    const parsed = parseJsonObject(response.choices[0]?.message.content ?? "");
    const summary = typeof parsed?.["summary"] === "string" ? parsed["summary"] : "Vision response was not valid JSON.";
    if (!target) return { summary, target: null };

    const t = (parsed?.["target"] ?? {}) as Record<string, unknown>;
    const column = Number(t["column"]);
    if (t["visible"] !== true || !Number.isInteger(column) || column < 1 || column > GRID_COLUMNS) {
      return { summary, target: { visible: false, source: "vlm" } };
    }
    const xMin = (column - 1) / GRID_COLUMNS;
    const xMax = column / GRID_COLUMNS;
    const heightFrac = SIZE_TO_HEIGHT_FRAC[String(t["apparent_size"])] ?? 0.12;
    return {
      summary,
      target: {
        visible: true,
        source: "vlm",
        bearing_deg: round(bearingFromImageX((xMin + xMax) / 2, this.opts.hfovDeg), 1),
        // Column band around the horizon: excludes the floor right in front of the robot.
        bbox: { x_min: xMin, y_min: 0.3, x_max: xMax, y_max: 0.55 },
        height_frac: heightFrac,
      },
    };
  }

  async dispose(): Promise<void> {
    await this.detector?.dispose().catch(() => {});
  }
}
