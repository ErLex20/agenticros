/**
 * Target perception for the goal loop.
 *
 * Two modes (NEBIUS_PERCEPTION):
 * - "vlm" (default): no local detector. The Nebius vision model localizes every target.
 * - "yolo": COCO targets use the local YOLO detector of @agenticros/object-detection
 *   with a tiled pass; other targets, COCO targets YOLO misses and attribute checks
 *   use the Nebius vision model.
 *
 * How the vision model localizes depends on its profile (vision.ts): models that
 * ground (Kimi-K2.6) return a bounding box; the others get a frame overlaid with
 * numbered columns and return a column.
 */

import type OpenAI from "openai";
import sharp from "sharp";
import { PersonDetector, detectClassTiled, resolveCocoClassId } from "@agenticros/object-detection";
import { bearingFromImageX, round, type BBoxNorm, type TargetObservation } from "./policy.js";
import { boxReplySchema, normalizeBox, parseJsonObject, type VisionProfile } from "./vision.js";

export { parseJsonObject } from "./vision.js";

const GRID_COLUMNS = 9;
/** Detections checked against the target's attributes per frame, highest confidence first. */
const MAX_CANDIDATES = 3;
/** More doorways than this in one frame are treated as noise. */
const MAX_DOORWAYS_PER_FRAME = 3;
/** One vision call; a slow reply is retried once, then the observation is reported as failed. */
const VISION_TIMEOUT_MS = 20_000;
const SIZE_TO_HEIGHT_FRAC: Record<string, number> = {
  tiny: 0.05,
  small: 0.12,
  medium: 0.25,
  large: 0.45,
  very_large: 0.7,
};

export type PerceptionMode = "yolo" | "vlm";

export interface TargetSpec {
  description: string;
  cocoClass: string | null;
  /** Visual attributes YOLO cannot check (e.g. colour); verified once with the vision model. */
  attributes: string[];
}

export interface PerceptionOptions {
  visionModel: string;
  vision: VisionProfile;
  hfovDeg: number;
  minConfidence: number;
  /** YOLO detections below this confidence are checked by the vision model on a crop. */
  verifyBelowConfidence: number;
}

/** One view of a search turn, for choosing where to explore next. */
export interface TurnView {
  image: Buffer;
  /** Heading of the view relative to the robot's current heading (deg, positive = left). */
  relativeDeg: number;
  /** The robot already walked this way from here, or came from it. */
  explored: boolean;
}

export interface ExploreChoice {
  view: number;
  /** Horizontal position in the chosen view to walk toward (fraction of its width, 0 = left). */
  x: number;
  reason: string;
}

/**
 * Crop check reply: the object is named before the verdict. Asked for the verdict first, the
 * model confirmed 46 of 131 look-alike crops of the house (34 of 49 tables as a desk); naming
 * it first, with parts of the target accepted, 19 of 131, at 35 of 45 targets against 38
 * (parent scripts/vision_eval/verify_eval.py). A sofa still passes for a bed about 1 time in 3.
 */
const VERIFY_SCHEMA = {
  type: "object",
  properties: { object: { type: "string" }, reason: { type: "string" }, match: { type: "boolean" } },
  required: ["object", "reason", "match"],
  additionalProperties: false,
};

const EXPLORE_TILE_W = 768;
const EXPLORE_TILE_H = 432;
const EXPLORE_SCHEMA = {
  type: "object",
  properties: { reason: { type: "string" }, view: { type: "integer" }, x: { type: "number" } },
  required: ["reason", "view", "x"],
  additionalProperties: false,
};

/** The views of a turn in one JPEG, two per row, each labelled with its number and heading. */
async function turnMosaic(views: TurnView[]): Promise<string> {
  const cols = 2;
  const rows = Math.ceil(views.length / cols);
  const tiles = await Promise.all(
    views.map(async (v, k) => {
      const deg = Math.round(v.relativeDeg);
      const label = `view ${k}: ${deg >= 0 ? "+" : ""}${deg} deg${v.explored ? " (explored)" : ""}`;
      const svg = Buffer.from(
        `<svg xmlns="http://www.w3.org/2000/svg" width="${EXPLORE_TILE_W}" height="${EXPLORE_TILE_H}">` +
          `<rect x="0" y="0" width="${label.length * 19 + 20}" height="50" fill="black"/>` +
          `<text x="10" y="37" font-size="32" font-family="sans-serif" font-weight="bold" fill="#ffff00">${label}</text></svg>`,
      );
      const input = await sharp(v.image)
        .resize({ width: EXPLORE_TILE_W, height: EXPLORE_TILE_H, fit: "fill" })
        .composite([{ input: svg }])
        .png()
        .toBuffer();
      return { input, left: (k % cols) * EXPLORE_TILE_W, top: Math.floor(k / cols) * EXPLORE_TILE_H };
    }),
  );
  const jpeg = await sharp({
    create: { width: cols * EXPLORE_TILE_W, height: rows * EXPLORE_TILE_H, channels: 3, background: { r: 0, g: 0, b: 0 } },
  })
    .composite(tiles)
    .jpeg({ quality: 80, mozjpeg: true })
    .toBuffer();
  return `data:image/jpeg;base64,${jpeg.toString("base64")}`;
}

export interface SceneResult {
  summary: string;
  target: TargetObservation | null;
  /** Bearings (deg, positive = left) of doorways or passages to other rooms; reported only when a target is given. */
  doorwayBearingsDeg: number[];
}

async function toVisionJpeg(image: Buffer, width: number, columns: boolean): Promise<string> {
  let pipeline = sharp(image).resize({ width, height: Math.round((width * 9) / 16), fit: "inside", withoutEnlargement: true });
  if (columns) {
    const { data, info } = await pipeline.raw().toBuffer({ resolveWithObject: true });
    const w = info.width;
    const h = info.height;
    const s = w / 768;
    const cols = Array.from({ length: GRID_COLUMNS }, (_, i) => {
      const x = (i * w) / GRID_COLUMNS;
      const line = i > 0 ? `<line x1="${x}" y1="0" x2="${x}" y2="${h}" stroke="#00a0ff" stroke-width="${2 * s}"/>` : "";
      const label = `<text x="${x + w / GRID_COLUMNS / 2}" y="${26 * s}" font-size="${22 * s}" font-family="sans-serif" font-weight="bold" fill="#0050ff" text-anchor="middle">${i + 1}</text>`;
      return line + label;
    }).join("");
    const svg = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">${cols}</svg>`);
    pipeline = sharp(data, { raw: { width: w, height: h, channels: info.channels } }).composite([{ input: svg }]);
  }
  const jpeg = await pipeline.jpeg({ quality: 80, mozjpeg: true }).toBuffer();
  return `data:image/jpeg;base64,${jpeg.toString("base64")}`;
}

function targetText(target: TargetSpec): string {
  return target.attributes.length > 0 ? `"${target.description}" (it must be: ${target.attributes.join(", ")})` : `"${target.description}"`;
}

export class Perception {
  private detector: PersonDetector | null = null;
  /** Set once a crop matched the attributes; reused only while a single candidate is in view. */
  private verifiedOnce = false;

  constructor(
    private readonly api: OpenAI,
    private readonly opts: PerceptionOptions,
  ) {}

  /** One vision-model call with the profile's provider fields, a bounded timeout and one retry. */
  private async ask(
    messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
    maxTokens: number,
    schema?: Record<string, unknown>,
  ): Promise<string> {
    const body = {
      model: this.opts.visionModel,
      temperature: 0,
      max_tokens: maxTokens,
      messages,
      ...(schema && this.opts.vision.structuredOutput
        ? { response_format: { type: "json_schema" as const, json_schema: { name: "observation", schema, strict: true } } }
        : {}),
      ...this.opts.vision.extraBody,
    } as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming;
    const startedAt = performance.now();
    const response = await this.api.chat.completions.create(body, { timeout: VISION_TIMEOUT_MS, maxRetries: 1 });
    process.stderr.write(`[Timing] Nebius vision: ${((performance.now() - startedAt) / 1000).toFixed(2)}s\n`);
    return response.choices[0]?.message.content ?? "";
  }

  /** Visible detections of the target's COCO class, highest confidence first (at most MAX_CANDIDATES). */
  async detectCoco(image: Buffer, target: TargetSpec): Promise<TargetObservation[]> {
    const classId = target.cocoClass ? resolveCocoClassId(target.cocoClass) : null;
    if (classId === null) return [];
    this.detector ??= new PersonDetector({ scoreThreshold: this.opts.minConfidence });
    const result = await detectClassTiled(this.detector, image, classId);
    return result.detections.slice(0, MAX_CANDIDATES).map((d) => ({
      visible: true,
      source: "yolo" as const,
      localization: "box" as const,
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

  /**
   * Vision-model check on a crop of a YOLO detection: attributes YOLO cannot see
   * (colour, material...) and, for low-confidence detections, the object class itself.
   * In simulation YOLO often reports rendered furniture as the wrong class (a white sofa as a bed).
   */
  async verifyCandidate(image: Buffer, candidate: TargetObservation, target: TargetSpec, candidates: number, force = false): Promise<boolean> {
    const bbox = candidate.bbox;
    if (!bbox) return false;
    const lowConfidence = (candidate.confidence ?? 0) < this.opts.verifyBelowConfidence;
    if (!force && target.attributes.length === 0 && !lowConfidence) return true;
    // With a single confident candidate in view after a positive check, assume it is the same object.
    if (!force && !lowConfidence && this.verifiedOnce && candidates === 1) return true;
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
    const content = await this.ask(
      [
        {
          role: "user",
          content: [
            {
              type: "text",
              text:
                "Name the main object in this crop with its most specific common name, then say whether it is a " +
                `${target.description} or a part of one (a mattress is part of a bed)` +
                (target.attributes.length > 0 ? `, with these attributes: ${target.attributes.join(", ")}` : "") +
                ". A similar object is not a match (a sofa is not a bed, a coffee table is not a desk, a cabinet is not a refrigerator). " +
                'Return JSON: {"object": string, "reason": string, "match": boolean}',
            },
            { type: "image_url", image_url: { url: `data:image/jpeg;base64,${crop.toString("base64")}` } },
          ],
        },
      ],
      200,
      VERIFY_SCHEMA,
    );
    const parsed = parseJsonObject(content);
    const match = parsed?.["match"] === true;
    process.stderr.write(
      `[Verify] ${target.description} (${candidate.source}${candidate.confidence != null ? ` ${candidate.confidence}` : ""}): ${match ? "match" : "no match"} ` +
        `(${String(parsed?.["object"] ?? "?")}) ${String(parsed?.["reason"] ?? "")}\n`,
    );
    // Forced checks of vision-model sightings must not stand in for YOLO candidates later.
    if (match && !force) this.verifiedOnce = true;
    return match;
  }

  /** Scene summary plus target localization from the vision model (box or column, per profile). */
  async describe(image: Buffer, goal: string, target: TargetSpec | null): Promise<SceneResult> {
    return this.opts.vision.grounding === "box" ? this.describeBox(image, goal, target) : this.describeColumns(image, goal, target);
  }

  private async describeBox(image: Buffer, goal: string, target: TargetSpec | null): Promise<SceneResult> {
    const url = await toVisionJpeg(image, this.opts.vision.width, false);
    const schema =
      '{"summary": string (one or two sentences: layout, obstacles, free space, openings)' +
      (target
        ? ', "target": {"visible": boolean, "bbox": [x1, y1, x2, y2] or null}, "doorways": [[x1, y1, x2, y2], ...] ' +
          "(interior doorways, open doors or corridors leading to another room of the house; not windows, glass walls or doors to the outside; [] if none)"
        : "") +
      "}";
    const content = await this.ask(
      [
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
                  ? `Look for: ${targetText(target)}. Report visible=true only for that object itself, not for a similar one ` +
                    "(a sofa is not a bed, a cabinet is not a refrigerator). If several are visible, give the nearest one. " +
                    "Coordinates are fractions of the image width and height: 0.0 = left/top edge, 1.0 = right/bottom edge " +
                    "(e.g. [0.12, 0.40, 0.25, 0.61]). "
                  : "") +
                `Return JSON: ${schema}`,
            },
            { type: "image_url", image_url: { url } },
          ],
        },
      ],
      500,
      boxReplySchema(target !== null),
    );
    const parsed = parseJsonObject(content);
    const summary = typeof parsed?.["summary"] === "string" ? parsed["summary"] : "Vision response was not valid JSON.";
    if (!target) return { summary, target: null, doorwayBearingsDeg: [] };

    const centreBearing = (b: BBoxNorm) => round(bearingFromImageX((b.x_min + b.x_max) / 2, this.opts.hfovDeg), 1);
    const doorways = (Array.isArray(parsed?.["doorways"]) ? (parsed["doorways"] as unknown[]) : [])
      .map(normalizeBox)
      .filter((b): b is BBoxNorm => b !== null);
    // A frame "full of doorways" is the model guessing, not seeing: drop it.
    const doorwayBearingsDeg = doorways.length <= MAX_DOORWAYS_PER_FRAME ? doorways.map(centreBearing) : [];
    const t = (parsed?.["target"] ?? {}) as Record<string, unknown>;
    const bbox = t["visible"] === true ? normalizeBox(t["bbox"]) : null;
    if (!bbox) return { summary, target: { visible: false, source: "vlm", localization: "box" }, doorwayBearingsDeg };
    return {
      summary,
      target: {
        visible: true,
        source: "vlm",
        localization: "box",
        bearing_deg: centreBearing(bbox),
        bbox,
        height_frac: round(bbox.y_max - bbox.y_min),
      },
      doorwayBearingsDeg,
    };
  }

  /**
   * Coarse localization for models that cannot ground: the frame carries numbered
   * columns so the model reports a column, not pixels.
   */
  private async describeColumns(image: Buffer, goal: string, target: TargetSpec | null): Promise<SceneResult> {
    const url = await toVisionJpeg(image, this.opts.vision.width, target !== null);
    const targetSchema = target
      ? `, "target": {"visible": boolean, "column": integer 1-${GRID_COLUMNS} or null, "apparent_size": "tiny"|"small"|"medium"|"large"|"very_large"|null}` +
        `, "doorways": integer[] (columns 1-${GRID_COLUMNS} showing an interior doorway, open door or corridor leading to another room of the house; not windows, glass walls or doors to the outside; [] if none)`
      : "";
    const content = await this.ask(
      [
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
                    `Look for: ${targetText(target)}. Report the column containing its centre, or visible=false if it is not in view. ` +
                    "Report visible=true only for that object itself, not for a similar one (a sofa is not a bed, a cabinet is not a refrigerator). "
                  : "") +
                `Schema: {"summary": string (one or two sentences: layout, obstacles, free space, openings)${targetSchema}}`,
            },
            { type: "image_url", image_url: { url } },
          ],
        },
      ],
      400,
    );
    const parsed = parseJsonObject(content);
    const summary = typeof parsed?.["summary"] === "string" ? parsed["summary"] : "Vision response was not valid JSON.";
    if (!target) return { summary, target: null, doorwayBearingsDeg: [] };

    const columnBearing = (column: number) =>
      round(bearingFromImageX((column - 0.5) / GRID_COLUMNS, this.opts.hfovDeg), 1);
    const doorwayColumns = (Array.isArray(parsed?.["doorways"]) ? (parsed["doorways"] as unknown[]) : [])
      .map(Number)
      .filter((c) => Number.isInteger(c) && c >= 1 && c <= GRID_COLUMNS);
    const doorwayBearingsDeg = doorwayColumns.length <= MAX_DOORWAYS_PER_FRAME ? doorwayColumns.map(columnBearing) : [];
    const t = (parsed?.["target"] ?? {}) as Record<string, unknown>;
    const column = Number(t["column"]);
    if (t["visible"] !== true || !Number.isInteger(column) || column < 1 || column > GRID_COLUMNS) {
      return { summary, target: { visible: false, source: "vlm", localization: "column" }, doorwayBearingsDeg };
    }
    const xMin = (column - 1) / GRID_COLUMNS;
    const xMax = column / GRID_COLUMNS;
    const heightFrac = SIZE_TO_HEIGHT_FRAC[String(t["apparent_size"])] ?? 0.12;
    return {
      summary,
      target: {
        visible: true,
        source: "vlm",
        localization: "column",
        bearing_deg: round(bearingFromImageX((xMin + xMax) / 2, this.opts.hfovDeg), 1),
        // Column band around the horizon: excludes the floor right in front of the robot.
        bbox: { x_min: xMin, y_min: 0.3, x_max: xMax, y_max: 0.55 },
        height_frac: heightFrac,
      },
      doorwayBearingsDeg,
    };
  }

  /**
   * Where to explore after a full turn without the target: the vision model sees all the
   * views of the turn at once and picks one, plus the point to walk toward in it. Measured on
   * the house frames (parent scripts/vision_eval/explore_choice_eval.py): a 2.5 m walk that
   * way got closer to the target's room in 58% of the cases (74% for rooms behind a passage),
   * against 38% (21%) for the most open direction.
   */
  async chooseExploreView(views: TurnView[], target: TargetSpec): Promise<ExploreChoice | null> {
    if (views.length < 2 || !this.opts.vision.structuredOutput) return null;
    const content = await this.ask(
      [
        {
          role: "user",
          content: [
            {
              type: "text",
              text:
                `The image shows the ${views.length} views of a full turn in place of a quadruped robot inside a house; each view is ` +
                "labelled with its number and its heading relative to the robot (positive = left). Views marked (explored) face a " +
                "way the robot already walked from here or came from. " +
                `The robot is looking for: ${targetText(target)}. It is not visible in any view. Which view should the robot walk ` +
                "toward to find it? Think about which room each view shows and where doorways, corridors or passages lead: e.g. a " +
                "bed is in a bedroom, usually reached through a doorway or corridor; a refrigerator is in the kitchen. Glass doors " +
                "and windows cannot be walked through. Choose an explored view only if no other view can lead to the target. " +
                'Return JSON: {"reason": string (one or two sentences), "view": <view number>, "x": number (horizontal position ' +
                "in the chosen view of the doorway, passage or free floor to walk toward, as a fraction of that view's width: " +
                "0.0 = its left edge, 1.0 = its right edge)}",
            },
            { type: "image_url", image_url: { url: await turnMosaic(views) } },
          ],
        },
      ],
      500,
      EXPLORE_SCHEMA,
    );
    const parsed = parseJsonObject(content);
    const view = Number(parsed?.["view"]);
    const x = Number(parsed?.["x"]);
    if (!Number.isInteger(view) || view < 0 || view >= views.length || !Number.isFinite(x)) return null;
    return { view, x: Math.min(1, Math.max(0, x)), reason: String(parsed?.["reason"] ?? "") };
  }

  async dispose(): Promise<void> {
    await this.detector?.dispose().catch(() => {});
  }
}
