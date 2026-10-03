/**
 * Tiled YOLO detection for small / distant objects.
 *
 * YOLOv8n letterboxes the whole frame to 640×640, so on a 1920×1080 camera a
 * target a few meters away shrinks to ~15×30 px and scores below any usable
 * threshold. Running the detector on overlapping square crops of the band
 * around the horizon (where ground-standing objects appear at distance)
 * upsamples those objects ~3× and restores confident detections, while the
 * full-frame pass still catches objects that are close and large.
 */

import sharp from "sharp";
import type { PersonDetection, PersonDetector } from "./detector.js";

export interface TiledDetectionOptions {
  /** Square tile side in source pixels (default: 25% of the frame width, min 320). */
  tileSize?: number;
  /** Fraction of the tile side shared by neighbouring tiles (default 0.25). */
  overlap?: number;
  /** Vertical centre of the tile band as a fraction of image height (default 0.5). */
  bandCenter?: number;
  /** IoU above which overlapping detections are merged (default 0.4). */
  iouThreshold?: number;
}

export interface TiledDetectionResult {
  width: number;
  height: number;
  detections: PersonDetection[];
  tiles: number;
}

function iou(a: PersonDetection, b: PersonDetection): number {
  const x0 = Math.max(a.x, b.x);
  const y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.width, b.x + b.width);
  const y1 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

function tileOrigins(length: number, tile: number, stride: number): number[] {
  if (tile >= length) return [0];
  const origins: number[] = [];
  for (let o = 0; o + tile < length; o += stride) origins.push(Math.round(o));
  origins.push(length - tile);
  return origins;
}

export async function detectClassTiled(
  detector: PersonDetector,
  image: Buffer | Uint8Array,
  classId: number,
  opts: TiledDetectionOptions = {},
): Promise<TiledDetectionResult> {
  const input = Buffer.isBuffer(image) ? image : Buffer.from(image);
  const full = await detector.detectClass(input, classId);
  const { width, height } = full;
  const all: PersonDetection[] = [...full.detections];

  const tile = Math.min(height, width, Math.max(320, Math.round(opts.tileSize ?? width * 0.25)));
  const stride = Math.max(1, tile * (1 - (opts.overlap ?? 0.25)));
  const top = Math.round(
    Math.max(0, Math.min(height - tile, (opts.bandCenter ?? 0.5) * height - tile / 2)),
  );
  const origins = tile < width ? tileOrigins(width, tile, stride) : [];

  // A box touching a tile edge that lies inside the frame is a truncated view of a
  // larger object (wrong centre and size): the full-frame pass or a neighbouring
  // tile sees it whole, so drop it.
  const EDGE_PX = 2;
  for (const left of origins) {
    const crop = await sharp(input).extract({ left, top, width: tile, height: tile }).png().toBuffer();
    const res = await detector.detectClass(crop, classId);
    for (const d of res.detections) {
      const cutLeft = left > 0 && d.x <= EDGE_PX;
      const cutRight = left + tile < width && d.x + d.width >= tile - EDGE_PX;
      const cutTop = top > 0 && d.y <= EDGE_PX;
      const cutBottom = top + tile < height && d.y + d.height >= tile - EDGE_PX;
      if (cutLeft || cutRight || cutTop || cutBottom) continue;
      all.push({ ...d, x: d.x + left, y: d.y + top, cx: d.cx + left, cy: d.cy + top });
    }
  }

  all.sort((a, b) => b.confidence - a.confidence);
  const kept: PersonDetection[] = [];
  const iouThreshold = opts.iouThreshold ?? 0.4;
  for (const d of all) {
    if (kept.every((k) => iou(k, d) < iouThreshold)) kept.push(d);
  }
  return { width, height, detections: kept, tiles: origins.length + 1 };
}
