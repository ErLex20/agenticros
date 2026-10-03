export {
  PersonDetector,
  type PersonDetection,
  type DetectorOptions,
  type LoadOptions,
} from "./detector.js";

export {
  findObject,
  type FindObjectOptions,
  type FindObjectResult,
} from "./find-object.js";

export {
  detectClassTiled,
  type TiledDetectionOptions,
  type TiledDetectionResult,
} from "./tiled.js";

export { COCO_CLASSES, resolveCocoClassId } from "./coco-classes.js";
