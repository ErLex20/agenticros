import { test } from "node:test";
import assert from "node:assert/strict";
import { normalizeBox, parseJsonObject, visionProfile } from "../vision.js";

test("JSON is extracted despite fences, think blocks, prose and a stray trailing brace", () => {
  // MiniCPM-V often closes compact JSON with one brace too many.
  const extra = '{"summary": "a bed", "target": {"visible": true, "column": 5, "apparent_size": "large"}}}';
  assert.deepEqual(parseJsonObject(extra)?.["target"], { visible: true, column: 5, apparent_size: "large" });
  assert.equal(parseJsonObject('<think>{"x": 1}</think>```json\n{"ok": true}\n```')?.["ok"], true);
  assert.equal(parseJsonObject('Here {not json} is the answer: {"a": "b}"}')?.["a"], "b}");
  assert.equal(parseJsonObject("no json here"), null);
});

test("boxes are normalized from fractions or 0-1000 and sanitized", () => {
  assert.deepEqual(normalizeBox([0.1, 0.4, 0.3, 0.6]), { x_min: 0.1, y_min: 0.4, x_max: 0.3, y_max: 0.6 });
  assert.deepEqual(normalizeBox([300, 600, 100, 400]), { x_min: 0.1, y_min: 0.4, x_max: 0.3, y_max: 0.6 });
  assert.equal(normalizeBox([0.1, 0.4, 0.1, 0.6]), null); // zero width
  assert.equal(normalizeBox([10, 20, 3000, 40]), null); // pixels of an unknown image size
  assert.equal(normalizeBox([0.1, 0.2, 0.3]), null);
  assert.equal(normalizeBox(null), null);
});

test("vision profiles: grounding models get boxes, others columns", () => {
  assert.equal(visionProfile("moonshotai/Kimi-K2.6").grounding, "box");
  assert.deepEqual(visionProfile("moonshotai/Kimi-K2.6").extraBody, { chat_template_kwargs: { thinking: false } });
  assert.equal(visionProfile("openbmb/MiniCPM-V-4_5").grounding, "columns");
  assert.equal(visionProfile("some/unknown-model").grounding, "columns");
  assert.equal(visionProfile("some/unknown-model", "box").grounding, "box");
});
