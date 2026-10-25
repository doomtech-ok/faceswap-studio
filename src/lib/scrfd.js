// SCRFD face detector helpers (InsightFace det_10g / scrfd_10g_bnkps).
//
// Pure functions with no DOM access, so they can be unit-tested in Node.
//
// Model facts (from InsightFace's scrfd.py):
// - Input [1, 3, 640, 640], RGB, normalized as (pixel - 127.5) / 128.
//   The image is resized keeping its aspect ratio and placed at the top-left
//   of the 640x640 input (the rest is zero padding).
// - 9 outputs: scores, bboxes and keypoints for strides 8, 16 and 32.
//   Each feature-map cell has 2 anchors. Boxes are distances
//   (left, top, right, bottom) from the anchor center, in stride units.

export const DET_SIZE = 640;
const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;

// Builds the planar NCHW float input from RGBA pixels already letterboxed
// into a DET_SIZE x DET_SIZE buffer.
export function rgbaToDetInput(rgba, size = DET_SIZE) {
  const plane = size * size;
  const out = new Float32Array(plane * 3);
  for (let i = 0; i < plane; i++) {
    out[i] = (rgba[i * 4] - 127.5) / 128;
    out[plane + i] = (rgba[i * 4 + 1] - 127.5) / 128;
    out[2 * plane + i] = (rgba[i * 4 + 2] - 127.5) / 128;
  }
  return out;
}

// Groups the raw outputs by stride and kind using their shapes, so the code
// doesn't depend on output names or order.
// `outputs` is an array of { data, dims }.
function groupOutputs(outputs, size) {
  const byStride = {};
  for (const o of outputs) {
    const channels = o.dims[o.dims.length - 1];
    const count = o.data.length / channels;
    const stride = STRIDES.find((s) => (size / s) * (size / s) * NUM_ANCHORS === count);
    if (!stride) continue;
    const g = (byStride[stride] ??= {});
    if (channels === 1) g.scores = o.data;
    else if (channels === 4) g.boxes = o.data;
    else if (channels === 10) g.kps = o.data;
  }
  return byStride;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter || 1);
}

/**
 * Decodes SCRFD outputs into faces in original-image pixels.
 * @param {{data: ArrayLike<number>, dims: readonly number[]}[]} outputs
 * @param {number} scale resize factor used for the letterbox (det / original)
 * @returns {{x: number, y: number, width: number, height: number, score: number, kps: number[][]}[]}
 *   sorted by score, highest first
 */
export function decodeScrfd(outputs, scale, { threshold = 0.5, nms = 0.4, size = DET_SIZE } = {}) {
  const groups = groupOutputs(outputs, size);
  const candidates = [];
  for (const stride of STRIDES) {
    const g = groups[stride];
    if (!g?.scores || !g?.boxes) continue;
    const fw = size / stride;
    for (let i = 0; i < g.scores.length; i++) {
      const score = g.scores[i];
      if (score < threshold) continue;
      const cell = Math.floor(i / NUM_ANCHORS);
      const cx = (cell % fw) * stride;
      const cy = Math.floor(cell / fw) * stride;
      const b = i * 4;
      const box = {
        x1: (cx - g.boxes[b] * stride) / scale,
        y1: (cy - g.boxes[b + 1] * stride) / scale,
        x2: (cx + g.boxes[b + 2] * stride) / scale,
        y2: (cy + g.boxes[b + 3] * stride) / scale,
        score,
        kps: [],
      };
      if (g.kps) {
        for (let k = 0; k < 5; k++) {
          box.kps.push([
            (cx + g.kps[i * 10 + k * 2] * stride) / scale,
            (cy + g.kps[i * 10 + k * 2 + 1] * stride) / scale,
          ]);
        }
      }
      candidates.push(box);
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of candidates) {
    if (kept.every((k) => iou(k, c) < nms)) kept.push(c);
  }
  return kept.map((f) => ({
    x: f.x1,
    y: f.y1,
    width: f.x2 - f.x1,
    height: f.y2 - f.y1,
    score: f.score,
    kps: f.kps,
  }));
}

// Turns a face box into a square crop (inswapper needs a square face),
// clamped to the frame. `pad` grows the square around the face.
export function squareCrop(face, frameW, frameH, pad = 1.0) {
  const side = Math.min(Math.max(face.width, face.height) * pad, frameW, frameH);
  const cx = face.x + face.width / 2;
  const cy = face.y + face.height / 2;
  const x = Math.min(Math.max(cx - side / 2, 0), frameW - side);
  const y = Math.min(Math.max(cy - side / 2, 0), frameH - side);
  return { x, y, width: side, height: side };
}
