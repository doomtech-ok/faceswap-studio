// Small DOM-free image/video helpers used by faceSwap.js (testable in Node).

// Working resolution for a video: shrink (never enlarge) so the long side is
// at most `maxLong` and the short side at most `maxShort` (720p by default).
// Dimensions are rounded down to even numbers, which H.264/VP9 (4:2:0) need.
export function fitWithin(width, height, maxLong = 1280, maxShort = 720) {
  const long = Math.max(width, height);
  const short = Math.min(width, height);
  const scale = Math.min(1, maxLong / long, maxShort / short);
  const even = (v) => Math.max(2, Math.floor((v * scale) / 2) * 2);
  return { width: even(width), height: even(height) };
}

// How much a face box moved between two detections, relative to the size of
// `prev`: the largest of the center shift (x, y) and the size change (w, h),
// each divided by the previous width/height. 0.05 = 5 %.
export function boxMotion(prev, cur) {
  const pw = Math.max(prev.width, 1e-6);
  const ph = Math.max(prev.height, 1e-6);
  const dcx = Math.abs(cur.x + cur.width / 2 - (prev.x + prev.width / 2)) / pw;
  const dcy = Math.abs(cur.y + cur.height / 2 - (prev.y + prev.height / 2)) / ph;
  const dw = Math.abs(cur.width - prev.width) / pw;
  const dh = Math.abs(cur.height - prev.height) / ph;
  return Math.max(dcx, dcy, dw, dh);
}

// Reuse rule for the swapped face pixels:
// - `prev` is the face detected in the PREVIOUS frame (not the last swapped one).
// - reuse only if the box moved less than MOTION_THRESHOLD (2 %) since then,
// - and the same swap output was reused fewer than MAX_REUSE (4) times in a row.
// The reused pixels are always pasted at the CURRENT frame's keypoints.
export const MOTION_THRESHOLD = 0.02;
export const MAX_REUSE = 4;

export function canReuseSwap(prev, cur, reuseCount = 0, threshold = MOTION_THRESHOLD, maxReuse = MAX_REUSE) {
  if (!prev || !cur) return false;
  if (reuseCount >= maxReuse) return false;
  return boxMotion(prev, cur) < threshold;
}

// Threads for the WASM backend. Multi-threaded WASM needs SharedArrayBuffer,
// which browsers only allow on cross-origin isolated pages; otherwise 1.
// Capped at 8 (e.g. a Ryzen 5 5600 reports 12 logical cores -> 8).
export function pickWasmThreads(hardwareConcurrency, crossOriginIsolated, max = 8) {
  if (!crossOriginIsolated) return 1;
  const cores = Number.isFinite(hardwareConcurrency) && hardwareConcurrency > 0 ? hardwareConcurrency : 4;
  return Math.max(1, Math.min(max, Math.floor(cores)));
}
