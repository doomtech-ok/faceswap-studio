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

// True when the swapped face from `prev` can be reused for `cur` (both faces
// detected and the box moved less than `threshold`). Otherwise the swapper
// must run again.
export function canReuseSwap(prev, cur, threshold = 0.05) {
  if (!prev || !cur) return false;
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
