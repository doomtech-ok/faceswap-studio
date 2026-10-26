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

// True on frame 0 and then every `every` frames; in between, the last
// detected face box + keypoints are reused.
export function shouldDetect(frameIndex, every = 5) {
  return frameIndex % every === 0;
}

// Threads for the WASM backend. Multi-threaded WASM needs SharedArrayBuffer,
// which browsers only allow on cross-origin isolated pages; otherwise 1.
// Capped at 8 (e.g. a Ryzen 5 5600 reports 12 logical cores -> 8).
export function pickWasmThreads(hardwareConcurrency, crossOriginIsolated, max = 8) {
  if (!crossOriginIsolated) return 1;
  const cores = Number.isFinite(hardwareConcurrency) && hardwareConcurrency > 0 ? hardwareConcurrency : 4;
  return Math.max(1, Math.min(max, Math.floor(cores)));
}
