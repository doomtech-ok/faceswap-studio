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

/**
 * Unsharp mask on RGBA pixels, in place: px + amount * (px - blur3x3(px)).
 * Only RGB is changed; alpha is kept. Edge pixels use clamped neighbours.
 * @param {Uint8ClampedArray} data RGBA pixels
 * @param {number} w
 * @param {number} h
 * @param {number} amount 0 = no change, ~0.5-1 = noticeable sharpening
 */
export function sharpenRGBA(data, w, h, amount = 0.6) {
  if (amount === 0) return data;
  const src = new Uint8ClampedArray(data);
  for (let y = 0; y < h; y++) {
    const y0 = y > 0 ? y - 1 : 0;
    const y2 = y < h - 1 ? y + 1 : h - 1;
    for (let x = 0; x < w; x++) {
      const x0 = x > 0 ? x - 1 : 0;
      const x2 = x < w - 1 ? x + 1 : w - 1;
      const i = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) {
        const sum =
          src[(y0 * w + x0) * 4 + c] + src[(y0 * w + x) * 4 + c] + src[(y0 * w + x2) * 4 + c] +
          src[(y * w + x0) * 4 + c] + src[i + c] + src[(y * w + x2) * 4 + c] +
          src[(y2 * w + x0) * 4 + c] + src[(y2 * w + x) * 4 + c] + src[(y2 * w + x2) * 4 + c];
        const px = src[i + c];
        data[i + c] = px + amount * (px - sum / 9);
      }
    }
  }
  return data;
}
