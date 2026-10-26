// @ts-nocheck -- JS module under test has loose JSDoc types
import { describe, expect, it } from 'vitest';
import { fitWithin, pickWasmThreads, shouldDetect } from './imageOps.js';

describe('fitWithin (720p working canvas)', () => {
  it('shrinks 1080p landscape to 1280x720', () => {
    expect(fitWithin(1920, 1080)).toEqual({ width: 1280, height: 720 });
  });
  it('shrinks portrait 1080x1920 to 720x1280', () => {
    expect(fitWithin(1080, 1920)).toEqual({ width: 720, height: 1280 });
  });
  it('never enlarges a smaller video', () => {
    expect(fitWithin(640, 360)).toEqual({ width: 640, height: 360 });
  });
  it('returns even dimensions', () => {
    const { width, height } = fitWithin(1921, 1081);
    expect(width % 2).toBe(0);
    expect(height % 2).toBe(0);
  });
});

describe('shouldDetect (detector every 5 frames)', () => {
  it('runs on frames 0, 5, 10 and skips the ones in between', () => {
    const detected = Array.from({ length: 11 }, (_, i) => i).filter((i) => shouldDetect(i, 5));
    expect(detected).toEqual([0, 5, 10]);
  });
});

describe('pickWasmThreads', () => {
  it('uses 8 threads on a 12-thread CPU (Ryzen 5 5600) when isolated', () => {
    expect(pickWasmThreads(12, true)).toBe(8);
  });
  it('uses all cores when there are fewer than 8', () => {
    expect(pickWasmThreads(4, true)).toBe(4);
  });
  it('falls back to 1 thread without cross-origin isolation', () => {
    expect(pickWasmThreads(12, false)).toBe(1);
  });
});
