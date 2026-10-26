// @ts-nocheck -- JS module under test has loose JSDoc types
import { describe, expect, it } from 'vitest';
import { fitWithin, sharpenRGBA, shouldDetect } from './imageOps.js';

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

describe('sharpenRGBA', () => {
  it('leaves a flat image unchanged', () => {
    const data = new Uint8ClampedArray(4 * 4 * 4).fill(100);
    sharpenRGBA(data, 4, 4, 1);
    expect(Array.from(data).every((v) => v === 100)).toBe(true);
  });
  it('increases contrast across an edge and keeps alpha', () => {
    // 4x1 image: dark | dark | light | light
    const vals = [50, 50, 200, 200];
    const data = new Uint8ClampedArray(vals.flatMap((v) => [v, v, v, 255]));
    sharpenRGBA(data, 4, 1, 1);
    expect(data[4]).toBeLessThan(50); // dark side of the edge gets darker
    expect(data[8]).toBeGreaterThan(200); // light side gets lighter
    expect(data[7]).toBe(255);
  });
});
