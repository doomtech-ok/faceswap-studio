// @ts-nocheck -- JS module under test has loose JSDoc types
import { describe, expect, it } from 'vitest';
import { boxMotion, canReuseSwap, fitWithin, pickWasmThreads } from './imageOps.js';

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

describe('motion check (reuse swap when the box moved less than 5 %)', () => {
  const box = { x: 100, y: 100, width: 200, height: 200 };
  it('measures a 10 px shift on a 200 px box as 5 %', () => {
    expect(boxMotion(box, { ...box, x: 110 })).toBeCloseTo(0.05);
  });
  it('reuses the previous swap when the box moved 4 %', () => {
    expect(canReuseSwap(box, { ...box, x: 108 })).toBe(true);
  });
  it('runs the swapper again when the box moved exactly 5 %', () => {
    expect(canReuseSwap(box, { ...box, x: 110 })).toBe(false);
  });
  it('runs the swapper again when the face grew 10 %', () => {
    expect(canReuseSwap(box, { ...box, width: 220, height: 220 })).toBe(false);
  });
  it('never reuses when there is no previous face or no current face', () => {
    expect(canReuseSwap(null, box)).toBe(false);
    expect(canReuseSwap(box, null)).toBe(false);
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
