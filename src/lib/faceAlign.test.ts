// @ts-nocheck -- JS module under test has loose JSDoc tuple types
import { describe, expect, it } from 'vitest';
import {
  ARCFACE_DST,
  alignMatrix,
  applyMatrix,
  estimateSimilarity,
  invertMatrix,
  templateFor,
  ffhqMatrix,
  FFHQ_TEMPLATE,
} from './faceAlign.js';

describe('face alignment', () => {
  it('uses the insightface template for 128 (x shifted by 8)', () => {
    expect(templateFor(128)[0][0]).toBeCloseTo(46.2946, 4);
    expect(templateFor(128)[0][1]).toBeCloseTo(51.6963, 4);
    expect(templateFor(112)).toEqual(ARCFACE_DST);
  });

  it('recovers a known rotation + scale + shift exactly', () => {
    const angle = 0.3;
    const s = 2.5;
    const truth = [s * Math.cos(angle), s * Math.sin(angle), -s * Math.sin(angle), s * Math.cos(angle), 40, -12];
    const src = ARCFACE_DST;
    const dst = src.map((p) => applyMatrix(truth, p));
    const m = estimateSimilarity(src, dst);
    m.forEach((v, i) => expect(v).toBeCloseTo(truth[i], 6));
  });

  it('puts tilted eyes on a horizontal line in the aligned crop', () => {
    // A face rotated 25 degrees and scaled 3x somewhere in a frame.
    const angle = (25 * Math.PI) / 180;
    const place = [3 * Math.cos(angle), 3 * Math.sin(angle), -3 * Math.sin(angle), 3 * Math.cos(angle), 500, 200];
    const kps = ARCFACE_DST.map((p) => applyMatrix(place, p));
    const m = alignMatrix(kps, 128);
    const [le, re] = [applyMatrix(m, kps[0]), applyMatrix(m, kps[1])];
    expect(le[1]).toBeCloseTo(re[1], 0);
    expect(le[0]).toBeCloseTo(templateFor(128)[0][0], 4);
  });

  it('inverse maps the crop back to the frame', () => {
    const m = [1.2, 0.4, -0.4, 1.2, 30, 50];
    const back = applyMatrix(invertMatrix(m), applyMatrix(m, [123, 456]));
    expect(back[0]).toBeCloseTo(123, 6);
    expect(back[1]).toBeCloseTo(456, 6);
  });
});

describe('FFHQ alignment for the GFPGAN restorer', () => {
  it('maps the 5 keypoints onto the FFHQ template in a 512 crop', () => {
    const kps = [[300, 210], [360, 214], [330, 250], [305, 280], [355, 282]];
    const m = ffhqMatrix(kps, 512);
    const eyeL = applyMatrix(m, kps[0]);
    const eyeR = applyMatrix(m, kps[1]);
    expect(eyeL[0]).toBeCloseTo(FFHQ_TEMPLATE[0][0] * 512, -1);
    expect(eyeR[0]).toBeCloseTo(FFHQ_TEMPLATE[1][0] * 512, -1);
    expect(Math.abs(eyeL[1] - eyeR[1])).toBeLessThan(3);
  });
});
