// Face alignment helpers (InsightFace face_align.estimate_norm), DOM-free.
//
// A similarity transform (rotation + uniform scale + translation) maps the 5
// detector keypoints (left eye, right eye, nose, left mouth, right mouth) onto
// the ArcFace template, so eyes and mouth end up level and in fixed spots.

// ArcFace template for a 112x112 crop.
export const ARCFACE_DST = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

// Template scaled for `size`, same rule as insightface: multiples of 112 scale
// directly; otherwise (e.g. 128 for inswapper) scale by size/128 and shift x by 8.
export function templateFor(size) {
  let ratio;
  let diffX;
  if (size % 112 === 0) {
    ratio = size / 112;
    diffX = 0;
  } else {
    ratio = size / 128;
    diffX = 8 * ratio;
  }
  return ARCFACE_DST.map(([x, y]) => [x * ratio + diffX, y * ratio]);
}

/**
 * Least-squares similarity transform from `src` points to `dst` points
 * (equivalent to skimage SimilarityTransform / Umeyama in 2D).
 * Returns canvas-style matrix [a, b, c, d, e, f]:
 *   x' = a*x + c*y + e,  y' = b*x + d*y + f
 */
export function estimateSimilarity(src, dst) {
  const n = src.length;
  let msx = 0, msy = 0, mdx = 0, mdy = 0;
  for (let i = 0; i < n; i++) {
    msx += src[i][0]; msy += src[i][1];
    mdx += dst[i][0]; mdy += dst[i][1];
  }
  msx /= n; msy /= n; mdx /= n; mdy /= n;

  let num1 = 0, num2 = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - msx, sy = src[i][1] - msy;
    const dx = dst[i][0] - mdx, dy = dst[i][1] - mdy;
    num1 += sx * dx + sy * dy;
    num2 += sx * dy - sy * dx;
    den += sx * sx + sy * sy;
  }
  const a = num1 / den; // s*cos
  const b = num2 / den; // s*sin
  const tx = mdx - (a * msx - b * msy);
  const ty = mdy - (b * msx + a * msy);
  return [a, b, -b, a, tx, ty];
}

// Matrix that maps image keypoints into an aligned `size` x `size` crop.
export function alignMatrix(kps, size) {
  return estimateSimilarity(kps, templateFor(size));
}

export function invertMatrix([a, b, c, d, e, f]) {
  const det = a * d - b * c;
  return [
    d / det,
    -b / det,
    -c / det,
    a / det,
    (c * f - d * e) / det,
    (b * e - a * f) / det,
  ];
}

export function applyMatrix([a, b, c, d, e, f], [x, y]) {
  return [a * x + c * y + e, b * x + d * y + f];
}
