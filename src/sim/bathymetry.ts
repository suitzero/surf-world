import { GRID_H, GRID_W } from '../contracts/types.ts';
import type { Bathymetry, BathymetryKind } from '../contracts/types.ts';
import { SeededRng } from './prng.ts';

/**
 * Procedural bathymetry (S02). Depth is a 2D height field in metres:
 * positive = water depth, <= 0 = dry land. y = 0 is offshore (deep),
 * y increases toward the beach at the bottom of the grid.
 *
 * Every generator is a pure function of (kind, seed): same seed, same reef.
 */

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

/**
 * Smooth value noise: random values on a coarse lattice, bilinearly
 * interpolated. Deterministic for a given rng stream position.
 */
function makeNoise(rng: SeededRng, w: number, h: number, cell: number): (x: number, y: number) => number {
  const lw = Math.ceil(w / cell) + 1;
  const lh = Math.ceil(h / cell) + 1;
  const lattice = new Float32Array(lw * lh);
  for (let i = 0; i < lattice.length; i++) lattice[i] = rng.next() * 2 - 1;
  return (x: number, y: number): number => {
    const fx = x / cell;
    const fy = y / cell;
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const tx = fx - x0;
    const ty = fy - y0;
    const sx = tx * tx * (3 - 2 * tx);
    const sy = ty * ty * (3 - 2 * ty);
    const i00 = lattice[y0 * lw + x0];
    const i10 = lattice[y0 * lw + x0 + 1];
    const i01 = lattice[(y0 + 1) * lw + x0];
    const i11 = lattice[(y0 + 1) * lw + x0 + 1];
    const top = i00 + (i10 - i00) * sx;
    const bot = i01 + (i11 - i01) * sx;
    return top + (bot - top) * sy;
  };
}

/** Base beach profile: 14 m offshore shoaling to dry sand. t in [0,1]. */
function beachProfile(t: number): number {
  return 14 * Math.pow(1 - t, 1.35) - 1.5 * t;
}

function genBeach(rng: SeededRng, w: number, h: number, out: Float32Array): void {
  const noise = makeNoise(rng, w, h, 9);
  for (let y = 0; y < h; y++) {
    const t = y / (h - 1);
    for (let x = 0; x < w; x++) {
      out[y * w + x] = beachProfile(t) + noise(x, y) * 0.18;
    }
  }
}

function genSandbar(rng: SeededRng, w: number, h: number, out: Float32Array): void {
  const noise = makeNoise(rng, w, h, 8);
  const phase = rng.range(0, Math.PI * 2);
  const barAmp = rng.range(2.3, 2.9); // metres the bar raises the bottom
  for (let y = 0; y < h; y++) {
    const t = y / (h - 1);
    for (let x = 0; x < w; x++) {
      // Gentler base slope than the plain beach.
      let d = 11 * Math.pow(1 - t, 1.2) - 1.2 * t;
      // Crescent bar: crest line wobbles alongshore, ends taper into channels.
      const barT = 0.42 + 0.028 * Math.sin((x / w) * Math.PI * 2 + phase) + noise(x, y) * 0.006;
      const along = smoothstep(0.02, 0.14, x / w) * (1 - smoothstep(0.86, 0.98, x / w));
      const barShape = Math.exp(-Math.pow((t - barT) / 0.045, 2));
      d -= barAmp * barShape * (0.35 + 0.65 * along);
      // Trough just shoreward of the bar.
      const troughShape = Math.exp(-Math.pow((t - (barT + 0.13)) / 0.05, 2));
      d += 1.1 * troughShape;
      out[y * w + x] = d + noise(x, y) * 0.15;
    }
  }
}

function genPoint(rng: SeededRng, w: number, h: number, out: Float32Array): void {
  const noise = makeNoise(rng, w, h, 8);
  for (let y = 0; y < h; y++) {
    const t = y / (h - 1);
    for (let x = 0; x < w; x++) {
      // Headland on the +x side: depth contours are pushed offshore there,
      // so the shallow reef tongue reaches out and waves peel away from it.
      const hx = smoothstep(0.42, 0.95, x / w);
      let d = beachProfile(clamp(t + hx * 0.26, 0, 1));
      // Reef flat on the point itself, feathered on every edge so depth
      // contours curve instead of forming a rectangle: the weight fades in
      // with hx and fades out at both t edges. The flat shallows toward the
      // point tip (+hx) and toward shore (+t) and is deeper at its offshore
      // edge, so the breaking ratio crosses its threshold along a smooth
      // gradient instead of only at noise bumps.
      const flatW =
        smoothstep(0.48, 0.66, hx) *
        smoothstep(0.14, 0.3, t) *
        (1 - smoothstep(0.72, 0.88, t));
      if (flatW > 0) {
        const flat = 4.1 - (hx - 0.45) * 1.7 - (t - 0.2) * 1.1 + noise(x, y) * 0.18;
        const target = Math.min(d, flat);
        d = d + (target - d) * flatW;
      }
      out[y * w + x] = d + noise(x, y) * 0.15;
    }
  }
}

function genReef(rng: SeededRng, w: number, h: number, out: Float32Array): void {
  const noise = makeNoise(rng, w, h, 6);
  const heads = makeNoise(rng, w, h, 3);
  for (let y = 0; y < h; y++) {
    const t = y / (h - 1);
    for (let x = 0; x < w; x++) {
      const x01 = x / w;
      // Reef edge runs diagonally: further offshore on the west, closer east.
      const edgeT = 0.4 - 0.12 * x01 + noise(x, y) * 0.008;
      // Smooth transition band across the edge (was a hard if/else step):
      // the offshore slope blends into the shallow platform over +/-0.035 of t.
      const edgeBlend = smoothstep(edgeT - 0.035, edgeT + 0.035, t);
      const offSlope = 13 - 9.5 * clamp(t / Math.max(edgeT, 0.05), 0, 1);
      // Shallow reef platform with coral-head bumps.
      const platform = 2.9 + noise(x, y) * 0.45 + heads(x, y) * 0.35;
      let d = offSlope * (1 - edgeBlend) + platform * edgeBlend;
      // Dry sand strip at the shore, feathered in.
      const dry = smoothstep(0.925, 0.965, t);
      d = d * (1 - dry) + -1.2 * dry;
      // Deep channel cutting diagonally across the platform (platform only).
      const chanCenter = 0.22 + 0.3 * t;
      const chanDist = Math.abs(x01 - chanCenter);
      const chan = Math.exp(-Math.pow(chanDist / 0.045, 2));
      d += 7.5 * chan * edgeBlend * (1 - dry);
      out[y * w + x] = d;
    }
  }
}

export function generateBathymetry(
  kind: BathymetryKind,
  seed: number,
  width: number = GRID_W,
  height: number = GRID_H
): Bathymetry {
  const rng = new SeededRng(seed);
  const depth = new Float32Array(width * height);
  switch (kind) {
    case 'beach':
      genBeach(rng, width, height, depth);
      break;
    case 'sandbar':
      genSandbar(rng, width, height, depth);
      break;
    case 'point':
      genPoint(rng, width, height, depth);
      break;
    case 'reef':
      genReef(rng, width, height, depth);
      break;
  }
  return { kind, seed, width, height, depth };
}
