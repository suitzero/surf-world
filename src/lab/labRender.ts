import type { OceanField } from '../contracts/types.ts';
import type { HybridSim } from '../sim/hybrid.ts';
import { BREAKING_RATIO } from '../sim/ocean.ts';

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function ratioColor(r: number, out: [number, number, number]): void {
  const t = clamp(r / 1.2, 0, 1);
  let cr = mix(6, 28, t);
  let cg = mix(52, 148, t);
  let cb = mix(80, 168, t);
  const hi = clamp((r - 0.7) / 0.5, 0, 1) * 0.8;
  out[0] = mix(cr, 255, hi);
  out[1] = mix(cg, 240, hi);
  out[2] = mix(cb, 200, hi);
}

export function foldTimeMax(timeMax: Float32Array, eta: Float32Array): void {
  const n = Math.min(timeMax.length, eta.length);
  for (let i = 0; i < n; i++) {
    const a = Math.abs(eta[i]);
    if (a > timeMax[i]) {
      timeMax[i] = a;
    }
  }
}

export function computeOnsetRows(
  timeMax: Float32Array,
  depth: Float32Array,
  w: number,
  h: number
): Int16Array {
  const out = new Int16Array(w).fill(-1);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      const i = y * w + x;
      const d = depth[i];
      if (d > 0 && (2 * timeMax[i]) / d >= BREAKING_RATIO) {
        out[x] = y;
        break;
      }
    }
  }
  return out;
}

export function physicsPixels(
  field: OceanField,
  depth: Float32Array,
  w: number,
  h: number
): Uint8ClampedArray {
  const n = w * h;
  const buf = new Uint8ClampedArray(n * 4);
  const rgb: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    if (depth[i] <= 0) {
      buf[o] = 186;
      buf[o + 1] = 168;
      buf[o + 2] = 120;
      buf[o + 3] = 255;
      continue;
    }
    ratioColor(field.ratio[i], rgb);
    const foamT = field.foam[i] * 0.5;
    let cr = mix(rgb[0], 255, foamT);
    let cg = mix(rgb[1], 255, foamT);
    let cb = mix(rgb[2], 255, foamT);
    if (field.breaking[i] === 1) {
      cr = mix(cr, 255, 0.45);
      cg = mix(cg, 255, 0.45);
      cb = mix(cb, 255, 0.45);
    }
    buf[o] = cr;
    buf[o + 1] = cg;
    buf[o + 2] = cb;
    buf[o + 3] = 255;
  }
  return buf;
}

export function etaPixels(
  eta: Float32Array,
  depth: Float32Array,
  w: number,
  h: number,
  ampM: number
): Uint8ClampedArray {
  const n = w * h;
  const buf = new Uint8ClampedArray(n * 4);
  const denom = Math.max(ampM, 0.01);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    if (depth[i] <= 0) {
      buf[o] = 120;
      buf[o + 1] = 104;
      buf[o + 2] = 72;
      buf[o + 3] = 255;
      continue;
    }
    const s = clamp(eta[i] / denom, -1, 1);
    let cr = 12;
    let cg = 70;
    let cb = 110;
    if (s < 0) {
      const t = -s;
      cr = mix(12, 6, t);
      cg = mix(70, 30, t);
      cb = mix(110, 80, t);
    } else if (s > 0) {
      cr = mix(12, 140, s);
      cg = mix(70, 225, s);
      cb = mix(110, 240, s);
      const hi = clamp((s - 0.75) / 0.25, 0, 1) * 0.6;
      cr = mix(cr, 255, hi);
      cg = mix(cg, 255, hi);
      cb = mix(cb, 255, hi);
    }
    buf[o] = cr;
    buf[o + 1] = cg;
    buf[o + 2] = cb;
    buf[o + 3] = 255;
  }
  return buf;
}

export function hybridPixels(
  hybrid: HybridSim,
  depth: Float32Array,
  w: number,
  h: number
): Uint8ClampedArray {
  const n = w * h;
  const buf = new Uint8ClampedArray(n * 4);
  const rgb: [number, number, number] = [0, 0, 0];
  const { corrValid, corrRatio, fast } = hybrid;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    if (depth[i] <= 0) {
      buf[o] = 186;
      buf[o + 1] = 168;
      buf[o + 2] = 120;
      buf[o + 3] = 255;
      continue;
    }
    if (corrValid[i] === 1) {
      ratioColor(corrRatio[i], rgb);
      buf[o] = mix(rgb[0], 255, 0.38);
      buf[o + 1] = mix(rgb[1], 110, 0.38);
      buf[o + 2] = mix(rgb[2], 199, 0.38);
      buf[o + 3] = 255;
    } else {
      const dEff = fast.depthEff[i];
      const r = dEff > 0 ? fast.amp[i] / dEff : 0;
      ratioColor(r, rgb);
      buf[o] = rgb[0];
      buf[o + 1] = rgb[1];
      buf[o + 2] = rgb[2];
      buf[o + 3] = 255;
    }
  }
  return buf;
}
