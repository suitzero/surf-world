import { GRID_H, GRID_W } from '../contracts/types.ts';
import type { Bathymetry, BathymetryKind } from '../contracts/types.ts';
import { generateBathymetry } from '../sim/bathymetry.ts';

export const BLOCK_M = 0.5;
export const DEPTH_MIN = -2;
export const DEPTH_MAX = 16;

export function clampDepth(d: number): number {
  return d < DEPTH_MIN ? DEPTH_MIN : d > DEPTH_MAX ? DEPTH_MAX : d;
}

export function createCustomBathymetry(depth: Float32Array): Bathymetry {
  return {
    kind: 'custom',
    seed: 0,
    width: GRID_W,
    height: GRID_H,
    depth,
  };
}

export function flatBeachDepth(): Float32Array {
  const depth = new Float32Array(GRID_W * GRID_H);
  for (let y = 0; y < GRID_H; y++) {
    const t = y / (GRID_H - 1);
    const d = clampDepth(14 * Math.pow(1 - t, 1.35) - 1.5 * t);
    const row = y * GRID_W;
    for (let x = 0; x < GRID_W; x++) {
      depth[row + x] = d;
    }
  }
  return depth;
}

export function fromPreset(preset: BathymetryKind | 'flat', seed: number): Bathymetry {
  if (preset === 'flat') {
    return createCustomBathymetry(flatBeachDepth());
  }
  const generated = generateBathymetry(preset, seed);
  return createCustomBathymetry(new Float32Array(generated.depth));
}

export function blocksAt(depth: number): number {
  return Math.round((DEPTH_MAX - depth) / BLOCK_M);
}

export function applyBrush(
  depth: Float32Array,
  cx: number,
  cy: number,
  radius: number,
  delta: 1 | -1
): number {
  const x0 = Math.max(0, Math.floor(cx) - radius);
  const x1 = Math.min(GRID_W - 1, Math.floor(cx) + radius);
  const y0 = Math.max(0, Math.floor(cy) - radius);
  const y1 = Math.min(GRID_H - 1, Math.floor(cy) + radius);
  let changed = 0;
  for (let y = y0; y <= y1; y++) {
    const row = y * GRID_W;
    for (let x = x0; x <= x1; x++) {
      const idx = row + x;
      const before = depth[idx];
      const next = clampDepth(before - delta * BLOCK_M);
      depth[idx] = next;
      if (depth[idx] !== before) {
        changed++;
      }
    }
  }
  return changed;
}

export function smoothPass(depth: Float32Array): Float32Array {
  const out = new Float32Array(GRID_W * GRID_H);
  for (let y = 0; y < GRID_H; y++) {
    const row = y * GRID_W;
    for (let x = 0; x < GRID_W; x++) {
      const idx = row + x;
      let sum = 0;
      let count = 0;
      if (x > 0) {
        sum += depth[idx - 1];
        count++;
      }
      if (x < GRID_W - 1) {
        sum += depth[idx + 1];
        count++;
      }
      if (y > 0) {
        sum += depth[idx - GRID_W];
        count++;
      }
      if (y < GRID_H - 1) {
        sum += depth[idx + GRID_W];
        count++;
      }
      const mean = count > 0 ? sum / count : depth[idx];
      out[idx] = clampDepth(0.5 * depth[idx] + 0.5 * mean);
    }
  }
  return out;
}

export function serializeBathymetry(b: Bathymetry): string {
  const depth = new Array<number>(b.depth.length);
  for (let i = 0; i < b.depth.length; i++) {
    depth[i] = Math.round(b.depth[i] * 100) / 100;
  }
  return JSON.stringify({
    v: 1,
    w: b.width,
    h: b.height,
    depth,
  });
}

export function deserializeBathymetry(json: string): Bathymetry {
  const parsed: unknown = JSON.parse(json);
  if (typeof parsed !== 'object' || parsed === null) {
    throw new Error('Invalid bathymetry JSON: expected an object');
  }
  const obj = parsed as { w?: unknown; h?: unknown; depth?: unknown };
  if (obj.w !== GRID_W || obj.h !== GRID_H) {
    throw new Error(
      `Invalid bathymetry dimensions: expected ${GRID_W}x${GRID_H}, got ${String(obj.w)}x${String(obj.h)}`
    );
  }
  if (!Array.isArray(obj.depth) || obj.depth.length !== GRID_W * GRID_H) {
    throw new Error(
      `Invalid bathymetry depth array: expected length ${GRID_W * GRID_H}, got ${Array.isArray(obj.depth) ? obj.depth.length : typeof obj.depth}`
    );
  }
  const depth = new Float32Array(GRID_W * GRID_H);
  for (let i = 0; i < depth.length; i++) {
    const val = obj.depth[i];
    if (typeof val !== 'number' || !Number.isFinite(val)) {
      throw new Error(`Invalid bathymetry depth value at index ${i}`);
    }
    depth[i] = clampDepth(val);
  }
  return createCustomBathymetry(depth);
}
