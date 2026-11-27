import { describe, expect, it } from 'vitest';
import { GRID_H, GRID_W, NO_INPUT } from '../src/contracts/types.ts';
import type { Bathymetry, SurferInput } from '../src/contracts/types.ts';
import {
  applyBrush,
  blocksAt,
  createCustomBathymetry,
  DEPTH_MAX,
  DEPTH_MIN,
  deserializeBathymetry,
  fromPreset,
  serializeBathymetry,
  smoothPass,
} from '../src/editor/terrainModel.ts';
import { renderTerrainPixels } from '../src/editor/terrainRender.ts';
import { RuleAgent } from '../src/policies/ruleAgent.ts';
import { createWorld, stepWorld } from '../src/sim/world.ts';

function makeHandMadeBarField(): Bathymetry {
  const field = fromPreset('flat', 0);
  for (let y = 38; y <= 44; y++) {
    for (let x = 16; x <= 112; x++) {
      const idx = y * GRID_W + x;
      field.depth[idx] = Math.max(field.depth[idx] - 2.6, 0.8);
    }
  }
  return field;
}

describe('Terrain Studio model', () => {
  it('applyBrush raises/lowers within radius, clips to grid, and clamps at bounds', () => {
    const depth = new Float32Array(GRID_W * GRID_H).fill(5);
    const changed1 = applyBrush(depth, 64, 48, 0, 1);
    expect(changed1).toBe(1);
    expect(depth[48 * GRID_W + 64]).toBe(4.5);
    expect(depth[48 * GRID_W + 63]).toBe(5);
    expect(depth[48 * GRID_W + 65]).toBe(5);
    expect(depth[47 * GRID_W + 64]).toBe(5);
    expect(depth[49 * GRID_W + 64]).toBe(5);

    const depthR1 = new Float32Array(GRID_W * GRID_H).fill(5);
    expect(applyBrush(depthR1, 64, 48, 1, 1)).toBe(9);

    const depthCorner = new Float32Array(GRID_W * GRID_H).fill(5);
    expect(applyBrush(depthCorner, 0, 0, 1, 1)).toBe(4);

    // Repeated raises clamp at DEPTH_MIN (-2) and stop reporting changes.
    const depthMin = new Float32Array(GRID_W * GRID_H).fill(5);
    let totalRaised = 0;
    for (let i = 0; i < 30; i++) {
      totalRaised += applyBrush(depthMin, 10, 10, 0, 1);
    }
    expect(depthMin[10 * GRID_W + 10]).toBe(DEPTH_MIN);
    expect(totalRaised).toBe(14); // (5 - (-2)) / 0.5 = 14
    expect(applyBrush(depthMin, 10, 10, 0, 1)).toBe(0);

    // Repeated lowers clamp at DEPTH_MAX (16) and stop reporting changes.
    const depthMax = new Float32Array(GRID_W * GRID_H).fill(5);
    let totalLowered = 0;
    for (let i = 0; i < 40; i++) {
      totalLowered += applyBrush(depthMax, 10, 10, 0, -1);
    }
    expect(depthMax[10 * GRID_W + 10]).toBe(DEPTH_MAX);
    expect(totalLowered).toBe(22); // (16 - 5) / 0.5 = 22
    expect(applyBrush(depthMax, 10, 10, 0, -1)).toBe(0);
  });

  it('blocksAt converts depth to block height', () => {
    expect(blocksAt(16)).toBe(0);
    expect(blocksAt(0)).toBe(32);
    expect(blocksAt(-2)).toBe(36);
  });

  it('smoothPass is deterministic, non-mutating, and diffuses spikes', () => {
    const input = new Float32Array(GRID_W * GRID_H).fill(10);
    const cx = 64;
    const cy = 48;
    const centerIdx = cy * GRID_W + cx;
    input[centerIdx] = -2;
    const snapshot = new Float32Array(input);

    const out1 = smoothPass(input);
    const out2 = smoothPass(input);

    // Does not mutate input
    expect(input).toEqual(snapshot);
    // Deterministic
    expect(out1).toEqual(out2);
    // Spike moves toward 10
    expect(out1[centerIdx]).toBeGreaterThan(-2);
    expect(out1[centerIdx]).toBeLessThan(10);
    expect(out1[centerIdx]).toBeCloseTo(4, 5);
    // Orthogonal neighbours move down from 10
    expect(out1[centerIdx - 1]).toBeLessThan(10);
    expect(out1[centerIdx + 1]).toBeLessThan(10);
    expect(out1[centerIdx - GRID_W]).toBeLessThan(10);
    expect(out1[centerIdx + GRID_W]).toBeLessThan(10);
  });

  it('serialize/deserialize round-trips identically and rejects wrong dimensions', () => {
    const sandbar = fromPreset('sandbar', 7);
    const s1 = serializeBathymetry(sandbar);
    const d1 = deserializeBathymetry(s1);
    const s2 = serializeBathymetry(d1);
    expect(s2).toBe(s1);
    expect(d1.kind).toBe('custom');

    expect(() =>
      deserializeBathymetry(JSON.stringify({ v: 1, w: 64, h: GRID_H, depth: new Array(64 * GRID_H).fill(5) }))
    ).toThrow();
    expect(() =>
      deserializeBathymetry(JSON.stringify({ v: 1, w: GRID_W, h: 48, depth: new Array(GRID_W * 48).fill(5) }))
    ).toThrow();
    expect(() =>
      deserializeBathymetry(JSON.stringify({ v: 1, w: GRID_W, h: GRID_H, depth: [1, 2, 3] }))
    ).toThrow();
  });
});

describe('Terrain Studio pixel renderer', () => {
  it('renders RGBA with alpha 255, deep water blue, and dry land sandy', () => {
    const depth = new Float32Array(GRID_W * GRID_H).fill(5);
    const deepIdx = 0;
    const dryIdx = 1;
    depth[deepIdx] = 15;
    depth[dryIdx] = -1;
    const bathy = createCustomBathymetry(depth);

    const out = new Uint8ClampedArray(GRID_W * GRID_H * 4);
    renderTerrainPixels(bathy, out);

    expect(out.length).toBe(GRID_W * GRID_H * 4);
    for (let i = 0; i < GRID_W * GRID_H; i++) {
      expect(out[i * 4 + 3]).toBe(255);
    }

    // Deep cell (15 m): dark blue (b > r)
    const dp = deepIdx * 4;
    expect(out[dp + 2]).toBeGreaterThan(out[dp]);

    // Dry cell (-1 m): sandy (r > 200, b < 200)
    const sp = dryIdx * 4;
    expect(out[sp]).toBeGreaterThan(200);
    expect(out[sp + 2]).toBeLessThan(200);
  });
});

describe('Custom bathymetry world integration', () => {
  it('createWorld accepts bathymetryField, deep-copies it, and validates dimensions', () => {
    const field = makeHandMadeBarField();
    const orig0 = field.depth[0];
    const world = createWorld({ bathymetryField: field, aiCount: 3 }, 7);

    expect(world.config.bathymetry).toBe('custom');
    expect(world.bathy.kind).toBe('custom');
    expect(world.bathy.depth).toEqual(field.depth);

    // Mutate original field and confirm world owns an independent copy
    field.depth[0] = -1.5;
    expect(world.bathy.depth[0]).toBe(orig0);

    // Wrong depth length throws
    const badField: Bathymetry = {
      kind: 'custom',
      seed: 0,
      width: GRID_W,
      height: GRID_H,
      depth: new Float32Array(10),
    };
    expect(() => createWorld({ bathymetryField: badField }, 7)).toThrow();
  });

  it(
    'waves break and AI surfers catch waves on a hand-made bar',
    () => {
      const field = makeHandMadeBarField();
      const aiCount = 3;
      const world = createWorld({ bathymetryField: field, aiCount }, 7);
      const agents: RuleAgent[] = [];
      for (let k = 1; k <= aiCount; k++) {
        agents.push(new RuleAgent(k - 1, aiCount));
      }
      const inputs: SurferInput[] = new Array(world.numSurfers);
      inputs[0] = NO_INPUT;

      let breakingObserved = false;
      for (let s = 0; s < 3600; s++) {
        for (let k = 1; k <= aiCount; k++) {
          inputs[k] = agents[k - 1].decide(world, k);
        }
        stepWorld(world, inputs);
        if (!breakingObserved) {
          const breaking = world.ocean.breaking;
          for (let i = 0; i < breaking.length; i++) {
            if (breaking[i] === 1) {
              breakingObserved = true;
              break;
            }
          }
        }
      }

      let totalCatches = 0;
      for (let k = 1; k <= aiCount; k++) {
        totalCatches += world.surferCatches[k];
      }

      expect(breakingObserved).toBe(true);
      expect(totalCatches).toBeGreaterThanOrEqual(1);
    },
    180000
  );
});
