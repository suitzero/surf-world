import { describe, expect, it } from 'vitest';
import type { Bathymetry, BathymetryKind } from '../src/contracts/types.ts';
import { GRID_H, GRID_W } from '../src/contracts/types.ts';
import { generateBathymetry } from '../src/sim/bathymetry.ts';
import { createTeacher } from '../src/sim/teacher.ts';

const DT = 0.05;

function makeFlatTank(depthM: number): Bathymetry {
  return {
    kind: 'beach',
    seed: 0,
    width: GRID_W,
    height: GRID_H,
    depth: new Float32Array(GRID_W * GRID_H).fill(depthM),
  };
}

function maxAbs(arr: Float32Array): { max: number; allFinite: boolean } {
  let max = 0;
  let allFinite = true;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (!Number.isFinite(v)) {
      allFinite = false;
      break;
    }
    const a = Math.abs(v);
    if (a > max) max = a;
  }
  return { max, allFinite };
}

describe('2D linear shallow-water teacher solver', () => {
  const kinds: readonly BathymetryKind[] = ['sandbar', 'beach', 'point', 'reef'];

  for (const kind of kinds) {
    it(`stability/CFL over 120 s on ${kind} (seed 7)`, () => {
      const bathy = generateBathymetry(kind, 7);
      const teacher = createTeacher(bathy);
      const steps = Math.round(120 / DT);
      for (let s = 0; s < steps; s++) {
        teacher.step(DT);
      }
      const { max, allFinite } = maxAbs(teacher.eta);
      expect(allFinite).toBe(true);
      expect(max).toBeLessThanOrEqual(3 * 0.9);
      expect(teacher.time).toBeCloseTo(120, 6);
    });
  }

  it('substep rule: large dt = 1.0 s triggers multiple CFL substeps', () => {
    const bathy = generateBathymetry('sandbar', 7);
    const teacher = createTeacher(bathy);
    teacher.step(1.0);
    expect(teacher.lastSubsteps).toBeGreaterThanOrEqual(2);
    expect(teacher.time).toBeCloseTo(1.0, 9);
  });

  it('dry cells stay exactly 0 in eta with no NaNs in u/v', () => {
    const bathy = generateBathymetry('sandbar', 7);
    const teacher = createTeacher(bathy);
    const steps = Math.round(30 / DT);
    for (let s = 0; s < steps; s++) {
      teacher.step(DT);
    }

    let dryCount = 0;
    for (let i = 0; i < bathy.depth.length; i++) {
      if (bathy.depth[i] <= 0) {
        dryCount++;
        expect(teacher.eta[i]).toBe(0);
      }
      expect(Number.isNaN(teacher.u[i])).toBe(false);
      expect(Number.isNaN(teacher.v[i])).toBe(false);
    }
    expect(dryCount).toBeGreaterThan(0);

    // Out-of-bounds and dry-cell queries via sampleEta return exactly 0.
    expect(teacher.sampleEta(-1, 20)).toBe(0);
    expect(teacher.sampleEta(20, -1)).toBe(0);
    expect(teacher.sampleEta(bathy.width * teacher.cellM + 10, 20)).toBe(0);
    expect(teacher.sampleEta(20, bathy.height * teacher.cellM + 10)).toBe(0);
    for (let y = 0; y < bathy.height; y++) {
      for (let x = 0; x < bathy.width; x++) {
        if (bathy.depth[y * bathy.width + x] <= 0) {
          expect(teacher.sampleEta((x + 0.5) * teacher.cellM, (y + 0.5) * teacher.cellM)).toBe(0);
          break;
        }
      }
    }
  });

  it('wave speed in a flat 16 m tank matches sqrt(g*h) within +/-25%', () => {
    const tank = makeFlatTank(16);
    const teacher = createTeacher(tank, { amplitudeM: 0.5, periodS: 9, directionDeg: 0 });
    const steps = Math.round(60 / DT);
    const xM = 256;
    const y1 = 40;
    const y2 = 200;

    const times = new Float64Array(steps);
    const s1 = new Float64Array(steps);
    const s2 = new Float64Array(steps);

    for (let s = 0; s < steps; s++) {
      teacher.step(DT);
      times[s] = teacher.time;
      s1[s] = teacher.sampleEta(xM, y1);
      s2[s] = teacher.sampleEta(xM, y2);
    }

    // Track the first crest (local maximum above 0.3) after t > 5 s at probe 1,
    // and the corresponding first crest at probe 2 after the wave can reach it.
    const findFirstCrestAfter = (series: Float64Array, minTime: number): number => {
      for (let i = 1; i < series.length - 1; i++) {
        if (times[i] <= minTime) continue;
        if (series[i] > 0.3 && series[i] >= series[i - 1] && series[i] > series[i + 1]) {
          return times[i];
        }
      }
      return -1;
    };

    const t1 = findFirstCrestAfter(s1, 5);
    const t2 = findFirstCrestAfter(s2, 5);
    expect(t1).toBeGreaterThan(5);
    expect(t2).toBeGreaterThan(t1);

    const measuredSpeed = (y2 - y1) / (t2 - t1);
    const expectedSpeed = Math.sqrt(9.81 * 16); // ~12.53 m/s
    expect(measuredSpeed).toBeGreaterThanOrEqual(expectedSpeed * 0.75);
    expect(measuredSpeed).toBeLessThanOrEqual(expectedSpeed * 1.25);
  });

  it('forcing enters domain and reflections remain bounded over 120 s', () => {
    const amplitude = 0.9;
    const bathy = generateBathymetry('sandbar', 7);
    const teacher = createTeacher(bathy, { amplitudeM: amplitude });
    const w = bathy.width;

    const steps10 = Math.round(10 / DT);
    const steps30 = Math.round(30 / DT);
    const steps120 = Math.round(120 / DT);

    let varianceAt10 = 0;
    let maxAt30 = 0;
    let maxAt120 = 0;

    for (let s = 1; s <= steps120; s++) {
      teacher.step(DT);
      if (s === steps10) {
        let sum = 0;
        let sumSq = 0;
        let count = 0;
        for (let y = 2; y <= 10; y++) {
          const row = y * w;
          for (let x = 0; x < w; x++) {
            const idx = row + x;
            if (bathy.depth[idx] > 0) {
              const val = teacher.eta[idx];
              sum += val;
              sumSq += val * val;
              count++;
            }
          }
        }
        const mean = sum / count;
        varianceAt10 = sumSq / count - mean * mean;
      } else if (s === steps30) {
        maxAt30 = maxAbs(teacher.eta).max;
      } else if (s === steps120) {
        maxAt120 = maxAbs(teacher.eta).max;
      }
    }

    expect(varianceAt10).toBeGreaterThan(0);
    expect(maxAt120).toBeLessThanOrEqual(maxAt30 * 1.5 + amplitude);
  });

  it('determinism: two identical teachers produce exact equal eta checksums at t = 60 s', () => {
    const bathyA = generateBathymetry('sandbar', 7);
    const bathyB = generateBathymetry('sandbar', 7);
    const opts = { amplitudeM: 0.9, periodS: 9, directionDeg: 10 };
    const teacherA = createTeacher(bathyA, opts);
    const teacherB = createTeacher(bathyB, opts);

    const steps = Math.round(60 / DT);
    for (let s = 0; s < steps; s++) {
      teacherA.step(DT);
      teacherB.step(DT);
    }

    let sumA = 0;
    let sumB = 0;
    for (let i = 0; i < teacherA.eta.length; i++) {
      sumA += teacherA.eta[i];
      sumB += teacherB.eta[i];
    }
    expect(sumA).toBe(sumB);
  });
});
