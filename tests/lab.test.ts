import { describe, expect, it } from 'vitest';
import type { SwellParams } from '../src/contracts/types.ts';
import {
  computeOnsetRows,
  etaPixels,
  foldTimeMax,
  hybridPixels,
  physicsPixels,
} from '../src/lab/labRender.ts';
import { generateBathymetry } from '../src/sim/bathymetry.ts';
import { planTick } from '../src/sim/governor.ts';
import type { TileStateInput } from '../src/sim/governor.ts';
import { createHybrid, SOLVER_DT } from '../src/sim/hybrid.ts';
import { buildOcean, updateOcean } from '../src/sim/ocean.ts';
import { createTeacher } from '../src/sim/teacher.ts';

const SWELL: SwellParams = { heightM: 1.8, periodS: 9, directionDeg: 0 };

function diffFrac(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) diff++;
  }
  return diff / a.length;
}

describe('Ocean Lab', () => {
  it('three lab modes render visibly different fields', () => {
    const bathy = generateBathymetry('sandbar', 7);
    const { width: w, height: h } = bathy;

    const field = buildOcean(bathy, SWELL, 0);
    for (const t of [28, 28.5, 29, 29.5]) {
      updateOcean(field, SWELL, t, w, h, 0.5);
    }
    const phys = physicsPixels(field, bathy.depth, w, h);

    const teacher = createTeacher(bathy, { amplitudeM: 0.9, periodS: 9, directionDeg: 0 });
    const timeMax = new Float32Array(w * h);
    for (let s = 0; s < 600; s++) {
      teacher.step(0.05);
      foldTimeMax(timeMax, teacher.eta);
    }
    const eta = etaPixels(teacher.eta, bathy.depth, w, h, 0.9);

    const hybrid = createHybrid(bathy, SWELL, 0);
    hybrid.pumpAccurate(30);
    hybrid.correctTiles();
    const hyb = hybridPixels(hybrid, bathy.depth, w, h);

    expect(phys.length).toBe(w * h * 4);
    expect(eta.length).toBe(w * h * 4);
    expect(hyb.length).toBe(w * h * 4);
    expect(diffFrac(phys, eta)).toBeGreaterThan(0.05);
    expect(diffFrac(phys, hyb)).toBeGreaterThan(0.05);
    expect(diffFrac(eta, hyb)).toBeGreaterThan(0.05);
  });

  it('computeOnsetRows finds onset on most wet columns after 60 s', () => {
    const bathy = generateBathymetry('sandbar', 7);
    const { width: w, height: h } = bathy;
    const teacher = createTeacher(bathy, { amplitudeM: 0.9, periodS: 9, directionDeg: 0 });
    const timeMax = new Float32Array(w * h);
    for (let s = 0; s < 1200; s++) {
      teacher.step(0.05);
      foldTimeMax(timeMax, teacher.eta);
    }
    const rows = computeOnsetRows(timeMax, bathy.depth, w, h);
    const valid = rows.filter((r) => r >= 0);
    expect(valid.length).toBeGreaterThanOrEqual(70);
    expect(valid.every((r) => r < h)).toBe(true);
  });

  it('budget dial changes hybrid correction coverage', { timeout: 120000 }, () => {
    const bathy = generateBathymetry('sandbar', 7);

    function run(budget: number, seconds: number): number {
      const hybrid = createHybrid(bathy, SWELL, 0);
      const costs = { teacherStepMs: 0.55, tileApplyMs: 0.06 };
      let banked = 0;
      for (let s = 0; s < seconds; s++) {
        const tiles: TileStateInput[] = hybrid.tilesSnapshot().map((t) => ({
          id: t.id,
          correctedAt: t.correctedAt,
          queries: t.queries,
          eligible: t.eligible,
        }));
        const plan = planTick({
          budgetMs: budget,
          bankedMs: banked,
          now: hybrid.time,
          tiles,
          costs,
        });
        banked = plan.bankedMs;
        const job = hybrid.startCorrection(plan.tilesToCorrect);
        if (plan.teacherSteps > 0) hybrid.pumpAccurate(plan.teacherSteps * SOLVER_DT);
        hybrid.finishCorrection(job);
        hybrid.advanceWorld(1);
      }
      let corrected = 0;
      for (let i = 0; i < hybrid.tileCount; i++) {
        if (hybrid.getTile(i).correctedVersion >= 0) corrected++;
      }
      return corrected / hybrid.tileCount;
    }

    // Deterministic under the fixed cost table above (planTick is pure):
    // 0.5 ms/s starves the teacher (S16 cold-start), 10 ms/s reaches exactly
    // half the tiles in 25 s and keeps growing — the dial visibly matters.
    expect(run(0.5, 25)).toBe(0);
    expect(run(10, 25)).toBeGreaterThanOrEqual(0.5);
    expect(run(10, 60)).toBeGreaterThan(run(10, 25));
  });
});
