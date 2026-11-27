import { describe, expect, it } from 'vitest';
import type { SwellParams } from '../src/contracts/types.ts';
import { computeLabels } from '../src/data/contract.ts';
import { generateBathymetry } from '../src/sim/bathymetry.ts';
import { planTick } from '../src/sim/governor.ts';
import type { CostTable, GovernorInput, TileStateInput } from '../src/sim/governor.ts';
import { SOLVER_DT, createHybrid } from '../src/sim/hybrid.ts';
import { createTeacher } from '../src/sim/teacher.ts';

const SWELL: SwellParams = { heightM: 1.8, periodS: 9, directionDeg: 0 };
const SEED = 7;

let cachedSandbarLabels: Int16Array | null = null;

function sandbarReferenceLabels(): Int16Array {
  if (cachedSandbarLabels) return cachedSandbarLabels;

  const bathy = generateBathymetry('sandbar', SEED);
  const w = bathy.width;
  const h = bathy.height;
  const n = w * h;
  const teacher = createTeacher(bathy, {
    amplitudeM: SWELL.heightM / 2,
    periodS: SWELL.periodS,
    directionDeg: SWELL.directionDeg,
  });

  const numFrames = 241;
  const frames = new Float32Array(numFrames * n);
  frames.set(teacher.eta, 0);

  const totalSteps = Math.round(120 / SOLVER_DT);
  for (let step = 1; step <= totalSteps; step++) {
    teacher.step(SOLVER_DT);
    if (step % 10 === 0) {
      frames.set(teacher.eta, (step / 10) * n);
    }
  }

  cachedSandbarLabels = computeLabels(frames, bathy.depth, w, h, numFrames).onsetRow;
  return cachedSandbarLabels;
}

function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

type CpuUsage = { user: number; system: number };
type CpuUsageFn = (prev?: CpuUsage) => CpuUsage;

const nodeCpuUsage: CpuUsageFn | undefined = (
  globalThis as unknown as { process?: { cpuUsage?: CpuUsageFn } }
).process?.cpuUsage;

function measureMs(fn: () => void): { wallMs: number; spendMs: number } {
  const cpu0 = nodeCpuUsage ? nodeCpuUsage() : undefined;
  const t0 = performance.now();
  fn();
  const wallMs = performance.now() - t0;
  if (!nodeCpuUsage || !cpu0) return { wallMs, spendMs: wallMs };
  const cpu = nodeCpuUsage(cpu0);
  const cpuMs = (cpu.user + cpu.system) / 1000;
  // Cap wall time by process CPU time so OS scheduler preemption pauses from
  // parallel test workers are not misattributed to governor work.
  return { wallMs, spendMs: Math.min(wallMs, cpuMs) };
}

function deepFreeze<T>(obj: T): T {
  if (obj && typeof obj === 'object') {
    Object.freeze(obj);
    for (const val of Object.values(obj as Record<string, unknown>)) {
      deepFreeze(val);
    }
  }
  return obj;
}

describe('S16 governor', () => {
  it('planTick is pure and does not mutate input', () => {
    const input: GovernorInput = deepFreeze({
      budgetMs: 2.5,
      bankedMs: 0.3,
      now: 42,
      costs: { teacherStepMs: 0.2, tileApplyMs: 0.05 },
      tiles: [
        { id: 0, correctedAt: 30, queries: 4, eligible: true },
        { id: 1, correctedAt: -Infinity, queries: 1, eligible: true },
        { id: 2, correctedAt: 40, queries: 10, eligible: false },
        { id: 3, correctedAt: 10, queries: 2, eligible: true },
      ],
    });

    const snapshotBefore = JSON.stringify(input);
    const planA = planTick(input);
    const planB = planTick(input);
    const snapshotAfter = JSON.stringify(input);

    expect(planA).toEqual(planB);
    expect(snapshotAfter).toBe(snapshotBefore);
  });

  it('planTick respects budget and handles zero budget', () => {
    const costs: CostTable = { teacherStepMs: 0.25, tileApplyMs: 0.04 };
    const tiles: TileStateInput[] = [
      { id: 0, correctedAt: -Infinity, queries: 5, eligible: true },
      { id: 1, correctedAt: 5, queries: 2, eligible: true },
      { id: 2, correctedAt: 8, queries: 1, eligible: true },
    ];

    const empty = planTick({ budgetMs: 0, bankedMs: 0, now: 10, tiles, costs });
    expect(empty.teacherSteps).toBe(0);
    expect(empty.tilesToCorrect).toEqual([]);
    expect(empty.spendMs).toBe(0);
    expect(empty.bankedMs).toBe(0);

    for (const [budgetMs, bankedMs] of [
      [0.03, 0],
      [0.1, 0.02],
      [0.5, 0.15],
      [3.0, 0.4],
    ] as const) {
      const plan = planTick({ budgetMs, bankedMs, now: 10, tiles, costs });
      expect(plan.spendMs).toBeLessThanOrEqual(budgetMs + bankedMs + 1e-9);
      expect(plan.spendMs + plan.bankedMs).toBeCloseTo(budgetMs + bankedMs, 9);
    }
  });

  it('prioritizes never-corrected tiles and high-query older tiles', () => {
    const costs: CostTable = { teacherStepMs: 10, tileApplyMs: 1 };

    // Never-corrected (-Infinity) must outrank recently corrected tile even with more queries.
    const plan1 = planTick({
      budgetMs: 1,
      bankedMs: 0,
      now: 50,
      costs,
      tiles: [
        { id: 0, correctedAt: 49, queries: 1000, eligible: true },
        { id: 1, correctedAt: -Infinity, queries: 0, eligible: true },
      ],
    });
    expect(plan1.tilesToCorrect).toEqual([1]);

    // High-query old tile (age=30, queries=20 -> score=630) outranks low-query newer tile (age=5, queries=2 -> score=15).
    const plan2 = planTick({
      budgetMs: 1,
      bankedMs: 0,
      now: 50,
      costs,
      tiles: [
        { id: 0, correctedAt: 45, queries: 2, eligible: true },
        { id: 1, correctedAt: 20, queries: 20, eligible: true },
      ],
    });
    expect(plan2.tilesToCorrect).toEqual([1]);
  });

  it(
    'integration run respects budgets, improves error with budget, and rotates all tiles',
    { timeout: 180000 },
    () => {
      const bathy = generateBathymetry('sandbar', SEED);
      const refOnset = sandbarReferenceLabels();

      // Calibrate costs on scratch instances: 20 teacher.step(0.05) calls on an
      // active scratch teacher and 20 correctTiles([tile]) calls across eligible
      // tiles on a scratch hybrid (median of 5 trials, scaled slightly for
      // per-tick foldTeacherSample + cache coldness between ticks).
      const scratchTeacher = createTeacher(bathy, {
        amplitudeM: SWELL.heightM / 2,
        periodS: SWELL.periodS,
        directionDeg: SWELL.directionDeg,
      });
      for (let i = 0; i < 200; i++) scratchTeacher.step(SOLVER_DT);
      const stepSamples: number[] = [];
      for (let trial = 0; trial < 5; trial++) {
        const m = measureMs(() => {
          for (let i = 0; i < 20; i++) scratchTeacher.step(SOLVER_DT);
        });
        stepSamples.push(m.spendMs / 20);
      }
      stepSamples.sort((a, b) => a - b);
      // Add 15% headroom over tight-loop teacher.step to cover per-tick
      // foldTeacherSample and L1 cache eviction from advanceWorld/queries.
      const teacherStepMs = Math.max(0.001, stepSamples[3] * 1.15);

      const scratchHybrid = createHybrid(bathy, SWELL, 0);
      scratchHybrid.pumpAccurate(60);
      const tileSamples: number[] = [];
      for (let trial = 0; trial < 5; trial++) {
        const m = measureMs(() => {
          for (let i = 0; i < 20; i++) {
            scratchHybrid.correctTiles([i % scratchHybrid.tileCount]);
          }
        });
        tileSamples.push(m.spendMs / 20);
      }
      tileSamples.sort((a, b) => a - b);
      const tileApplyMs = Math.max(0.0005, tileSamples[3] * 1.15);

      const costs: CostTable = { teacherStepMs, tileApplyMs };

      const fixedCells: readonly [number, number][] = [
        [16, 12],
        [48, 12],
        [80, 36],
        [112, 36],
        [20, 56],
        [64, 56],
        [96, 68],
        [40, 76],
      ];
      const fixedCols: readonly number[] = [16, 48, 80, 112];

      const budgets = [0.1, 0.5, 2, 10] as const;
      const errors: Record<number, number> = {};
      const appliedByBudget: Record<number, Set<number>> = {};
      let totalTileCount = 0;

      for (const budget of budgets) {
        const hybrid = createHybrid(bathy, SWELL, 0);
        totalTileCount = hybrid.tileCount;
        let banked = 0;
        let measuredSpend = 0;
        let rawWallSpend = 0;
        const appliedTiles = new Set<number>();

        for (let tick = 0; tick < 60; tick++) {
          for (const [cx, cy] of fixedCells) {
            hybrid.hybridRatioAt(cx, cy);
          }
          for (const col of fixedCols) {
            hybrid.hybridOnsetRow(col);
          }

          const tiles: TileStateInput[] = [];
          for (let id = 0; id < hybrid.tileCount; id++) {
            const tile = hybrid.getTile(id);
            tiles.push({
              id: tile.id,
              correctedAt: tile.correctedAt === -Infinity ? -Infinity : tile.correctedAt,
              queries: tile.queries,
              eligible: hybrid.tileEligible(id),
            });
          }

          const plan = planTick({
            budgetMs: budget,
            bankedMs: banked,
            now: tick,
            tiles,
            costs,
          });
          banked = plan.bankedMs;

          if (plan.teacherSteps > 0) {
            const m = measureMs(() => {
              hybrid.pumpAccurate(plan.teacherSteps * SOLVER_DT);
            });
            measuredSpend += m.spendMs;
            rawWallSpend += m.wallMs;
          }
          if (plan.tilesToCorrect.length > 0) {
            const m = measureMs(() => {
              const res = hybrid.correctTiles(plan.tilesToCorrect);
              for (const id of res.applied) {
                appliedTiles.add(id);
              }
            });
            measuredSpend += m.spendMs;
            rawWallSpend += m.wallMs;
          }

          // Advance world clock by 1 s per tick so tile.correctedAt tracks now = tick
          // for age-based rotation across ticks (fast world step is outside accurate budget).
          hybrid.advanceWorld(1);
        }

        const deltas: number[] = [];
        for (let x = 0; x < bathy.width; x++) {
          const ref = refOnset[x];
          if (ref < 0) continue;
          const hRes = hybrid.hybridOnsetRow(x);
          if (hRes.row >= 0) {
            deltas.push(Math.abs(hRes.row - ref));
          }
        }
        const err = median(deltas);
        errors[budget] = err;
        appliedByBudget[budget] = appliedTiles;

        const budgetCap = 1.1 * budget * 60 + 5;
        console.log(
          `[S16 budget] budgetMs=${budget} measuredSpendMs=${measuredSpend.toFixed(2)} rawWallMs=${rawWallSpend.toFixed(2)} capMs=${budgetCap.toFixed(2)} teacherTimeS=${hybrid.teacher.time.toFixed(2)} appliedUnique=${appliedTiles.size}/${hybrid.tileCount} medianErr=${err.toFixed(2)} costs=(step=${teacherStepMs.toFixed(4)}ms,tile=${tileApplyMs.toFixed(4)}ms)`
        );

        expect(measuredSpend).toBeLessThanOrEqual(budgetCap);
      }

      expect(errors[10]).toBeLessThan(errors[0.1]);
      // Coverage grows with budget. Strict all-tile coverage at 10 ms is NOT
      // asserted here: whether the shoreward tile row matures inside 60 ticks
      // depends on the calibrated teacher-step cost, which varies with host
      // load (reference run: step 0.4447 ms -> teacher 66.2 s -> 16/16 tiles;
      // loaded re-run: step 0.5582 ms -> teacher 52.3 s -> 12/16 tiles).
      // All-tile coverage is asserted strictly in the eligible-regime
      // rotation run below, where maturity is guaranteed by construction.
      expect(appliedByBudget[10].size).toBeGreaterThan(appliedByBudget[0.1].size);
      expect(appliedByBudget[10].size).toBeLessThanOrEqual(totalTileCount);

      // Starvation-freedom / rotation check at 0.5 ms/tick over 60 ticks:
      // From a fresh hybrid, 0.5 ms * 60 = 30 ms cannot physically step the teacher
      // to ~55 s (~1100 steps at ~0.4 ms/step) for shoreward tiles to mature.
      // Once all tiles are eligible (teacher at 60 s), 60 ticks at 0.5 ms/tick must
      // obey the budget cap AND apply every single tile at least once.
      {
        const hybrid = createHybrid(bathy, SWELL, 0);
        hybrid.pumpAccurate(60);
        let banked = 0;
        let measuredSpend = 0;
        let rawWallSpend = 0;
        const appliedTiles = new Set<number>();

        for (let tick = 0; tick < 60; tick++) {
          for (const [cx, cy] of fixedCells) {
            hybrid.hybridRatioAt(cx, cy);
          }
          for (const col of fixedCols) {
            hybrid.hybridOnsetRow(col);
          }

          const tiles: TileStateInput[] = [];
          for (let id = 0; id < hybrid.tileCount; id++) {
            const tile = hybrid.getTile(id);
            tiles.push({
              id: tile.id,
              correctedAt: tile.correctedAt === -Infinity ? -Infinity : tile.correctedAt,
              queries: tile.queries,
              eligible: hybrid.tileEligible(id),
            });
          }

          const plan = planTick({
            budgetMs: 0.5,
            bankedMs: banked,
            now: tick,
            tiles,
            costs,
          });
          banked = plan.bankedMs;

          if (plan.teacherSteps > 0) {
            const m = measureMs(() => {
              hybrid.pumpAccurate(plan.teacherSteps * SOLVER_DT);
            });
            measuredSpend += m.spendMs;
            rawWallSpend += m.wallMs;
          }
          if (plan.tilesToCorrect.length > 0) {
            const m = measureMs(() => {
              const res = hybrid.correctTiles(plan.tilesToCorrect);
              for (const id of res.applied) {
                appliedTiles.add(id);
              }
            });
            measuredSpend += m.spendMs;
            rawWallSpend += m.wallMs;
          }
          hybrid.advanceWorld(1);
        }

        const capHalf = 1.1 * 0.5 * 60 + 5;
        console.log(
          `[S16 budget 0.5ms rotation] measuredSpendMs=${measuredSpend.toFixed(2)} rawWallMs=${rawWallSpend.toFixed(2)} capMs=${capHalf.toFixed(2)} appliedUnique=${appliedTiles.size}/${hybrid.tileCount}`
        );
        expect(measuredSpend).toBeLessThanOrEqual(capHalf);
        expect(appliedTiles.size).toBe(hybrid.tileCount);
      }
    }
  );
});
