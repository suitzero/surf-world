import { describe, expect, it } from 'vitest';
import type { BathymetryKind } from '../src/contracts/types.ts';
import { generateBathymetry } from '../src/sim/bathymetry.ts';
import { createPartitionedTeacher } from '../src/sim/partition.ts';
import { createTeacher } from '../src/sim/teacher.ts';

const DT = 0.05;

function compareEta(a: Float32Array, b: Float32Array): { maxAbsDiff: number; mse: number } {
  let maxAbsDiff = 0;
  let sumSq = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = Math.abs(a[i] - b[i]);
    if (diff > maxAbsDiff) maxAbsDiff = diff;
    sumSq += diff * diff;
  }
  return { maxAbsDiff, mse: sumSq / a.length };
}

describe('partitioned 2D linear shallow-water teacher solver', () => {
  const kinds: readonly BathymetryKind[] = ['sandbar', 'beach'];

  for (const kind of kinds) {
    it(
      `correctness vs monolithic teacher on ${kind} (seed 7, 30 s, P in {1, 2, 4})`,
      () => {
        const bathy = generateBathymetry(kind, 7);
        const mono = createTeacher(bathy);
        const steps = 600;
        for (let s = 0; s < steps; s++) {
          mono.step(DT);
        }

        for (const p of [1, 2, 4]) {
          const part = createPartitionedTeacher(bathy, { partitions: p });
          let totalSubsteps = 0;
          for (let s = 0; s < steps; s++) {
            part.step(DT);
            totalSubsteps += part.lastSubsteps;
          }

          const { maxAbsDiff, mse } = compareEta(part.eta, mono.eta);
          console.log(
            `[correctness] kind=${kind} P=${p} maxAbsDiff=${maxAbsDiff} mse=${mse} substeps=${totalSubsteps}`
          );
          expect(maxAbsDiff).toBeLessThanOrEqual(1e-3);

          // Accounting consistency check in the P=4 sandbar correctness run.
          if (kind === 'sandbar' && p === 4) {
            const expectedExchanges = totalSubsteps * (p - 1) * 2;
            const expectedBytes = expectedExchanges * bathy.height * 4;
            expect(part.stats.exchanges).toBe(expectedExchanges);
            expect(part.stats.bytesExchanged).toBe(expectedBytes);
          }
        }
      },
      300_000
    );
  }

  it(
    'cost at 128x96 on sandbar (seed 7, 60 s, P in {1, 2, 4})',
    () => {
      const bathy = generateBathymetry('sandbar', 7);
      const steps = 1200;
      const rows: Array<{
        P: number;
        totalMs: number;
        computeMs: number;
        exchangeMs: number;
        exchangeSharePct: number;
      }> = [];

      for (const p of [1, 2, 4]) {
        const part = createPartitionedTeacher(bathy, { partitions: p });
        const t0 = performance.now();
        for (let s = 0; s < steps; s++) {
          part.step(DT);
        }
        const totalMs = performance.now() - t0;
        const { computeMs, exchangeMs } = part.stats;
        const exchangeSharePct = totalMs > 0 ? (exchangeMs / totalMs) * 100 : 0;

        rows.push({ P: p, totalMs, computeMs, exchangeMs, exchangeSharePct });

        expect(totalMs).toBeGreaterThan(0);
        expect(computeMs + exchangeMs).toBeLessThanOrEqual(totalMs * 1.2);
      }

      console.log('[cost 128x96 sandbar 60s]');
      console.log('P | totalMs | computeMs | exchangeMs | exchangeShare%');
      for (const r of rows) {
        console.log(
          `${r.P} | ${r.totalMs.toFixed(2)} | ${r.computeMs.toFixed(2)} | ${r.exchangeMs.toFixed(2)} | ${r.exchangeSharePct.toFixed(2)}%`
        );
      }
    },
    300_000
  );

  it(
    'crossover at 512x384 on beach (seed 7, cellM=1, 30 s, P in {1, 4})',
    () => {
      const bathy = generateBathymetry('beach', 7, 512, 384);
      const steps = 600;
      const results = new Map<number, { eta: Float32Array; totalMs: number; computeMs: number; exchangeMs: number }>();

      console.log('[crossover 512x384 beach cellM=1 30s]');
      console.log('P | totalMs | computeMs | exchangeMs | exchangeShare% | exchanges | bytesExchanged');

      for (const p of [1, 4]) {
        const part = createPartitionedTeacher(bathy, { cellM: 1, partitions: p });
        const t0 = performance.now();
        for (let s = 0; s < steps; s++) {
          part.step(DT);
        }
        const totalMs = performance.now() - t0;
        const { computeMs, exchangeMs, exchanges, bytesExchanged } = part.stats;
        const exchangeSharePct = totalMs > 0 ? (exchangeMs / totalMs) * 100 : 0;

        console.log(
          `${p} | ${totalMs.toFixed(2)} | ${computeMs.toFixed(2)} | ${exchangeMs.toFixed(2)} | ${exchangeSharePct.toFixed(2)}% | ${exchanges} | ${bytesExchanged}`
        );

        results.set(p, { eta: part.eta, totalMs, computeMs, exchangeMs });
      }

      const r1 = results.get(1)!;
      const r4 = results.get(4)!;
      const { maxAbsDiff, mse } = compareEta(r1.eta, r4.eta);
      console.log(`[crossover 512x384 diff P=1 vs P=4] maxAbsDiff=${maxAbsDiff} mse=${mse}`);
      expect(maxAbsDiff).toBeLessThanOrEqual(1e-3);
    },
    600_000
  );

  it('invalid partitions throws RangeError', () => {
    const bathy = generateBathymetry('sandbar', 7);
    expect(() => createPartitionedTeacher(bathy, { partitions: 0 })).toThrow(RangeError);
    expect(() => createPartitionedTeacher(bathy, { partitions: bathy.width + 1 })).toThrow(
      RangeError
    );
  });
});
