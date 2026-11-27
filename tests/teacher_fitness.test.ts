/**
 * S07d pre-registered teacher-fitness test comparing the 2D linear
 * shallow-water teacher solver (src/sim/teacher.ts, createTeacher) against the
 * eikonal ocean model (src/sim/ocean.ts, buildOcean) using an onset-line method.
 *
 * Honest statement of the method:
 * - This onset-line proxy was pre-registered after the S07c zone-centroid proxy
 *   failed.
 * - The linear shallow-water teacher has NO wave-breaking model or surf-zone
 *   dissipation, so its breaking onset is defined per column as the first
 *   offshore-to-shore crossing of 2 * |eta|_max / depth >= 0.78 (BREAKING_RATIO)
 *   over the steady window t >= 84 s.
 * - The ocean onset is defined per column as the first offshore-to-shore
 *   crossing of the static shoaled height field amp / depthEff >= 0.78
 *   (corresponding to the set-envelope peak, env max = 1.0, with amp > 0).
 * - Crest timing in a periodic wave train is only defined modulo the period
 *   T = 9 s, so the pre-registered timing bar applies to the phase-wrapped
 *   median while the unwrapped median is also reported.
 * - Pairing: crests are collected over the full run (t >= 2 s, skipping the
 *   startup ramp) and paired by index — onset crest j is boundary crest j
 *   propagated shoreward (the forced train is coherent, one crest per
 *   period). Deltas are accumulated only for pairs whose onset crest falls
 *   in the steady window. An earlier draft paired each onset crest with the
 *   latest in-window boundary crest, which silently reduced the teacher
 *   travel time modulo T and inflated the "unwrapped" statistic; index
 *   pairing was verified against an independent probe (sandbar x=32/64/96:
 *   travel 16.6-18.2 s, groupTau 20.2-21.9 s, delta -3.5..-4.0 s).
 */
import { describe, expect, it } from 'vitest';
import type { BathymetryKind, SwellParams } from '../src/contracts/types.ts';
import { generateBathymetry } from '../src/sim/bathymetry.ts';
import { BREAKING_RATIO, buildOcean } from '../src/sim/ocean.ts';
import { createTeacher } from '../src/sim/teacher.ts';

const DT = 0.05;
const TOTAL_S = 120;
const TOTAL_STEPS = Math.round(TOTAL_S / DT); // 2400
const WINDOW_START_S = 84;
const SEED = 7;
const TIDE_M = 0;
const PERIOD_S = 9;

const SWELL: SwellParams = {
  heightM: 1.8,
  periodS: PERIOD_S,
  directionDeg: 0,
};

const TEACHER_OPTS = {
  amplitudeM: SWELL.heightM / 2, // 0.9 m
  periodS: PERIOD_S,
  directionDeg: 0,
};

const KINDS: readonly BathymetryKind[] = ['sandbar', 'beach', 'point', 'reef'];

function median(values: readonly number[]): number {
  const n = values.length;
  if (n === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(n / 2);
  return n % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

function percentile90(values: readonly number[]): number {
  const n = values.length;
  if (n === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.ceil(0.9 * n) - 1;
  return sorted[idx];
}

function wrapPeriod(delta: number, periodS: number): number {
  const r = ((Math.abs(delta) % periodS) + periodS) % periodS;
  return Math.min(r, periodS - r);
}

/** Local maxima of the full-run series, skipping the 1 s startup ramp. */
function crestTimesFullRun(series: Float64Array): number[] {
  const times: number[] = [];
  for (let s = 2; s <= TOTAL_STEPS - 1; s++) {
    const t = s * DT;
    if (t < 2 || t > TOTAL_S) continue;
    if (series[s] > series[s - 1] && series[s] >= series[s + 1]) {
      times.push(t);
    }
  }
  return times;
}

describe('S07d teacher fitness (onset-line and crest timing)', () => {
  for (const kind of KINDS) {
    it(
      `evaluates onset-line agreement and crest timing on ${kind} (seed ${SEED})`,
      { timeout: 120000 },
      () => {
        const bathy = generateBathymetry(kind, SEED);
        const w = bathy.width;
        const h = bathy.height;
        const n = w * h;

        // Ocean onset (static, no time stepping).
        const field = buildOcean(bathy, SWELL, TIDE_M);
        const oceanOnset = new Int32Array(w).fill(-1);
        for (let x = 0; x < w; x++) {
          for (let y = 0; y < h; y++) {
            const i = y * w + x;
            if (field.amp[i] > 0 && field.amp[i] / field.depthEff[i] >= BREAKING_RATIO) {
              oceanOnset[x] = y;
              break;
            }
          }
        }

        // Teacher pass 1: step 2400 times, accumulate steady-window max |eta|.
        const teacherPass1 = createTeacher(bathy, TEACHER_OPTS);
        const timeMaxEta = new Float64Array(n);
        for (let s = 1; s <= TOTAL_STEPS; s++) {
          const t = s * DT;
          teacherPass1.step(DT);
          if (t >= WINDOW_START_S) {
            const eta = teacherPass1.eta;
            for (let i = 0; i < n; i++) {
              const absEta = Math.abs(eta[i]);
              if (absEta > timeMaxEta[i]) timeMaxEta[i] = absEta;
            }
          }
        }

        const teacherOnset = new Int32Array(w).fill(-1);
        for (let x = 0; x < w; x++) {
          for (let y = 0; y < h; y++) {
            const i = y * w + x;
            const d = bathy.depth[i];
            const ratio = d > 0 ? (2 * timeMaxEta[i]) / d : 0;
            if (ratio >= BREAKING_RATIO) {
              teacherOnset[x] = y;
              break;
            }
          }
        }

        // Onset stats across columns.
        let bothCount = 0;
        let teacherOnlyCount = 0;
        let oceanOnlyCount = 0;
        const absDeltaRows: number[] = [];
        const bothCols: number[] = [];

        for (let x = 0; x < w; x++) {
          const tY = teacherOnset[x];
          const oY = oceanOnset[x];
          if (tY >= 0 && oY >= 0) {
            bothCount++;
            bothCols.push(x);
            absDeltaRows.push(Math.abs(tY - oY));
          } else if (tY >= 0) {
            teacherOnlyCount++;
          } else if (oY >= 0) {
            oceanOnlyCount++;
          }
        }

        const fractionBoth = bothCount / w;
        const medianAbsRow = median(absDeltaRows);
        const p90AbsRow = percentile90(absDeltaRows);

        // Teacher pass 2 (deterministic re-run): record eta series at onset and boundary cells.
        const onsetCellIdx = new Int32Array(bothCols.length);
        const boundaryCellIdx = new Int32Array(bothCols.length);
        const onsetSeries: Float64Array[] = [];
        const boundarySeries: Float64Array[] = [];

        for (let c = 0; c < bothCols.length; c++) {
          const x = bothCols[c];
          const tY = teacherOnset[x];
          let yB = -1;
          for (let y = 0; y < h; y++) {
            if (bathy.depth[y * w + x] > 0) {
              yB = y;
              break;
            }
          }
          onsetCellIdx[c] = tY * w + x;
          boundaryCellIdx[c] = yB >= 0 ? yB * w + x : -1;
          onsetSeries.push(new Float64Array(TOTAL_STEPS + 1));
          boundarySeries.push(new Float64Array(TOTAL_STEPS + 1));
        }

        const teacherPass2 = createTeacher(bathy, TEACHER_OPTS);
        for (let s = 1; s <= TOTAL_STEPS; s++) {
          teacherPass2.step(DT);
          const eta = teacherPass2.eta;
          for (let c = 0; c < bothCols.length; c++) {
            onsetSeries[c][s] = eta[onsetCellIdx[c]];
            const bIdx = boundaryCellIdx[c];
            boundarySeries[c][s] = bIdx >= 0 ? eta[bIdx] : 0;
          }
        }

        const absDeltas: number[] = [];
        const wrappedDeltas: number[] = [];

        for (let c = 0; c < bothCols.length; c++) {
          const bIdx = boundaryCellIdx[c];
          if (bIdx < 0) continue;
          const oIdx = onsetCellIdx[c];
          const groupTravel = field.groupTau[oIdx] - field.groupTau[bIdx];
          const onsetCrests = crestTimesFullRun(onsetSeries[c]);
          const boundaryCrests = crestTimesFullRun(boundarySeries[c]);

          // Index pairing: onset crest j is boundary crest j propagated
          // shoreward. Only pairs landing in the steady window are scored.
          const pairCount = Math.min(onsetCrests.length, boundaryCrests.length);
          for (let j = 0; j < pairCount; j++) {
            const tc = onsetCrests[j];
            if (tc < WINDOW_START_S || tc > TOTAL_S) continue;
            const teacherTravel = tc - boundaryCrests[j];
            const delta = teacherTravel - groupTravel;
            absDeltas.push(Math.abs(delta));
            wrappedDeltas.push(wrapPeriod(delta, PERIOD_S));
          }
        }

        const nDeltas = absDeltas.length;
        const medianAbsUnwrapped = median(absDeltas);
        const medianAbsWrapped = median(wrappedDeltas);

        console.log(
          `[S07d ${kind}] ` +
            `both=${bothCount} ` +
            `teacherOnly=${teacherOnlyCount} ` +
            `oceanOnly=${oceanOnlyCount} ` +
            `fracBoth=${fractionBoth.toFixed(6)} ` +
            `medianAbsRow=${medianAbsRow.toFixed(3)} ` +
            `p90AbsRow=${p90AbsRow.toFixed(3)} ` +
            `timingN=${nDeltas} ` +
            `medianAbsTiming=${medianAbsUnwrapped.toFixed(4)} ` +
            `medianAbsTimingWrapped=${medianAbsWrapped.toFixed(4)}`
        );

        // Pre-registered assertions (strict; failing tests are records).
        expect(bothCount).toBeGreaterThan(0);
        expect(Number.isFinite(fractionBoth)).toBe(true);
        expect(Number.isFinite(medianAbsRow)).toBe(true);
        expect(Number.isFinite(p90AbsRow)).toBe(true);
        expect(Number.isFinite(nDeltas)).toBe(true);
        expect(Number.isFinite(medianAbsUnwrapped)).toBe(true);
        expect(Number.isFinite(medianAbsWrapped)).toBe(true);

        if (kind === 'sandbar' || kind === 'beach') {
          expect(medianAbsRow).toBeLessThanOrEqual(3);
          expect(medianAbsWrapped).toBeLessThanOrEqual(1.0);
        }
      }
    );
  }
});
