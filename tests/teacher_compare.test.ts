/**
 * S07c comparison between the 2D linear shallow-water teacher solver
 * (src/sim/teacher.ts, createTeacher) and the game's eikonal ocean
 * approximation (src/sim/ocean.ts, buildOcean / updateOcean).
 *
 * Honest statement of the breaking-zone proxy:
 * - The linear shallow-water teacher has NO wave-breaking model or surf-zone
 *   dissipation; its breaking-zone proxy is the shoaling amplification zone of
 *   the steady-window time-max |eta| / depth field.
 * - The ocean mask is the eikonal model's active breaking flag (`field.breaking`).
 * - Agreement is judged on the cross-shore (y-row) centroid of the teacher's
 *   50%-of-peak amplification zone versus the ocean's ever-breaking mask.
 */
import { describe, expect, it } from 'vitest';
import type { BathymetryKind, SwellParams } from '../src/contracts/types.ts';
import { generateBathymetry } from '../src/sim/bathymetry.ts';
import { buildOcean, updateOcean } from '../src/sim/ocean.ts';
import { createTeacher } from '../src/sim/teacher.ts';

const DT = 0.05;
const TOTAL_S = 120;
const TOTAL_STEPS = Math.round(TOTAL_S / DT); // 2400
const WINDOW_START_S = 84;
const SEED = 7;
const TIDE_M = 0;

const SWELL: SwellParams = {
  heightM: 1.8,
  periodS: 9,
  directionDeg: 0,
};

const KINDS: readonly BathymetryKind[] = ['sandbar', 'beach', 'point', 'reef'];

function zoneStats(mask: Uint8Array, w: number, h: number): { count: number; centroidRow: number } {
  let count = 0;
  let sumY = 0;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      if (mask[row + x]) {
        sumY += y;
        count++;
      }
    }
  }
  return {
    count,
    centroidRow: count > 0 ? sumY / count : Number.NaN,
  };
}

function maskIoU(a: Uint8Array, b: Uint8Array): number {
  let intersection = 0;
  let union = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i] !== 0;
    const bi = b[i] !== 0;
    if (ai && bi) intersection++;
    if (ai || bi) union++;
  }
  return union > 0 ? intersection / union : Number.NaN;
}

function pearsonCorr(a: Float64Array, b: Float64Array): number {
  const n = a.length;
  if (n === 0 || n !== b.length) return Number.NaN;
  let sumA = 0;
  let sumB = 0;
  for (let i = 0; i < n; i++) {
    sumA += a[i];
    sumB += b[i];
  }
  const meanA = sumA / n;
  const meanB = sumB / n;
  let num = 0;
  let denA = 0;
  let denB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    num += da * db;
    denA += da * da;
    denB += db * db;
  }
  if (denA <= 0 || denB <= 0) return Number.NaN;
  return num / Math.sqrt(denA * denB);
}

describe('S07c linear teacher vs eikonal ocean comparison', () => {
  for (const kind of KINDS) {
    it(
      `compares breaking-zone proxy and ocean breaking mask on ${kind} (seed ${SEED})`,
      { timeout: 180000 },
      () => {
        const bathy = generateBathymetry(kind, SEED);
        const w = bathy.width;
        const h = bathy.height;
        const n = w * h;

        // Teacher side: step DT=0.05 for 120 s (2400 steps), accumulate time-max |eta| for t >= 84 s.
        const teacher = createTeacher(bathy, {
          amplitudeM: 0.9,
          periodS: 9,
          directionDeg: 0,
        });
        const timeMaxEta = new Float64Array(n);

        for (let s = 1; s <= TOTAL_STEPS; s++) {
          const t = s * DT;
          teacher.step(DT);
          if (t >= WINDOW_START_S) {
            const eta = teacher.eta;
            for (let i = 0; i < n; i++) {
              const absEta = Math.abs(eta[i]);
              if (absEta > timeMaxEta[i]) timeMaxEta[i] = absEta;
            }
          }
        }

        const ampRatio = new Float64Array(n);
        let maxAmpRatio = 0;
        for (let i = 0; i < n; i++) {
          const d = bathy.depth[i];
          if (d > 0) {
            const r = timeMaxEta[i] / d;
            ampRatio[i] = r;
            if (r > maxAmpRatio) maxAmpRatio = r;
          }
        }

        const teacherA = new Uint8Array(n);
        const teacherB = new Uint8Array(n);
        const halfPeak = 0.5 * maxAmpRatio;
        for (let i = 0; i < n; i++) {
          const r = ampRatio[i];
          if (maxAmpRatio > 0 && r >= halfPeak) teacherA[i] = 1;
          if (r >= 0.78) teacherB[i] = 1;
        }

        // Ocean side: step s = 1..2400, t = s * DT, accumulate everBreaking and breakCount for t >= 84 s.
        const field = buildOcean(bathy, SWELL, TIDE_M);
        const everBreaking = new Uint8Array(n);
        const breakCount = new Uint32Array(n);
        let windowSteps = 0;

        for (let s = 1; s <= TOTAL_STEPS; s++) {
          const t = s * DT;
          updateOcean(field, SWELL, t, w, h, DT);
          if (t >= WINDOW_START_S) {
            windowSteps++;
            const breaking = field.breaking;
            for (let i = 0; i < n; i++) {
              if (breaking[i] === 1) {
                everBreaking[i] = 1;
                breakCount[i]++;
              }
            }
          }
        }

        const oceanA = everBreaking;
        const oceanB = new Uint8Array(n);
        const minBreakCount = 0.05 * windowSteps;
        for (let i = 0; i < n; i++) {
          if (breakCount[i] >= minBreakCount) oceanB[i] = 1;
        }

        // Metrics per kind.
        const oceanAStats = zoneStats(oceanA, w, h);
        const oceanBStats = zoneStats(oceanB, w, h);
        const teacherAStats = zoneStats(teacherA, w, h);
        const teacherBStats = zoneStats(teacherB, w, h);

        const centroidDeltaA = teacherAStats.centroidRow - oceanAStats.centroidRow;
        const centroidDeltaB = teacherBStats.centroidRow - oceanBStats.centroidRow;
        const iouA = maskIoU(teacherA, oceanA);
        const iouB = maskIoU(teacherB, oceanB);

        const teacherCol = new Float64Array(h);
        const oceanCol = new Float64Array(h);
        for (let y = 0; y < h; y++) {
          const row = y * w;
          let rowMaxAmp = 0;
          let rowBreakCells = 0;
          for (let x = 0; x < w; x++) {
            const i = row + x;
            if (ampRatio[i] > rowMaxAmp) rowMaxAmp = ampRatio[i];
            if (everBreaking[i]) rowBreakCells++;
          }
          teacherCol[y] = rowMaxAmp;
          oceanCol[y] = rowBreakCells / w;
        }
        const colCorr = pearsonCorr(teacherCol, oceanCol);

        console.log(
          `[S07c ${kind}] ` +
            `oceanAcentroid=${oceanAStats.centroidRow.toFixed(3)} ` +
            `teacherAcentroid=${teacherAStats.centroidRow.toFixed(3)} ` +
            `deltaA=${centroidDeltaA.toFixed(3)} ` +
            `oceanBcentroid=${oceanBStats.centroidRow.toFixed(3)} ` +
            `teacherBcentroid=${teacherBStats.centroidRow.toFixed(3)} ` +
            `deltaB=${centroidDeltaB.toFixed(3)} ` +
            `iouA=${iouA.toFixed(6)} ` +
            `iouB=${iouB.toFixed(6)} ` +
            `colCorr=${colCorr.toFixed(6)} ` +
            `maxAmpRatio=${maxAmpRatio.toFixed(6)}`
        );

        // Assertions: all four zones non-empty for every kind; all metrics finite.
        expect(oceanAStats.count).toBeGreaterThan(0);
        expect(oceanBStats.count).toBeGreaterThan(0);
        expect(teacherAStats.count).toBeGreaterThan(0);
        expect(teacherBStats.count).toBeGreaterThan(0);

        expect(Number.isFinite(oceanAStats.centroidRow)).toBe(true);
        expect(Number.isFinite(oceanBStats.centroidRow)).toBe(true);
        expect(Number.isFinite(teacherAStats.centroidRow)).toBe(true);
        expect(Number.isFinite(teacherBStats.centroidRow)).toBe(true);
        expect(Number.isFinite(centroidDeltaA)).toBe(true);
        expect(Number.isFinite(centroidDeltaB)).toBe(true);
        expect(Number.isFinite(iouA)).toBe(true);
        expect(Number.isFinite(iouB)).toBe(true);
        expect(Number.isFinite(colCorr)).toBe(true);
        expect(Number.isFinite(maxAmpRatio)).toBe(true);

        if (kind === 'sandbar' || kind === 'beach') {
          expect(Math.abs(centroidDeltaA)).toBeLessThanOrEqual(3);
        }
      }
    );
  }
});
