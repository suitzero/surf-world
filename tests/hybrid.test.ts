import { describe, expect, it } from 'vitest';
import type { BathymetryKind, SwellParams } from '../src/contracts/types.ts';
import { computeLabels } from '../src/data/contract.ts';
import { generateBathymetry } from '../src/sim/bathymetry.ts';
import { SOLVER_DT, createHybrid } from '../src/sim/hybrid.ts';
import { createTeacher } from '../src/sim/teacher.ts';

const SWELL: SwellParams = { heightM: 1.8, periodS: 9, directionDeg: 0 };
const SEED = 7;

const refLabelsCache = new Map<BathymetryKind, Int16Array>();

function referenceLabels(kind: BathymetryKind): Int16Array {
  const cached = refLabelsCache.get(kind);
  if (cached) return cached;

  const bathy = generateBathymetry(kind, SEED);
  const w = bathy.width;
  const h = bathy.height;
  const n = w * h;
  const teacher = createTeacher(bathy, {
    amplitudeM: SWELL.heightM / 2,
    periodS: SWELL.periodS,
    directionDeg: SWELL.directionDeg,
  });

  const numFrames = 241; // t = 0, 0.5, ..., 120.0
  const frames = new Float32Array(numFrames * n);
  frames.set(teacher.eta, 0);

  const totalSteps = Math.round(120 / SOLVER_DT);
  for (let step = 1; step <= totalSteps; step++) {
    teacher.step(SOLVER_DT);
    if (step % 10 === 0) {
      const frame = step / 10;
      frames.set(teacher.eta, frame * n);
    }
  }

  const labels = computeLabels(frames, bathy.depth, w, h, numFrames);
  refLabelsCache.set(kind, labels.onsetRow);
  return labels.onsetRow;
}

function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

function fnv1a32(chunks: readonly Uint8Array[]): number {
  let hash = 0x811c9dc5;
  for (const bytes of chunks) {
    for (let i = 0; i < bytes.length; i++) {
      hash ^= bytes[i];
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  }
  return hash >>> 0;
}

describe('S15 hybrid simulator', () => {
  it('enforces version rule and drops stale corrections', { timeout: 180000 }, () => {
    const bathy = generateBathymetry('sandbar', SEED);
    const hybrid = createHybrid(bathy, SWELL, 0);

    const colX = 64;
    const fastRow = hybrid.fastOnsetRow(colX);
    expect(fastRow).toBeGreaterThanOrEqual(0);
    const tileA = hybrid.tileIdAt(colX, fastRow);
    const tileB = 0;
    expect(tileA).not.toBe(tileB);

    const job = hybrid.startCorrection();
    hybrid.pumpAccurate(120);
    hybrid.bumpVersion([tileA]);
    const result = hybrid.finishCorrection(job);

    expect(result.dropped).toContain(tileA);
    expect(hybrid.droppedStaleCorrections).toBeGreaterThanOrEqual(1);
    expect(result.applied.length).toBeGreaterThan(0);
    expect(result.applied).not.toContain(tileA);
    expect(result.applied).toContain(tileB);

    const queryInA = hybrid.hybridRatioAt(colX, fastRow);
    expect(queryInA.provenance).toBe('fast');

    const tileBInfo = hybrid.getTile(tileB);
    let foundMature = false;
    for (let y = tileBInfo.y0; y < tileBInfo.y1 && !foundMature; y++) {
      for (let x = tileBInfo.x0; x < tileBInfo.x1; x++) {
        if (hybrid.cellMature(x, y)) {
          const queryInB = hybrid.hybridRatioAt(x, y);
          expect(queryInB.provenance).toBe('corrected');
          foundMature = true;
          break;
        }
      }
    }
    expect(foundMature).toBe(true);

    // Mid-flight swell change must invalidate all tiles.
    const job2 = hybrid.startCorrection();
    hybrid.setSwell({ heightM: 2.4, periodS: 9, directionDeg: 0 });
    const res2 = hybrid.finishCorrection(job2);
    expect(res2.dropped.length).toBe(hybrid.tileCount);
    expect(res2.applied.length).toBe(0);
  });

  for (const kind of ['sandbar', 'beach'] as const) {
    it(`tracks teacher onset accuracy on ${kind}`, { timeout: 180000 }, () => {
      const bathy = generateBathymetry(kind, SEED);
      const hybrid = createHybrid(bathy, SWELL, 0);

      while (hybrid.teacher.time < 120 - 1e-6) {
        hybrid.pumpAccurate(10);
        hybrid.correctTiles();
      }

      const refOnset = referenceLabels(kind);
      const hybridDeltas: number[] = [];
      const fastDeltas: number[] = [];

      for (let x = 0; x < bathy.width; x++) {
        const ref = refOnset[x];
        if (ref < 0) continue;
        const hRes = hybrid.hybridOnsetRow(x);
        if (hRes.row >= 0) {
          hybridDeltas.push(Math.abs(hRes.row - ref));
        }
        const fRow = hybrid.fastOnsetRow(x);
        if (fRow >= 0) {
          fastDeltas.push(Math.abs(fRow - ref));
        }
      }

      const hybridMedian = median(hybridDeltas);
      const fastMedian = median(fastDeltas);

      console.log(
        `[S15 accuracy ${kind}] hybridMedian=${hybridMedian.toFixed(2)} fastMedian=${fastMedian.toFixed(2)} hybridCols=${hybridDeltas.length} fastCols=${fastDeltas.length}`
      );

      expect(hybridDeltas.length).toBeGreaterThan(0);
      expect(hybridMedian).toBeLessThanOrEqual(1);
      expect(hybridMedian).toBeLessThanOrEqual(fastMedian);
    });
  }

  it('is strictly deterministic across identical operation sequences', { timeout: 180000 }, () => {
    const runOnce = () => {
      const bathy = generateBathymetry('sandbar', SEED);
      const hybrid = createHybrid(bathy, SWELL, 0);

      hybrid.pumpAccurate(30);
      hybrid.correctTiles();
      hybrid.advanceWorld(1);
      hybrid.pumpAccurate(30);
      hybrid.bumpVersion([3]);
      const job = hybrid.startCorrection();
      hybrid.finishCorrection(job);

      for (let i = 0; i < 8; i++) {
        hybrid.hybridRatioAt((i * 15 + 3) % bathy.width, (i * 11 + 5) % bathy.height);
      }

      const onsetRows = new Int16Array(bathy.width);
      for (let x = 0; x < bathy.width; x++) {
        onsetRows[x] = hybrid.hybridOnsetRow(x).row;
      }

      const counterInts = new Int32Array([
        hybrid.appliedCorrections,
        hybrid.droppedStaleCorrections,
        hybrid.skippedIneligibleTiles,
      ]);

      const hash = fnv1a32([
        new Uint8Array(hybrid.corrRatio.buffer, hybrid.corrRatio.byteOffset, hybrid.corrRatio.byteLength),
        new Uint8Array(onsetRows.buffer, onsetRows.byteOffset, onsetRows.byteLength),
        new Uint8Array(counterInts.buffer, counterInts.byteOffset, counterInts.byteLength),
      ]);

      return { hash, onsetRows };
    };

    const a = runOnce();
    const b = runOnce();

    expect(a.hash).toBe(b.hash);
    expect(Array.from(a.onsetRows)).toEqual(Array.from(b.onsetRows));
  });
});
