import { describe, expect, it } from 'vitest';
import type { SwellParams } from '../src/contracts/types.ts';
import { generateBathymetry } from '../src/sim/bathymetry.ts';
import { buildOcean, dispersionK, groupSpeed, updateOcean } from '../src/sim/ocean.ts';

const SWELL: SwellParams = { heightM: 1.8, periodS: 9, directionDeg: 0 };

describe('Set envelope group-velocity propagation (P5)', () => {
  it('group speed: near half the phase speed in deep water, near-equal in shallow water', () => {
    const omega = (2 * Math.PI) / 9;

    const kDeep = dispersionK(omega, 40);
    const cDeep = omega / kDeep;
    const ratioDeep = groupSpeed(omega, 40) / cDeep;
    expect(ratioDeep).toBeLessThan(0.6);
    expect(ratioDeep).toBeGreaterThan(0.4);

    const kShallow = dispersionK(omega, 1.2);
    const cShallow = omega / kShallow;
    const ratioShallow = groupSpeed(omega, 1.2) / cShallow;
    expect(ratioShallow).toBeGreaterThan(0.9);
  });

  it('envelope arrives at group speed offshore, not phase speed', () => {
    const bathy = generateBathymetry('beach', 5);
    const field = buildOcean(bathy, SWELL, 0);
    const w = bathy.width;
    const h = bathy.height;

    // Pick a deep cell in column x = 64 with depthEff closest to 11 m (> 9 m).
    const x = 64;
    let di = -1;
    let bestDiff = Infinity;
    for (let y = 0; y < h; y++) {
      const idx = y * w + x;
      const d = field.depthEff[idx];
      if (d > 9) {
        const diff = Math.abs(d - 11);
        if (diff < bestDiff) {
          bestDiff = diff;
          di = idx;
        }
      }
    }
    expect(di).toBeGreaterThanOrEqual(0);
    expect(field.depthEff[di]).toBeGreaterThan(9);

    const tauVal = field.tau[di];
    const groupTauVal = field.groupTau[di];
    expect(groupTauVal).toBeGreaterThan(tauVal + 1.5);

    const dt = 0.05;
    const steps = 2800; // 140 s
    const times = new Float64Array(steps);
    const envSeries = new Float64Array(steps);
    for (let step = 1; step <= steps; step++) {
      const t = step * dt;
      updateOcean(field, SWELL, t, w, h, dt);
      times[step - 1] = t;
      envSeries[step - 1] = (field.ratio[di] * field.depthEff[di]) / field.amp[di];
    }

    const groupT = 7 * SWELL.periodS; // 63 s
    const expectedGroup = groupTauVal % groupT;
    const expectedPhase = tauVal % groupT;

    const findPeakInWindow = (center: number, halfWidth: number): { time: number; val: number } => {
      let bestVal = -Infinity;
      let bestTime = center;
      for (let i = 0; i < steps; i++) {
        const t = times[i];
        if (t >= center - halfWidth && t <= center + halfWidth) {
          if (envSeries[i] > bestVal) {
            bestVal = envSeries[i];
            bestTime = t;
          }
        }
      }
      return { time: bestTime, val: bestVal };
    };

    const peakNearGroup = findPeakInWindow(expectedGroup, 5);
    // At x = 64, depth ~11 m is ~60 m from the offshore boundary, so
    // groupTau - tau is ~2.0 s (< 5 s). Use a non-overlapping ±0.5 s window
    // around expectedPhase so it does not capture the group-velocity peak;
    // with groupT = 63 s, a 1.5 s offset gives ~0.0045 cosine drop.
    const peakNearPhase = findPeakInWindow(expectedPhase, 0.5);

    expect(peakNearGroup.val).toBeGreaterThan(peakNearPhase.val + 0.004);
    expect(Math.abs(peakNearGroup.time - expectedGroup)).toBeLessThanOrEqual(0.5);
    expect(Math.abs(peakNearGroup.time - expectedPhase)).toBeGreaterThanOrEqual(1.5);
  });
});
