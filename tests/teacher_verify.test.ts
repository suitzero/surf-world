import { describe, expect, it } from 'vitest';
import type { Bathymetry } from '../src/contracts/types.ts';
import { generateBathymetry } from '../src/sim/bathymetry.ts';
import { createTeacher } from '../src/sim/teacher.ts';
import type { TeacherSolver } from '../src/sim/teacher.ts';

const DT = 0.05;
const GRAVITY = 9.81;

function makeFlatTankRes(depthM: number, cellM: number): Bathymetry {
  const width = Math.round(512 / cellM);
  const height = Math.round(384 / cellM);
  return {
    kind: 'beach',
    seed: 0,
    width,
    height,
    depth: new Float32Array(width * height).fill(depthM),
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

function findFirstCrestAfter(
  times: Float64Array,
  series: Float64Array,
  minTime: number
): number {
  for (let i = 1; i < series.length - 1; i++) {
    if (times[i] <= minTime) continue;
    if (series[i] > 0.3 && series[i] >= series[i - 1] && series[i] > series[i + 1]) {
      return times[i];
    }
  }
  return -1;
}

describe('S07b linear shallow-water teacher verification', () => {
  it(
    'grid convergence: flat 16 m tank at cell sizes 8/4/2 m',
    { timeout: 60000 },
    () => {
      const depthM = 16;
      const amplitudeM = 0.5;
      const theory = Math.sqrt(GRAVITY * depthM);
      const steps = Math.round(40 / DT);
      const resolutions = [8, 4, 2] as const;

      const errors = new Map<number, number>();

      for (const cellM of resolutions) {
        const bathy = makeFlatTankRes(depthM, cellM);
        const teacher = createTeacher(bathy, {
          amplitudeM,
          periodS: 9,
          directionDeg: 0,
          cellM,
        });

        const times = new Float64Array(steps);
        const probe1 = new Float64Array(steps);
        const probe2 = new Float64Array(steps);
        let maxEta = 0;
        let allFinite = true;

        for (let s = 0; s < steps; s++) {
          teacher.step(DT);
          times[s] = teacher.time;
          probe1[s] = teacher.sampleEta(256, 96);
          probe2[s] = teacher.sampleEta(256, 256);
          const scan = maxAbs(teacher.eta);
          if (!scan.allFinite) allFinite = false;
          if (scan.max > maxEta) maxEta = scan.max;
        }

        const t1 = findFirstCrestAfter(times, probe1, 5);
        const t2 = findFirstCrestAfter(times, probe2, 5);
        const speed = 160 / (t2 - t1);
        const err = Math.abs(speed - theory) / theory;
        const errPct = err * 100;

        console.log(
          `[grid convergence] cellM=${cellM}m grid=${bathy.width}x${bathy.height} ` +
            `t1=${t1.toFixed(3)}s t2=${t2.toFixed(3)}s speed=${speed.toFixed(4)}m/s ` +
            `theory=${theory.toFixed(4)}m/s errPct=${errPct.toFixed(3)}% maxEta=${maxEta.toFixed(4)}m`
        );

        expect(Number.isFinite(speed)).toBe(true);
        expect(t1).toBeGreaterThan(5);
        expect(t2).toBeGreaterThan(t1);
        expect(err).toBeLessThan(0.05);
        expect(allFinite).toBe(true);
        expect(Number.isFinite(maxEta)).toBe(true);
        expect(maxEta).toBeLessThanOrEqual(3 * amplitudeM);

        errors.set(cellM, err);
      }

      const err2 = errors.get(2)!;
      const err8 = errors.get(8)!;
      expect(err2).toBeLessThanOrEqual(err8 + 0.002);
    }
  );

  it(
    'offshore reflection: standing-wave ratio in flat tank',
    { timeout: 60000 },
    () => {
      // Standing-wave ratio method: a pure travelling wave has constant envelope
      // amplitude along y; a reflected component modulates the envelope between
      // A(1 - R) and A(1 + R), yielding R = (Amax - Amin) / (Amax + Amin).
      const bathy = makeFlatTankRes(16, 4);
      const teacher = createTeacher(bathy, {
        amplitudeM: 0.5,
        periodS: 9,
        directionDeg: 0,
      });

      const probeYs: number[] = [];
      for (let y = 60; y <= 340; y += 4) {
        probeYs.push(y);
      }
      const mins = new Float64Array(probeYs.length).fill(Infinity);
      const maxs = new Float64Array(probeYs.length).fill(-Infinity);

      const steps = Math.round(90 / DT);
      for (let s = 0; s < steps; s++) {
        teacher.step(DT);
        const t = teacher.time;
        if (t >= 72 - 1e-9 && t <= 90 + 1e-9) {
          for (let j = 0; j < probeYs.length; j++) {
            const val = teacher.sampleEta(256, probeYs[j]);
            if (val < mins[j]) mins[j] = val;
            if (val > maxs[j]) maxs[j] = val;
          }
        }
      }

      let Amax = -Infinity;
      let Amin = Infinity;
      for (let j = 0; j < probeYs.length; j++) {
        const amp = (maxs[j] - mins[j]) / 2;
        if (amp > Amax) Amax = amp;
        if (amp < Amin) Amin = amp;
      }

      const R = (Amax - Amin) / (Amax + Amin);

      console.log(
        `[offshore reflection] Amax=${Amax.toFixed(5)}m Amin=${Amin.toFixed(5)}m R=${R.toFixed(5)}`
      );

      expect(Number.isFinite(R)).toBe(true);
      expect(R).toBeLessThan(0.2);
    }
  );

  it(
    'energy does not grow after forcing is switched off',
    { timeout: 60000 },
    () => {
      const bathy = makeFlatTankRes(16, 4);
      const teacherOn = createTeacher(bathy, {
        amplitudeM: 0.9,
        periodS: 9,
        directionDeg: 0,
      });

      const stepsOn = Math.round(30 / DT);
      for (let s = 0; s < stepsOn; s++) {
        teacherOn.step(DT);
      }

      // Energy proxy over wet cells (d > 0). Because u and v are staggered face
      // values stored at cell indices rather than co-located cell-centre values,
      // this sum is an energy proxy rather than an exact discrete energy integral.
      const energy = (solver: TeacherSolver): number => {
        const { depth } = bathy;
        const { eta, u, v } = solver;
        let sum = 0;
        for (let i = 0; i < depth.length; i++) {
          const d = depth[i];
          if (d > 0) {
            const e = eta[i];
            const ui = u[i];
            const vi = v[i];
            sum += 0.5 * (e * e * d + d * (ui * ui + vi * vi));
          }
        }
        return sum;
      };

      const E0 = energy(teacherOn);
      expect(Number.isFinite(E0)).toBe(true);

      const teacherOff = createTeacher(bathy, { amplitudeM: 0 });
      teacherOff.eta.set(teacherOn.eta);
      teacherOff.u.set(teacherOn.u);
      teacherOff.v.set(teacherOn.v);

      const stepsOff = Math.round(60 / DT);
      let maxE = E0;
      let finalE = E0;
      for (let s = 1; s <= stepsOff; s++) {
        teacherOff.step(DT);
        if (s % 20 === 0) {
          const eNow = energy(teacherOff);
          expect(Number.isFinite(eNow)).toBe(true);
          if (eNow > maxE) maxE = eNow;
          finalE = eNow;
        }
      }

      const ratio = maxE / E0;
      console.log(
        `[energy decay] E0=${E0.toFixed(4)} maxE=${maxE.toFixed(4)} ` +
          `finalE=${finalE.toFixed(4)} maxE/E0=${ratio.toFixed(6)}`
      );

      expect(Number.isFinite(maxE)).toBe(true);
      expect(Number.isFinite(finalE)).toBe(true);
      expect(maxE).toBeLessThanOrEqual(1.05 * E0);
    }
  );

  it(
    'reef amplification scales linearly with forcing amplitude',
    { timeout: 60000 },
    () => {
      const amplitudes = [0.45, 0.9, 1.8] as const;
      const steps = Math.round(120 / DT);
      const ratios = new Map<number, number>();

      for (const A of amplitudes) {
        const bathy = generateBathymetry('reef', 7);
        const teacher = createTeacher(bathy, { amplitudeM: A });

        let maxAbsEta = 0;
        let allFinite = true;
        for (let s = 0; s < steps; s++) {
          teacher.step(DT);
          const scan = maxAbs(teacher.eta);
          if (!scan.allFinite) allFinite = false;
          if (scan.max > maxAbsEta) maxAbsEta = scan.max;
        }

        const ratio = maxAbsEta / A;
        console.log(
          `[reef linearity] A=${A.toFixed(2)}m maxAbsEta=${maxAbsEta.toFixed(5)}m ` +
            `ratio=${ratio.toFixed(5)}`
        );

        expect(allFinite).toBe(true);
        expect(Number.isFinite(maxAbsEta)).toBe(true);
        expect(Number.isFinite(ratio)).toBe(true);
        ratios.set(A, ratio);
      }

      const rRef = ratios.get(0.9)!;
      const rLow = ratios.get(0.45)!;
      const rHigh = ratios.get(1.8)!;
      expect(Math.abs(rLow / rRef - 1)).toBeLessThanOrEqual(0.2);
      expect(Math.abs(rHigh / rRef - 1)).toBeLessThanOrEqual(0.2);
    }
  );
});
