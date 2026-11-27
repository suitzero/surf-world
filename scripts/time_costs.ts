import { mkdirSync, writeFileSync } from 'node:fs';
import { generateBathymetry } from '../src/sim/bathymetry.ts';
import { createTeacher } from '../src/sim/teacher.ts';
import { buildOcean } from '../src/sim/ocean.ts';

declare const process: { cwd(): string };

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

const condition = {
  kind: 'beach' as const,
  seed: 200000,
  swell: {
    heightM: 2.1,
    periodS: 10,
    directionDeg: 6,
  },
};

// Generate bathymetry once (untimed).
const bathy = generateBathymetry(condition.kind, condition.seed);

// 1. Teacher simulation of 60 s: createTeacher + 1200 steps of 0.05 s, repeated 3 times.
const TEACHER_STEPS = 1200;
const TEACHER_DT_S = 0.05;
const TEACHER_REPEATS = 3;
const teacherRunsS: number[] = [];
let teacherLastTime = 0;

for (let r = 0; r < TEACHER_REPEATS; r++) {
  const t0 = performance.now();
  const teacher = createTeacher(bathy, {
    amplitudeM: condition.swell.heightM / 2,
    periodS: condition.swell.periodS,
    directionDeg: condition.swell.directionDeg,
  });
  for (let step = 0; step < TEACHER_STEPS; step++) {
    teacher.step(TEACHER_DT_S);
  }
  const elapsedS = (performance.now() - t0) / 1000;
  teacherRunsS.push(elapsedS);
  teacherLastTime = teacher.time;
}

const teacherMedianS = median(teacherRunsS);

// 2. buildOcean(bathy, swell, 0), repeated 5 times.
const OCEAN_REPEATS = 5;
const oceanRunsS: number[] = [];
let oceanChecksum = 0;

for (let r = 0; r < OCEAN_REPEATS; r++) {
  const t0 = performance.now();
  const ocean = buildOcean(bathy, condition.swell, 0);
  const elapsedS = (performance.now() - t0) / 1000;
  oceanRunsS.push(elapsedS);
  oceanChecksum = ocean.tau[0];
}

const oceanMedianS = median(oceanRunsS);

const results = {
  condition,
  teacher_60s: {
    steps: TEACHER_STEPS,
    dt_s: TEACHER_DT_S,
    duration_s: TEACHER_STEPS * TEACHER_DT_S,
    sim_time_s: teacherLastTime,
    repeats: TEACHER_REPEATS,
    runs_s: teacherRunsS,
    median_s: teacherMedianS,
  },
  build_ocean: {
    tide_m: 0,
    repeats: OCEAN_REPEATS,
    runs_s: oceanRunsS,
    median_s: oceanMedianS,
    checksum: oceanChecksum,
  },
};

const outDir = `${process.cwd()}/tools/train/results`;
mkdirSync(outDir, { recursive: true });
const outPath = `${outDir}/s10_costs_ts.json`;
writeFileSync(outPath, `${JSON.stringify(results, null, 2)}\n`, 'utf-8');

console.log(JSON.stringify(results, null, 2));
