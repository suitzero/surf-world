import { mkdirSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import type { BathymetryKind, SwellParams } from '../src/contracts/types.ts';
import { readEpisodeFile } from '../src/data/episode.ts';
import { generateBathymetry } from '../src/sim/bathymetry.ts';
import { createTeacher } from '../src/sim/teacher.ts';

declare const process: { cwd(): string };

interface EpisodeSpec {
  file: string;
  kind: BathymetryKind;
  seed: number;
}

interface SanityCheckResult {
  mse: number;
  maxAbsDiff: number;
}

interface LevelResult {
  cellM: number;
  gridW: number;
  gridH: number;
  substepsPerStep: number;
  costRunsS: number[];
  costMedianS: number;
  fieldMSE: number;
}

interface EpisodeResult {
  file: string;
  kind: BathymetryKind;
  seed: number;
  swell: SwellParams;
  sanityCheck?: SanityCheckResult;
  levels: LevelResult[];
}

interface AggregateLevel {
  meanCostMedianS: number;
  meanFieldMSE: number;
}

const WORLD_W_M = 512;
const WORLD_H_M = 384;
const FINE_W = 128;
const FINE_H = 96;
const FINE_CELLS = FINE_W * FINE_H;
const NUM_FRAMES = 121;
const TOTAL_SAMPLES = NUM_FRAMES * FINE_CELLS;

const SWELL: SwellParams = {
  heightM: 2.1,
  periodS: 10,
  directionDeg: 6,
};
const AMPLITUDE_M = SWELL.heightM / 2; // 1.05 m

const CELL_LEVELS = [4, 8, 16] as const;
const TIMED_REPEATS = 3;
const COARSE_STEPS = 120;
const COARSE_DT_S = 0.5;
const SANITY_STEPS = 1200;
const SANITY_DT_S = 0.05;
const SANITY_SAMPLE_EVERY = 10;

const EPISODES: readonly EpisodeSpec[] = [
  { file: 'ep00-beach-seed200000.bin', kind: 'beach', seed: 200000 },
  { file: 'ep02-sandbar-seed200010.bin', kind: 'sandbar', seed: 200010 },
  { file: 'ep04-point-seed200020.bin', kind: 'point', seed: 200020 },
  { file: 'ep06-reef-seed200030.bin', kind: 'reef', seed: 200030 },
];

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

function runSanityCheck(truthEta: Float32Array): SanityCheckResult {
  const bathy = generateBathymetry('beach', 200000);
  const teacher = createTeacher(bathy, {
    amplitudeM: AMPLITUDE_M,
    periodS: SWELL.periodS,
    directionDeg: SWELL.directionDeg,
  });

  let sumSq = 0;
  let maxAbsDiff = 0;

  const accumulateFrame = (frameIdx: number): void => {
    const offset = frameIdx * FINE_CELLS;
    const eta = teacher.eta;
    for (let i = 0; i < FINE_CELLS; i++) {
      const diff = eta[i] - truthEta[offset + i];
      const absDiff = Math.abs(diff);
      if (absDiff > maxAbsDiff) maxAbsDiff = absDiff;
      sumSq += diff * diff;
    }
  };

  // Frame 0 before stepping.
  accumulateFrame(0);

  // 1200 steps of 0.05 s, capturing after every 10th step (frames 1..120).
  for (let step = 1; step <= SANITY_STEPS; step++) {
    teacher.step(SANITY_DT_S);
    if (step % SANITY_SAMPLE_EVERY === 0) {
      accumulateFrame(step / SANITY_SAMPLE_EVERY);
    }
  }

  return {
    mse: sumSq / TOTAL_SAMPLES,
    maxAbsDiff,
  };
}

function computeUpsampledFieldMSE(
  coarseFrames: Float32Array[],
  gridW: number,
  gridH: number,
  cellM: number,
  truthEta: Float32Array
): number {
  const scale = 4 / cellM;

  // Precompute cell-centre-aligned bilinear indices and weights along x and y.
  const x0Arr = new Int32Array(FINE_W);
  const x1Arr = new Int32Array(FINE_W);
  const txArr = new Float64Array(FINE_W);
  for (let x = 0; x < FINE_W; x++) {
    const gx = clamp((x + 0.5) * scale - 0.5, 0, gridW - 1);
    const x0 = Math.floor(gx);
    const x1 = Math.ceil(gx);
    x0Arr[x] = x0;
    x1Arr[x] = x1;
    txArr[x] = gx - x0;
  }

  const y0Arr = new Int32Array(FINE_H);
  const y1Arr = new Int32Array(FINE_H);
  const tyArr = new Float64Array(FINE_H);
  for (let y = 0; y < FINE_H; y++) {
    const gy = clamp((y + 0.5) * scale - 0.5, 0, gridH - 1);
    const y0 = Math.floor(gy);
    const y1 = Math.ceil(gy);
    y0Arr[y] = y0;
    y1Arr[y] = y1;
    tyArr[y] = gy - y0;
  }

  let totalSq = 0;

  for (let f = 0; f < NUM_FRAMES; f++) {
    const coarse = coarseFrames[f];
    const truthFrameOffset = f * FINE_CELLS;

    for (let y = 0; y < FINE_H; y++) {
      const row0 = y0Arr[y] * gridW;
      const row1 = y1Arr[y] * gridW;
      const ty = tyArr[y];
      const truthRowOffset = truthFrameOffset + y * FINE_W;

      for (let x = 0; x < FINE_W; x++) {
        const x0 = x0Arr[x];
        const x1 = x1Arr[x];
        const tx = txArr[x];

        const e00 = coarse[row0 + x0];
        const e10 = coarse[row0 + x1];
        const e01 = coarse[row1 + x0];
        const e11 = coarse[row1 + x1];

        const top = e00 + (e10 - e00) * tx;
        const bot = e01 + (e11 - e01) * tx;
        const pred = top + (bot - top) * ty;

        const diff = pred - truthEta[truthRowOffset + x];
        totalSq += diff * diff;
      }
    }
  }

  return totalSq / TOTAL_SAMPLES;
}

function measureLevel(
  kind: BathymetryKind,
  seed: number,
  cellM: number,
  truthEta: Float32Array
): LevelResult {
  const gridW = WORLD_W_M / cellM;
  const gridH = WORLD_H_M / cellM;
  const bathy = generateBathymetry(kind, seed, gridW, gridH);
  const teacherOpts = {
    amplitudeM: AMPLITUDE_M,
    periodS: SWELL.periodS,
    directionDeg: SWELL.directionDeg,
    cellM,
  };

  // Timed runs x3.
  const costRunsS: number[] = [];
  let substepsPerStep = 0;

  for (let r = 0; r < TIMED_REPEATS; r++) {
    const t0 = performance.now();
    const teacher = createTeacher(bathy, teacherOpts);
    for (let s = 0; s < COARSE_STEPS; s++) {
      teacher.step(COARSE_DT_S);
    }
    const elapsedS = (performance.now() - t0) / 1000;
    costRunsS.push(elapsedS);
    substepsPerStep = teacher.lastSubsteps;
  }

  const costMedianS = median(costRunsS);

  // Untimed error run: capture frame 0 and 120 stepped frames.
  const errTeacher = createTeacher(bathy, teacherOpts);
  const coarseFrames: Float32Array[] = new Array<Float32Array>(NUM_FRAMES);
  coarseFrames[0] = errTeacher.eta.slice();
  for (let s = 1; s <= COARSE_STEPS; s++) {
    errTeacher.step(COARSE_DT_S);
    coarseFrames[s] = errTeacher.eta.slice();
  }

  const fieldMSE = computeUpsampledFieldMSE(coarseFrames, gridW, gridH, cellM, truthEta);

  return {
    cellM,
    gridW,
    gridH,
    substepsPerStep,
    costRunsS,
    costMedianS,
    fieldMSE,
  };
}

const repoRoot = process.cwd();
const heldoutDir = `${repoRoot}/data/heldout`;
const episodeResults: EpisodeResult[] = [];

for (const ep of EPISODES) {
  const truth = readEpisodeFile(`${heldoutDir}/${ep.file}`);
  if (
    truth.header.grid.width !== FINE_W ||
    truth.header.grid.height !== FINE_H ||
    truth.header.episode.numFrames !== NUM_FRAMES
  ) {
    throw new Error(
      `Unexpected truth shape in ${ep.file}: ${truth.header.episode.numFrames}x${truth.header.grid.height}x${truth.header.grid.width}`
    );
  }

  const result: EpisodeResult = {
    file: ep.file,
    kind: ep.kind,
    seed: ep.seed,
    swell: { ...SWELL },
    levels: [],
  };

  if (ep.file === 'ep00-beach-seed200000.bin') {
    result.sanityCheck = runSanityCheck(truth.eta);
  }

  for (const cellM of CELL_LEVELS) {
    result.levels.push(measureLevel(ep.kind, ep.seed, cellM, truth.eta));
  }

  episodeResults.push(result);
}

const aggregate: Record<string, AggregateLevel> = {};
for (const cellM of CELL_LEVELS) {
  let sumCost = 0;
  let sumMSE = 0;
  for (const ep of episodeResults) {
    const level = ep.levels.find((l) => l.cellM === cellM);
    if (!level) throw new Error(`Missing cellM=${cellM} for ${ep.file}`);
    sumCost += level.costMedianS;
    sumMSE += level.fieldMSE;
  }
  aggregate[String(cellM)] = {
    meanCostMedianS: sumCost / episodeResults.length,
    meanFieldMSE: sumMSE / episodeResults.length,
  };
}

const cpus = os.cpus();
const output = {
  machine: {
    cpu: cpus.length > 0 ? cpus[0].model : 'unknown',
    platform: os.platform(),
    arch: os.arch(),
  },
  episodes: episodeResults,
  aggregate,
};

const outDir = `${repoRoot}/tools/train/results`;
mkdirSync(outDir, { recursive: true });
const outPath = `${outDir}/s11_anytime.json`;
writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`, 'utf-8');

// Print human-readable summary table.
console.log(`Machine: ${output.machine.cpu} (${output.machine.platform}/${output.machine.arch})`);
const ep00Sanity = episodeResults[0]?.sanityCheck;
if (ep00Sanity) {
  console.log(
    `Sanity check (ep00, cellM=4, dt=0.05s): MSE=${ep00Sanity.mse.toExponential(4)}, ` +
      `maxAbsDiff=${ep00Sanity.maxAbsDiff.toExponential(4)}`
  );
}
console.log('');
console.log(
  `${'Episode'.padEnd(28)} ${'cellM'.padStart(5)} ${'Grid'.padStart(8)} ` +
    `${'Substeps'.padStart(8)} ${'Median(s)'.padStart(10)} ${'FieldMSE'.padStart(12)}`
);
console.log('-'.repeat(77));
for (const ep of episodeResults) {
  for (const lvl of ep.levels) {
    const grid = `${lvl.gridW}x${lvl.gridH}`;
    console.log(
      `${ep.file.padEnd(28)} ${String(lvl.cellM).padStart(5)} ${grid.padStart(8)} ` +
        `${String(lvl.substepsPerStep).padStart(8)} ${lvl.costMedianS.toFixed(4).padStart(10)} ` +
        `${lvl.fieldMSE.toExponential(4).padStart(12)}`
    );
  }
}
console.log('');
console.log('Aggregate across 4 heldout episodes:');
console.log(`${'cellM'.padStart(5)} ${'MeanCostMedian(s)'.padStart(18)} ${'MeanFieldMSE'.padStart(14)}`);
console.log('-'.repeat(40));
for (const cellM of CELL_LEVELS) {
  const agg = aggregate[String(cellM)];
  console.log(
    `${String(cellM).padStart(5)} ${agg.meanCostMedianS.toFixed(4).padStart(18)} ` +
      `${agg.meanFieldMSE.toExponential(4).padStart(14)}`
  );
}
console.log('');
console.log(`Wrote ${outPath}`);
