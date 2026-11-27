import { CELL_M, CONTRACT_VERSION } from '../contracts/types.ts';
import { generateBathymetry } from '../sim/bathymetry.ts';
import { createTeacher } from '../sim/teacher.ts';
import {
  EPISODE_FORMAT,
  EPISODE_FORMAT_VERSION,
  SAMPLE_EVERY_STEPS,
  SAMPLE_INTERVAL_S,
  SOLVER_DT_S,
  TEACHER_CAVEATS,
  TEACHER_ID,
  TEACHER_MODEL,
  computeLabels,
} from './contract.ts';
import type { EpisodeData, EpisodeHeader, EpisodeSpec } from './contract.ts';

const DEFAULT_CFL_SAFETY = 0.9;
const DEFAULT_SPONGE_CELLS = 8;

export function generateEpisode(spec: EpisodeSpec): EpisodeData {
  const bathy = generateBathymetry(spec.kind, spec.seed);
  const w = bathy.width;
  const h = bathy.height;
  const n = w * h;
  const amplitudeM = spec.swell.heightM / 2;

  const teacher = createTeacher(bathy, {
    amplitudeM,
    periodS: spec.swell.periodS,
    directionDeg: spec.swell.directionDeg,
  });

  const numFrames = Math.round(spec.durationS / SAMPLE_INTERVAL_S) + 1;
  const totalSteps = Math.round(spec.durationS / SOLVER_DT_S);

  const eta = new Float32Array(numFrames * n);
  const u = new Float32Array(numFrames * n);
  const v = new Float32Array(numFrames * n);

  // Capture frame 0 before any solver step.
  eta.set(teacher.eta, 0);
  u.set(teacher.u, 0);
  v.set(teacher.v, 0);

  for (let step = 1; step <= totalSteps; step++) {
    teacher.step(SOLVER_DT_S);
    if (step % SAMPLE_EVERY_STEPS === 0) {
      const frame = step / SAMPLE_EVERY_STEPS;
      const offset = frame * n;
      eta.set(teacher.eta, offset);
      u.set(teacher.u, offset);
      v.set(teacher.v, offset);
    }
  }

  const depth = bathy.depth.slice();
  const labels = computeLabels(eta, depth, w, h, numFrames);

  const header: EpisodeHeader = {
    format: EPISODE_FORMAT,
    formatVersion: EPISODE_FORMAT_VERSION,
    contractVersion: CONTRACT_VERSION,
    episode: {
      index: spec.index,
      kind: spec.kind,
      seed: spec.seed,
      swell: { ...spec.swell },
      durationS: spec.durationS,
      sampleIntervalS: SAMPLE_INTERVAL_S,
      numFrames,
    },
    grid: {
      width: w,
      height: h,
      cellM: CELL_M,
    },
    teacher: {
      id: TEACHER_ID,
      model: TEACHER_MODEL,
      dispersive: false,
    },
    solver: {
      dtS: SOLVER_DT_S,
      cflSafety: DEFAULT_CFL_SAFETY,
      spongeCells: DEFAULT_SPONGE_CELLS,
      amplitudeM,
    },
    caveats: [...TEACHER_CAVEATS],
    contentHash: '',
    fields: [],
  };

  return {
    header,
    depth,
    eta,
    u,
    v,
    timeMaxAbsEta: labels.timeMaxAbsEta,
    onsetMask: labels.onsetMask,
    onsetRow: labels.onsetRow,
    onsetConfidence: labels.onsetConfidence,
    crestArrivalS: labels.crestArrivalS,
  };
}
