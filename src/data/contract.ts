/**
 * S07e state/dataset contract for the M2 AI Ocean pipeline.
 *
 * The teacher (linear shallow-water solver, src/sim/teacher.ts) is its OWN
 * ground truth: all labels are derived strictly from the teacher's own stored
 * fields, never from the M1 eikonal ocean.
 *
 * The 5 caveats from files/S07D-VERDICT.md travel in every episode header.
 */

import { BATHYMETRY_KINDS } from '../contracts/types.ts';
import type { BathymetryKind, SwellParams } from '../contracts/types.ts';
import { BREAKING_RATIO } from '../sim/ocean.ts';

export const EPISODE_FORMAT = 'surf-world-episode';
export const EPISODE_FORMAT_VERSION = 1;

export const SOLVER_DT_S = 0.05;
export const SAMPLE_INTERVAL_S = 0.5;
export const SAMPLE_EVERY_STEPS = 10; // SAMPLE_INTERVAL_S / SOLVER_DT_S

export const TEACHER_ID = 'linear-shallow-water-teacher-v1';
export const TEACHER_MODEL =
  '2D linear shallow-water equations (depth-averaged, non-dispersive), Arakawa C-grid';

export const TEACHER_CAVEATS: readonly string[] = [
  'Timing bias: teacher crest times lead dispersive physics (and the M1 eikonal groupTau) by ~3.4-5.8 s (unwrapped medians, per bathymetry kind). Labels are self-consistent within the teacher; never mix them with measured or dispersive-model times without correcting this offset.',
  'Offshore boundary reflection R = 0.23744 (S07b; the 0.20 bound was not met and two fix attempts were reverted over a reef-stability trade-off). Steady-state eta fields contain a standing-wave component.',
  'No breaking dissipation: the teacher does not dissipate energy after onset, so amplitudes shoreward of the onset line are overestimates. Only the onset line and the offshore-side fields are label-grade.',
  'Reef amplification: on reef bathymetry max|eta|/amplitude = 3.33336 (exactly linear, not a numerical artifact - S07b). Reef amplitude labels carry this amplification.',
  "Onset-line tail: sandbar p90 |delta row| vs M1 = 18 cells (median 1). Per-column label confidence is stored in onsetConfidence; do not treat every column's label as equally reliable.",
];

export interface EpisodeSpec {
  index: number;
  kind: BathymetryKind;
  seed: number;
  swell: SwellParams;
  durationS: number;
}

export const PILOT_EPISODES: readonly EpisodeSpec[] = BATHYMETRY_KINDS.flatMap((kind, kIdx) => [
  {
    index: kIdx * 2,
    kind,
    seed: 7,
    swell: { heightM: 1.8, periodS: 9, directionDeg: 0 },
    durationS: 60,
  },
  {
    index: kIdx * 2 + 1,
    kind,
    seed: 123,
    swell: { heightM: 1.2, periodS: 11, directionDeg: -12 },
    durationS: 60,
  },
]);

export function episodeFileName(spec: EpisodeSpec): string {
  return `ep${String(spec.index).padStart(2, '0')}-${spec.kind}-seed${spec.seed}.bin`;
}

export type FieldDType = 'f32' | 'u8' | 'i16';

export const DTYPE_SIZES: Record<FieldDType, number> = {
  f32: 4,
  u8: 1,
  i16: 2,
};

export interface FieldSpec {
  name: string;
  dtype: FieldDType;
  shape: number[];
  offset: number;
  byteLength: number;
}

export interface EpisodeHeader {
  format: string;
  formatVersion: number;
  contractVersion: number;
  episode: {
    index: number;
    kind: BathymetryKind;
    seed: number;
    swell: SwellParams;
    durationS: number;
    sampleIntervalS: number;
    numFrames: number;
  };
  grid: {
    width: number;
    height: number;
    cellM: number;
  };
  teacher: {
    id: string;
    model: string;
    dispersive: boolean;
  };
  solver: {
    dtS: number;
    cflSafety: number;
    spongeCells: number;
    amplitudeM: number;
  };
  caveats: string[];
  contentHash: string;
  fields: FieldSpec[];
}

export interface EpisodeData {
  header: EpisodeHeader;
  depth: Float32Array;
  eta: Float32Array;
  u: Float32Array;
  v: Float32Array;
  timeMaxAbsEta: Float32Array;
  onsetMask: Uint8Array;
  onsetRow: Int16Array;
  onsetConfidence: Float32Array;
  crestArrivalS: Float32Array;
}

export const FIELD_ORDER: readonly string[] = [
  'depth',
  'eta',
  'u',
  'v',
  'timeMaxAbsEta',
  'onsetMask',
  'onsetRow',
  'onsetConfidence',
  'crestArrivalS',
];

export function fieldShape(
  name: string,
  width: number,
  height: number,
  numFrames: number
): number[] {
  switch (name) {
    case 'depth':
    case 'timeMaxAbsEta':
    case 'onsetMask':
    case 'crestArrivalS':
      return [height, width];
    case 'eta':
    case 'u':
    case 'v':
      return [numFrames, height, width];
    case 'onsetRow':
    case 'onsetConfidence':
      return [width];
    default:
      throw new Error(`Unknown episode field: ${name}`);
  }
}

export interface EpisodeLabels {
  timeMaxAbsEta: Float32Array;
  onsetMask: Uint8Array;
  onsetRow: Int16Array;
  onsetConfidence: Float32Array;
  crestArrivalS: Float32Array;
}

export function computeLabels(
  eta: Float32Array,
  depth: Float32Array,
  width: number,
  height: number,
  numFrames: number
): EpisodeLabels {
  const n = width * height;
  const timeMaxAbsEta = new Float32Array(n);

  for (let k = 0; k < numFrames; k++) {
    const base = k * n;
    for (let i = 0; i < n; i++) {
      const v = Math.abs(eta[base + i]);
      if (v > timeMaxAbsEta[i]) timeMaxAbsEta[i] = v;
    }
  }

  const ratio = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    ratio[i] = depth[i] > 0 ? (2 * timeMaxAbsEta[i]) / depth[i] : 0;
  }

  const onsetRow = new Int16Array(width);
  const onsetMask = new Uint8Array(n);
  const onsetConfidence = new Float32Array(width);

  for (let x = 0; x < width; x++) {
    let foundY = -1;
    for (let y = 0; y < height; y++) {
      const idx = y * width + x;
      if (depth[idx] > 0 && ratio[idx] >= BREAKING_RATIO) {
        foundY = y;
        break;
      }
    }
    onsetRow[x] = foundY;
    if (foundY < 0) {
      onsetConfidence[x] = 0;
    } else {
      onsetMask[foundY * width + x] = 1;
      const rOn = ratio[foundY * width + x];
      const rPrev = foundY > 0 ? ratio[(foundY - 1) * width + x] : 0;
      const diff = rOn - rPrev;
      onsetConfidence[x] = diff < 0 ? 0 : diff > 1 ? 1 : diff;
    }
  }

  const crestArrivalS = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const peak = timeMaxAbsEta[i];
    if (peak <= 0) {
      crestArrivalS[i] = -1;
      continue;
    }
    const thresh = 0.5 * peak;
    let arrival = -1;
    for (let k = 0; k < numFrames; k++) {
      if (Math.abs(eta[k * n + i]) >= thresh) {
        arrival = k * SAMPLE_INTERVAL_S;
        break;
      }
    }
    crestArrivalS[i] = arrival;
  }

  return {
    timeMaxAbsEta,
    onsetMask,
    onsetRow,
    onsetConfidence,
    crestArrivalS,
  };
}
