import type { BathymetryKind, SwellParams } from '../contracts/types.ts';
import { generateBathymetry } from '../sim/bathymetry.ts';
import { createTeacher } from '../sim/teacher.ts';
import type { TeacherSolver } from '../sim/teacher.ts';
import { computeOnsetRows, foldTimeMax } from './labRender.ts';

export interface TeacherConfigureMsg {
  type: 'configure';
  kind: BathymetryKind;
  seed: number;
  swell: SwellParams;
}

export interface TeacherSnapshotMsg {
  type: 'snapshot';
  time: number;
  eta: ArrayBuffer;
  onsetRow: Int16Array;
  stepMsAvg: number;
}

interface WorkerScope {
  onmessage: ((e: MessageEvent) => void) | null;
  postMessage: (msg: unknown, transfer?: Transferable[]) => void;
  setInterval: (fn: () => void, ms: number) => number;
  clearInterval: (id: number) => void;
}

const scope = globalThis as unknown as WorkerScope;

let teacher: TeacherSolver | null = null;
let timeMax: Float32Array = new Float32Array(0);
let depth: Float32Array = new Float32Array(0);
let w = 0;
let h = 0;
let timerId: number | null = null;
let stepMsEma = 0;
let hasStepSample = false;

function postSnapshot(): void {
  if (!teacher) return;
  const onset = computeOnsetRows(timeMax, depth, w, h);
  const etaCopy = Float32Array.from(teacher.eta);
  const msg: TeacherSnapshotMsg = {
    type: 'snapshot',
    time: teacher.time,
    eta: etaCopy.buffer,
    onsetRow: onset,
    stepMsAvg: stepMsEma,
  };
  scope.postMessage(msg, [etaCopy.buffer]);
}

scope.onmessage = (e: MessageEvent): void => {
  const msg = e.data as TeacherConfigureMsg | undefined;
  if (!msg || msg.type !== 'configure') return;

  if (timerId !== null) {
    scope.clearInterval(timerId);
    timerId = null;
  }

  const bathy = generateBathymetry(msg.kind, msg.seed);
  w = bathy.width;
  h = bathy.height;
  depth = bathy.depth;
  teacher = createTeacher(bathy, {
    amplitudeM: msg.swell.heightM / 2,
    periodS: msg.swell.periodS,
    directionDeg: msg.swell.directionDeg,
  });
  timeMax = new Float32Array(w * h);
  stepMsEma = 0;
  hasStepSample = false;

  postSnapshot();

  timerId = scope.setInterval(() => {
    if (!teacher) return;
    const t0 = performance.now();
    for (let i = 0; i < 5; i++) {
      teacher.step(0.05);
      foldTimeMax(timeMax, teacher.eta);
    }
    const perStepMs = (performance.now() - t0) / 5;
    if (!hasStepSample) {
      stepMsEma = perStepMs;
      hasStepSample = true;
    } else {
      stepMsEma = stepMsEma * 0.8 + perStepMs * 0.2;
    }
    postSnapshot();
  }, 250);
};
