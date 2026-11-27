import { normalizeSeed } from '../sim/prng.ts';
import { createTeacher } from './teacher.ts';
import type { SlackAction, SlackState, SlackTeacher, SlackTeacherOptions } from './types.ts';

/**
 * World wrapper and deterministic FNV-1a state hashing for the 1D slackline domain.
 */

const FNV_OFFSET_BASIS = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

function mix(h: number, v: number): number {
  return Math.imul(h ^ (v >>> 0), FNV_PRIME) >>> 0;
}

function mixFloat(h: number, v: number): number {
  const q = Math.round(v * 4096);
  h = mix(h, q | 0);
  return mix(h, (q / 4294967296) | 0);
}

export interface SlackWorld extends SlackState {
  readonly seed: number;
  readonly options: Readonly<SlackTeacherOptions>;
  readonly teacher: SlackTeacher;
  stepCount: number;
}

/** Create a deterministic slackline world wrapping the nonlinear teacher. */
export function createSlackWorld(seed: number, opts: SlackTeacherOptions = {}): SlackWorld {
  const normSeed = normalizeSeed(seed);
  const optionsCopy: SlackTeacherOptions = { ...opts };
  const teacher = createTeacher(normSeed, optionsCopy);

  const world: SlackWorld = {
    seed: normSeed,
    options: optionsCopy,
    teacher,
    stepCount: 0,
    get s(): number {
      return teacher.s;
    },
    get theta(): number {
      return teacher.theta;
    },
    get omega(): number {
      return teacher.omega;
    },
    get y(): number {
      return teacher.y;
    },
    get yVel(): number {
      return teacher.yVel;
    },
    get time(): number {
      return teacher.time;
    },
    get fallen(): boolean {
      return teacher.fallen;
    },
    get finished(): boolean {
      return teacher.finished;
    },
  };

  return world;
}

/** Advance the slackline world by one control step (0.05 s) using `action`. */
export function stepSlackWorld(world: SlackWorld, action: SlackAction): void {
  world.teacher.step(action);
  world.stepCount++;
}

/**
 * Deterministic 32-bit FNV-1a hash over stepCount, seed/PRNG wind state,
 * and all continuous & terminal slackline state variables.
 */
export function hashSlackState(world: SlackWorld): number {
  let h = FNV_OFFSET_BASIS;
  h = mix(h, world.stepCount);
  h = mix(h, world.seed);
  h = mix(h, world.teacher.rngState);
  h = mixFloat(h, world.teacher.phase1);
  h = mixFloat(h, world.teacher.phase2);
  h = mixFloat(h, world.s);
  h = mixFloat(h, world.theta);
  h = mixFloat(h, world.omega);
  h = mixFloat(h, world.y);
  h = mixFloat(h, world.yVel);
  h = mixFloat(h, world.time);
  h = mix(h, world.fallen ? 1 : 0);
  h = mix(h, world.finished ? 1 : 0);
  return h >>> 0;
}
