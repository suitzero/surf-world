import type { SurferInput, WorldState } from '../contracts/types.ts';
import { createWorld, stepWorld } from './world.ts';
import type { CreateWorldOptions } from './world.ts';

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

/** Deterministic 32-bit FNV-1a hash over step/PRNG state, surfers, and foam. */
export function hashWorldState(world: WorldState): number {
  let h = FNV_OFFSET_BASIS;
  h = mix(h, world.stepCount);
  h = mix(h, world.rngState);

  for (let i = 0; i < world.numSurfers; i++) {
    h = mix(h, world.surferState[i]);
    h = mixFloat(h, world.surferX[i]);
    h = mixFloat(h, world.surferY[i]);
    h = mixFloat(h, world.surferVX[i]);
    h = mixFloat(h, world.surferVY[i]);
    h = mixFloat(h, world.surferStamina[i]);
    h = mixFloat(h, world.surferScore[i]);
  }

  const foam = world.ocean.foam;
  let foamSum = 0;
  for (let i = 0; i < foam.length; i++) {
    foamSum = (foamSum + Math.round(foam[i] * 65536)) >>> 0;
  }
  h = mix(h, foamSum);
  h = mix(h, foam.length);
  return h >>> 0;
}

/** Recorded initial config and per-step inputs for deterministic replay. */
export interface Recording {
  seed: number;
  options: CreateWorldOptions;
  inputs: SurferInput[][];
}

/** Captures per-step surfer inputs alongside the world's seed and options. */
export class Recorder {
  private readonly seed: number;
  private readonly options: CreateWorldOptions;
  private readonly inputs: SurferInput[][];

  constructor(world: WorldState, options: CreateWorldOptions) {
    this.seed = world.seed;
    this.options = options.swell
      ? { ...options, swell: { ...options.swell } }
      : { ...options };
    this.inputs = [];
  }

  record(inputs: readonly SurferInput[]): void {
    this.inputs.push(
      inputs.map((inp) => ({
        moveX: inp.moveX,
        moveY: inp.moveY,
        takeoff: inp.takeoff,
      }))
    );
  }

  get recording(): Recording {
    return {
      seed: this.seed,
      options: this.options,
      inputs: this.inputs,
    };
  }
}

export function startRecording(world: WorldState, options: CreateWorldOptions): Recorder {
  return new Recorder(world, options);
}

/** Rebuilds the world from a recording and replays every recorded step. */
export function replayRecording(recording: Recording): WorldState {
  const world = createWorld(recording.options, recording.seed);
  for (let step = 0; step < recording.inputs.length; step++) {
    stepWorld(world, recording.inputs[step]);
  }
  return world;
}
