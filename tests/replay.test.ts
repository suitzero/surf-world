import { describe, expect, it } from 'vitest';
import { NO_INPUT } from '../src/contracts/types.ts';
import type { SurferInput } from '../src/contracts/types.ts';
import { SeededRng } from '../src/sim/prng.ts';
import { hashWorldState, replayRecording, startRecording } from '../src/sim/replay.ts';
import { createWorld, stepWorld } from '../src/sim/world.ts';
import type { CreateWorldOptions } from '../src/sim/world.ts';

describe('P0 deterministic replay', () => {
  it('replays a 600-step recorded session with bit-identical state hashes', () => {
    const options: CreateWorldOptions = {
      bathymetry: 'sandbar',
      swell: { heightM: 1.8, periodS: 9, directionDeg: 8 },
      aiCount: 2,
    };
    const seed = 1234;
    const checkpoints = new Set([100, 300, 600]);

    const world = createWorld(options, seed);
    const recorder = startRecording(world, options);
    const rng = new SeededRng(777);
    const liveHashes: number[] = [];

    for (let step = 0; step < 600; step++) {
      const playerInput: SurferInput = {
        moveX: rng.range(-1, 1),
        moveY: rng.range(-1, 1),
        takeoff: rng.next() < 0.05,
      };
      const inputs: SurferInput[] = [playerInput, NO_INPUT, NO_INPUT];
      recorder.record(inputs);
      stepWorld(world, inputs);
      if (checkpoints.has(world.stepCount)) {
        liveHashes.push(hashWorldState(world));
      }
    }

    const recorded = recorder.recording;
    const replayWorld = createWorld(options, seed);
    const replayHashes: number[] = [];
    for (let step = 0; step < recorded.inputs.length; step++) {
      stepWorld(replayWorld, recorded.inputs[step]);
      if (checkpoints.has(replayWorld.stepCount)) {
        replayHashes.push(hashWorldState(replayWorld));
      }
    }

    expect(replayHashes).toEqual(liveHashes);
    expect(hashWorldState(replayRecording(recorded))).toEqual(liveHashes[liveHashes.length - 1]);
    expect(replayWorld.stepCount).toBe(600);
  });
});
