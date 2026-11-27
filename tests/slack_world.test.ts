import { describe, expect, it } from 'vitest';
import { SlackPolicy } from '../src/slack/policy.ts';
import { SlackRecorder, replaySlackRecording } from '../src/slack/replay.ts';
import { createTeacher } from '../src/slack/teacher.ts';
import { INITIAL_POS_M, ZERO_ACTION } from '../src/slack/types.ts';
import { createSlackWorld, hashSlackState, stepSlackWorld } from '../src/slack/world.ts';

describe('slackline world, teacher, policy, and replay', () => {
  it('is deterministic for identical seeds and diverges for different seeds', () => {
    const worldA = createSlackWorld(7);
    const worldB = createSlackWorld(7);
    const worldOther = createSlackWorld(8);

    const policyA = new SlackPolicy();
    const policyB = new SlackPolicy();
    const policyOther = new SlackPolicy();

    for (let step = 0; step < 600; step++) {
      stepSlackWorld(worldA, policyA.decide(worldA));
      stepSlackWorld(worldB, policyB.decide(worldB));
      stepSlackWorld(worldOther, policyOther.decide(worldOther));
    }

    const hashA = hashSlackState(worldA);
    const hashB = hashSlackState(worldB);
    const hashOther = hashSlackState(worldOther);

    expect(hashA).toBe(hashB);
    expect(hashA).not.toBe(hashOther);
  });

  it('has pure deterministic windAt(t) across instances with the same seed', () => {
    const t1 = createTeacher(42);
    const t2 = createTeacher(42);
    const sampleTimes = [0, 0.37, 1.5, 4.0, 9.25, 17.8, 45.125];

    // Query forward on t1 and reverse on t2 to verify order-independent purity.
    const vals1 = sampleTimes.map((t) => t1.windAt(t));
    const vals2Reversed = [...sampleTimes].reverse().map((t) => t2.windAt(t));
    const vals2 = vals2Reversed.reverse();

    expect(vals1).toEqual(vals2);

    // Stepping t1 must not mutate windAt(t).
    t1.step(ZERO_ACTION);
    const vals1AfterStep = sampleTimes.map((t) => t1.windAt(t));
    expect(vals1AfterStep).toEqual(vals1);
  });

  it('falls within 10 s under zero control from theta0 = 0.12', () => {
    const teacher = createTeacher(7, { theta0: 0.12 });
    const maxSteps = 200; // 200 * 0.05 s = 10 s
    for (let step = 0; step < maxSteps; step++) {
      teacher.step(ZERO_ACTION);
      if (teacher.fallen) break;
    }

    expect(teacher.fallen).toBe(true);
    expect(teacher.time).toBeLessThanOrEqual(10.0 + 1e-9);
  });

  it('balances >=20 s and advances >=5 m within 60 s in >=3 of 4 seeds from theta0 = 0.08', () => {
    const seeds = [7, 42, 123, 2024];
    const maxSteps = 1200; // 60 s at dt = 0.05 s
    const twentySecStep = 400; // 20 s at dt = 0.05 s

    let passedCount = 0;
    for (const seed of seeds) {
      const world = createSlackWorld(seed, { theta0: 0.08 });
      const policy = new SlackPolicy();
      let balancedAt20s = false;

      for (let step = 1; step <= maxSteps; step++) {
        stepSlackWorld(world, policy.decide(world));
        if (step === twentySecStep && !world.fallen) {
          balancedAt20s = true;
        }
      }

      const advancedM = world.s - INITIAL_POS_M;
      if (balancedAt20s && advancedM >= 5.0) {
        passedCount++;
      }
    }

    expect(passedCount).toBeGreaterThanOrEqual(3);
  });

  it('replays a 600-step policy-driven recording with an identical state hash', () => {
    const seed = 123;
    const world = createSlackWorld(seed, { theta0: 0.08 });
    const policy = new SlackPolicy();
    const recorder = new SlackRecorder(world);

    for (let step = 0; step < 600; step++) {
      const action = policy.decide(world);
      recorder.record(action);
      stepSlackWorld(world, action);
    }

    const liveHash = hashSlackState(world);
    const replayed = replaySlackRecording(recorder.recording);
    const replayHash = hashSlackState(replayed);

    expect(replayed.stepCount).toBe(600);
    expect(replayHash).toBe(liveHash);
  });
});
