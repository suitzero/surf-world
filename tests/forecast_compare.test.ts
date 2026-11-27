import { describe, expect, it } from 'vitest';
import { NO_INPUT, SurferState } from '../src/contracts/types.ts';
import type { BathymetryKind, SurferInput, SwellParams, WorldState } from '../src/contracts/types.ts';
import { ForecastAgent } from '../src/policies/forecastAgent.ts';
import { RuleAgent } from '../src/policies/ruleAgent.ts';
import { createWorld, stepWorld } from '../src/sim/world.ts';

const SWELL: SwellParams = { heightM: 1.8, periodS: 9, directionDeg: 8 };
const STEPS = 3600; // 3 sim-minutes at dt = 0.05
const AI_COUNT = 4;

interface PolicyAgent {
  decide(world: WorldState, i: number): SurferInput;
}

interface ArmMetrics {
  catches: number;
  rideSeconds: number;
  meanRide: number;
  wipeouts: number;
  dropins: number;
}

function runArm(
  kind: BathymetryKind,
  seed: number,
  makeAgents: (k: number) => PolicyAgent
): ArmMetrics {
  const world = createWorld({ bathymetry: kind, swell: SWELL, aiCount: AI_COUNT }, seed);
  const agents: PolicyAgent[] = [];
  for (let k = 0; k < AI_COUNT; k++) {
    agents.push(makeAgents(k));
  }

  const inputs: SurferInput[] = new Array(world.numSurfers);
  const dt = world.config.dt;
  let rideSeconds = 0;

  for (let s = 0; s < STEPS; s++) {
    inputs[0] = NO_INPUT;
    for (let k = 1; k <= AI_COUNT; k++) {
      inputs[k] = agents[k - 1].decide(world, k);
    }
    stepWorld(world, inputs);
    for (let k = 1; k <= AI_COUNT; k++) {
      if (world.surferState[k] === SurferState.RIDE) {
        rideSeconds += dt;
      }
    }
  }

  let catches = 0;
  let wipeouts = 0;
  let dropins = 0;
  for (let k = 1; k <= AI_COUNT; k++) {
    catches += world.surferCatches[k];
    wipeouts += world.surferWipeouts[k];
    dropins += world.surferDropins[k];
  }

  return {
    catches,
    rideSeconds,
    meanRide: catches > 0 ? rideSeconds / catches : 0,
    wipeouts,
    dropins,
  };
}

describe('S12 forecast vs rule agent comparison', () => {
  it(
    'forecast vs rule: matched comparison',
    { timeout: 900000 },
    () => {
      const kinds: readonly BathymetryKind[] = ['sandbar', 'beach'];
      const seeds = [7, 42, 123] as const;

      const ruleTotal = { catches: 0, rideSeconds: 0, wipeouts: 0, dropins: 0 };
      const forecastTotal = { catches: 0, rideSeconds: 0, wipeouts: 0, dropins: 0 };

      for (const kind of kinds) {
        for (const seed of seeds) {
          const rule = runArm(kind, seed, (k) => new RuleAgent(k, AI_COUNT));
          const forecast = runArm(kind, seed, (k) => new ForecastAgent(k, AI_COUNT));

          console.log(
            `[S12 ${kind} seed=${seed}] ` +
              `rule(catches=${rule.catches}, meanRide=${rule.meanRide.toFixed(2)}s, wipeouts=${rule.wipeouts}, dropins=${rule.dropins}) | ` +
              `forecast(catches=${forecast.catches}, meanRide=${forecast.meanRide.toFixed(2)}s, wipeouts=${forecast.wipeouts}, dropins=${forecast.dropins})`
          );

          ruleTotal.catches += rule.catches;
          ruleTotal.rideSeconds += rule.rideSeconds;
          ruleTotal.wipeouts += rule.wipeouts;
          ruleTotal.dropins += rule.dropins;

          forecastTotal.catches += forecast.catches;
          forecastTotal.rideSeconds += forecast.rideSeconds;
          forecastTotal.wipeouts += forecast.wipeouts;
          forecastTotal.dropins += forecast.dropins;
        }
      }

      const ruleMeanRide = ruleTotal.catches > 0 ? ruleTotal.rideSeconds / ruleTotal.catches : 0;
      const forecastMeanRide =
        forecastTotal.catches > 0 ? forecastTotal.rideSeconds / forecastTotal.catches : 0;

      console.log(
        `[S12 TOTAL] ` +
          `rule(catches=${ruleTotal.catches}, meanRide=${ruleMeanRide.toFixed(2)}s, wipeouts=${ruleTotal.wipeouts}, dropins=${ruleTotal.dropins}) | ` +
          `forecast(catches=${forecastTotal.catches}, meanRide=${forecastMeanRide.toFixed(2)}s, wipeouts=${forecastTotal.wipeouts}, dropins=${forecastTotal.dropins})`
      );

      // Sanity: harness is not broken.
      expect(ruleTotal.catches).toBeGreaterThan(30);
      expect(forecastTotal.catches).toBeGreaterThan(30);

      // Verdict (plan section 13: prediction use must raise catch rate without more drop-ins).
      expect(forecastTotal.catches).toBeGreaterThanOrEqual(Math.ceil(ruleTotal.catches * 1.1));
      expect(forecastTotal.dropins).toBeLessThanOrEqual(ruleTotal.dropins);
    }
  );
});
