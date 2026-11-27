import { describe, expect, it } from 'vitest';
import { NO_INPUT, SurferState } from '../src/contracts/types.ts';
import type { BathymetryKind, SurferInput, SwellParams, WorldState } from '../src/contracts/types.ts';
import { PlannerAgent } from '../src/policies/plannerAgent.ts';
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
  /** Wall-time per decide() call across all agents in the run, milliseconds. */
  decideMeanMs: number;
  decideP95Ms: number;
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
  const decideMs: number[] = [];
  let rideSeconds = 0;

  for (let s = 0; s < STEPS; s++) {
    inputs[0] = NO_INPUT;
    for (let k = 1; k <= AI_COUNT; k++) {
      const t0 = performance.now();
      inputs[k] = agents[k - 1].decide(world, k);
      decideMs.push(performance.now() - t0);
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

  decideMs.sort((a, b) => a - b);
  const decideMeanMs = decideMs.reduce((a, b) => a + b, 0) / decideMs.length;
  const decideP95Ms = decideMs[Math.min(decideMs.length - 1, Math.floor(decideMs.length * 0.95))];

  return {
    catches,
    rideSeconds,
    meanRide: catches > 0 ? rideSeconds / catches : 0,
    wipeouts,
    dropins,
    decideMeanMs,
    decideP95Ms,
  };
}

describe('S13 planner vs rule agent comparison', () => {
  it(
    'planner vs rule: matched comparison with decision cost',
    { timeout: 1200000 },
    () => {
      const kinds: readonly BathymetryKind[] = ['sandbar', 'beach'];
      const seeds = [7, 42, 123] as const;

      const ruleTotal = { catches: 0, rideSeconds: 0, wipeouts: 0, dropins: 0 };
      const plannerTotal = { catches: 0, rideSeconds: 0, wipeouts: 0, dropins: 0 };
      const ruleDecide: number[] = [];
      const plannerDecide: number[] = [];

      for (const kind of kinds) {
        for (const seed of seeds) {
          const rule = runArm(kind, seed, (k) => new RuleAgent(k, AI_COUNT));
          const planner = runArm(kind, seed, (k) => new PlannerAgent(k, AI_COUNT));

          console.log(
            `[S13 ${kind} seed=${seed}] ` +
              `rule(catches=${rule.catches}, meanRide=${rule.meanRide.toFixed(2)}s, wipeouts=${rule.wipeouts}, dropins=${rule.dropins}, decideMean=${rule.decideMeanMs.toFixed(3)}ms, decideP95=${rule.decideP95Ms.toFixed(3)}ms) | ` +
              `planner(catches=${planner.catches}, meanRide=${planner.meanRide.toFixed(2)}s, wipeouts=${planner.wipeouts}, dropins=${planner.dropins}, decideMean=${planner.decideMeanMs.toFixed(3)}ms, decideP95=${planner.decideP95Ms.toFixed(3)}ms)`
          );

          ruleTotal.catches += rule.catches;
          ruleTotal.rideSeconds += rule.rideSeconds;
          ruleTotal.wipeouts += rule.wipeouts;
          ruleTotal.dropins += rule.dropins;
          ruleDecide.push(rule.decideMeanMs);

          plannerTotal.catches += planner.catches;
          plannerTotal.rideSeconds += planner.rideSeconds;
          plannerTotal.wipeouts += planner.wipeouts;
          plannerTotal.dropins += planner.dropins;
          plannerDecide.push(planner.decideMeanMs);
        }
      }

      const ruleMeanRide = ruleTotal.catches > 0 ? ruleTotal.rideSeconds / ruleTotal.catches : 0;
      const plannerMeanRide =
        plannerTotal.catches > 0 ? plannerTotal.rideSeconds / plannerTotal.catches : 0;
      const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

      console.log(
        `[S13 TOTAL] ` +
          `rule(catches=${ruleTotal.catches}, meanRide=${ruleMeanRide.toFixed(2)}s, wipeouts=${ruleTotal.wipeouts}, dropins=${ruleTotal.dropins}, decideMean=${avg(ruleDecide).toFixed(3)}ms) | ` +
          `planner(catches=${plannerTotal.catches}, meanRide=${plannerMeanRide.toFixed(2)}s, wipeouts=${plannerTotal.wipeouts}, dropins=${plannerTotal.dropins}, decideMean=${avg(plannerDecide).toFixed(3)}ms)`
      );
      // Cross-check: S12 measured ruleTotal.catches = 646 on this identical set.
      console.log(`[S13 CHECK] rule catches reproduced: ${ruleTotal.catches} (S12 record: 646)`);

      // Sanity: harness is not broken.
      expect(ruleTotal.catches).toBeGreaterThan(30);
      expect(plannerTotal.catches).toBeGreaterThan(30);

      // Verdict (pre-registered): planner must catch >= +10% vs rule on the
      // identical matched set, with no more drop-ins and wipeouts up at most 10%.
      // If this fails, the failure stays as the measurement record (S12 convention).
      expect(plannerTotal.catches).toBeGreaterThanOrEqual(Math.ceil(ruleTotal.catches * 1.1));
      expect(plannerTotal.dropins).toBeLessThanOrEqual(ruleTotal.dropins);
      expect(plannerTotal.wipeouts).toBeLessThanOrEqual(ruleTotal.wipeouts * 1.1);
    }
  );
});
