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
  wipeouts: number;
  dropins: number;
  decideMeanMs: number;
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
  let decideTotalMs = 0;
  let decideCalls = 0;
  let rideSeconds = 0;

  for (let s = 0; s < STEPS; s++) {
    inputs[0] = NO_INPUT;
    for (let k = 1; k <= AI_COUNT; k++) {
      const t0 = performance.now();
      inputs[k] = agents[k - 1].decide(world, k);
      decideTotalMs += performance.now() - t0;
      decideCalls++;
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
    wipeouts,
    dropins,
    decideMeanMs: decideTotalMs / decideCalls,
  };
}

describe('S14 planner budget curve (candidates x horizon)', () => {
  it(
    'budget sweep on reduced matched set',
    { timeout: 1200000 },
    () => {
      // Reduced matched set (stated in advance): both kinds x seeds {7, 42}.
      // The full S13 set (x seed 123) is unaffordable across 12 configs;
      // the rule reference below is measured on this same reduced set.
      const kinds: readonly BathymetryKind[] = ['sandbar', 'beach'];
      const seeds = [7, 42] as const;
      const candidateGrid = [4, 12, 24, 48] as const;
      const horizonGrid = [10, 20, 40] as const;

      const runSet = (makeAgents: (k: number) => PolicyAgent) => {
        const total = { catches: 0, rideSeconds: 0, wipeouts: 0, dropins: 0, decideMeanMs: 0 };
        let runs = 0;
        for (const kind of kinds) {
          for (const seed of seeds) {
            const m = runArm(kind, seed, makeAgents);
            total.catches += m.catches;
            total.rideSeconds += m.rideSeconds;
            total.wipeouts += m.wipeouts;
            total.dropins += m.dropins;
            total.decideMeanMs += m.decideMeanMs;
            runs++;
          }
        }
        total.decideMeanMs /= runs;
        return total;
      };

      const rule = runSet((k) => new RuleAgent(k, AI_COUNT));
      console.log(
        `[S14 REF rule] catches=${rule.catches}, wipeouts=${rule.wipeouts}, dropins=${rule.dropins}, decideMean=${rule.decideMeanMs.toFixed(4)}ms`
      );
      expect(rule.catches).toBeGreaterThan(30);

      console.log('[S14 TABLE] candidates | horizonS | catches | wipeouts | dropins | decideMeanMs');
      for (const candidates of candidateGrid) {
        for (const horizonS of horizonGrid) {
          const m = runSet((k) => new PlannerAgent(k, AI_COUNT, { candidates, horizonS }));
          console.log(
            `[S14 ROW] ${candidates} | ${horizonS} | ${m.catches} | ${m.wipeouts} | ${m.dropins} | ${m.decideMeanMs.toFixed(4)}`
          );
          expect(m.catches).toBeGreaterThan(0);
        }
      }
    }
  );
});
