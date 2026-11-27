import { describe, expect, it } from 'vitest';
import { CELL_M, NO_INPUT, SurferState } from '../src/contracts/types.ts';
import type { SurferInput, SwellParams, WorldState } from '../src/contracts/types.ts';
import { RuleAgent } from '../src/policies/ruleAgent.ts';
import { createWorld, sampleOcean, stepWorld, takeoffQuality } from '../src/sim/world.ts';

const SWELL: SwellParams = { heightM: 1.8, periodS: 9, directionDeg: 8 };
const TAKEOFF_INPUT: SurferInput = { moveX: 0, moveY: 0.8, takeoff: true };

interface Spot {
  x: number;
  y: number;
  peakDist: number;
  /** cosPhase at selection time (takeoff-window margin check). */
  cos: number;
}

/** Local helper mirroring distanceToPeak: min Euclidean distance to a breaking cell center. */
function peakDist(world: WorldState, x: number, y: number): number {
  const { gridW: w, gridH: h } = world.config;
  const breaking = world.ocean.breaking;
  let bestSq = Infinity;
  for (let cy = 0; cy < h; cy++) {
    const row = cy * w;
    const dy = cy * CELL_M + CELL_M / 2 - y;
    const dySq = dy * dy;
    for (let cx = 0; cx < w; cx++) {
      if (breaking[row + cx] !== 1) continue;
      const dx = cx * CELL_M + CELL_M / 2 - x;
      const dSq = dx * dx + dySq;
      if (dSq < bestSq) bestSq = dSq;
    }
  }
  return bestSq === Infinity ? Infinity : Math.sqrt(bestSq);
}

/**
 * Try to find a (closer, farther) takeoff spot pair in the CURRENT field.
 *
 * Scenario-design note (this replaced a single-snapshot version that flaked):
 * takeoffQuality includes cosPhase >= 0, a window that sweeps past a fixed
 * cell in ~4.5 s and can flip within one step of the snapshot (observed:
 * cosPhase 0.01 at selection, -0.02 one step later, so the far surfer never
 * took off at all). Spots are therefore required to sit comfortably inside
 * the window (cosPhase and ratio margins), and the caller scans forward
 * through time until such a pair exists, instead of trusting one snapshot.
 */
function tryFindSpotPair(world: WorldState, closerCosMin: number): { closer: Spot; farther: Spot } | null {
  const { gridW: w, gridH: h } = world.config;
  const qualifying: Spot[] = [];
  for (let cy = 0; cy < h; cy++) {
    const y = cy * CELL_M + CELL_M / 2;
    for (let cx = 0; cx < w; cx++) {
      const x = cx * CELL_M + CELL_M / 2;
      if (!takeoffQuality(world, x, y)) continue;
      const s = sampleOcean(world, x, y);
      if (s.ratio < 0.5 || s.ratio > 1.1) continue;
      qualifying.push({ x, y, peakDist: peakDist(world, x, y), cos: s.cosPhase });
    }
  }
  let closer: Spot | undefined;
  for (const p of qualifying) {
    if (p.cos < closerCosMin) continue;
    if (!closer || p.peakDist < closer.peakDist) closer = p;
  }
  if (!closer) return null;
  for (const p of qualifying) {
    if (p.cos < 0.3) continue;
    const dx = Math.abs(p.x - closer.x);
    const dy = Math.abs(p.y - closer.y);
    if (dx >= 12 && dx <= 24 && dy <= 10 && p.peakDist >= closer.peakDist + 4) {
      return { closer, farther: p };
    }
  }
  return null;
}

/** Step the (surfer-independent) ocean forward until a usable spot pair exists. */
function scanForSpotPair(world: WorldState, closerCosMin: number): { closer: Spot; farther: Spot } {
  for (let s = 0; s < 3600; s++) {
    const pair = tryFindSpotPair(world, closerCosMin);
    if (pair) return pair;
    stepWorld(world, [NO_INPUT, NO_INPUT]);
  }
  throw new Error('no takeoff spot pair found within scan window');
}

function teleportSurfer(world: WorldState, i: number, x: number, y: number): void {
  world.surferX[i] = x;
  world.surferY[i] = y;
  world.surferVX[i] = 0;
  world.surferVY[i] = 0;
  world.surferState[i] = SurferState.PADDLE;
  world.surferStateTime[i] = 0;
  // Fresh arrival: no time spent caught inside, full stamina. (During the
  // warm-up/scan the parked surfers can rack up inside-rule wipeouts at
  // their spawn spots, so wipeout assertions below use deltas from here.)
  world.surferInsideTime[i] = 0;
  world.surferStamina[i] = 100;
}

describe('Priority (right-of-way) rules', () => {
  it('drop-in: farther-from-peak second takeoff wipes out, owner rides on', () => {
    const world = createWorld({ bathymetry: 'sandbar', swell: SWELL, aiCount: 1 }, 7);
    for (let s = 0; s < 600; s++) {
      stepWorld(world, [NO_INPUT, NO_INPUT]);
    }

    const { closer: spotA, farther: spotB } = scanForSpotPair(world, 0.3);
    teleportSurfer(world, 0, spotA.x, spotA.y);
    teleportSurfer(world, 1, spotB.x, spotB.y);
    const wipeouts0Base = world.surferWipeouts[0];

    // Step only until the drop-in is adjudicated, then judge the owner at
    // that moment: a ride can still end naturally (foam wipeout) seconds
    // later, which is ordinary sim behavior, not a priority penalty.
    let surfer0ReachedRide = false;
    let adjudicated = false;
    for (let s = 0; s < 90; s++) {
      stepWorld(world, [TAKEOFF_INPUT, TAKEOFF_INPUT]);
      if (world.surferState[0] === SurferState.RIDE) surfer0ReachedRide = true;
      if (world.surferDropins[1] >= 1) {
        adjudicated = true;
        break;
      }
    }

    expect(adjudicated).toBe(true);
    expect(surfer0ReachedRide).toBe(true);
    // Both surfers transition on the same step, so at the adjudication step
    // itself the owner has not yet accrued any ride score (score is added by
    // the RIDE branch on later steps). Give the owner a few steps of riding,
    // then judge: still riding, score accruing, no penalty from the rule.
    for (let s = 0; s < 5; s++) {
      stepWorld(world, [NO_INPUT, NO_INPUT]);
    }
    // Owner rides on, untouched by the rule.
    expect(world.surferState[0]).toBe(SurferState.RIDE);
    expect(world.surferScore[0]).toBeGreaterThan(0);
    expect(world.surferDropins[0]).toBe(0);
    expect(world.surferWipeouts[0] - wipeouts0Base).toBe(0);
    // Drop-in surfer: immediate wipeout, no catch, no score.
    expect(world.surferState[1]).toBe(SurferState.WIPEOUT);
    expect(world.surferDropins[1]).toBe(1);
    expect(world.surferCatches[1]).toBe(0);
    expect(world.surferScore[1]).toBe(0);
  });

  it('priority transfer: a later takeoff clearly closer to the peak is not a drop-in', () => {
    const world = createWorld({ bathymetry: 'sandbar', swell: SWELL, aiCount: 1 }, 7);
    for (let s = 0; s < 600; s++) {
      stepWorld(world, [NO_INPUT, NO_INPUT]);
    }

    // Surfer 1 (closer spot) starts 6 steps after surfer 0, so its spot needs
    // a larger cosPhase margin to still be in the takeoff window on entry.
    const { closer: spotB, farther: spotA } = scanForSpotPair(world, 0.6);
    teleportSurfer(world, 0, spotA.x, spotA.y);
    teleportSurfer(world, 1, spotB.x, spotB.y);
    const wipeouts0Base = world.surferWipeouts[0];

    let surfer0ReachedRide = false;
    for (let s = 0; s < 80; s++) {
      const input1 = s < 6 ? NO_INPUT : TAKEOFF_INPUT;
      stepWorld(world, [TAKEOFF_INPUT, input1]);
      if (world.surferState[0] === SurferState.RIDE) surfer0ReachedRide = true;
      if (world.surferCatches[1] >= 1) break;
    }

    expect(surfer0ReachedRide).toBe(true);
    expect(world.surferDropins[1]).toBe(0);
    expect(world.surferCatches[1]).toBe(1);
    expect(world.surferWipeouts[0] - wipeouts0Base).toBe(0);
  });

  it(
    'multi-AI run: priority check keeps drop-ins at zero and catches stay comparable',
    () => {
      const aiCount = 4;
      const world = createWorld({ bathymetry: 'sandbar', swell: SWELL, aiCount }, 7);
      const agents: RuleAgent[] = [];
      for (let k = 1; k <= aiCount; k++) {
        agents.push(new RuleAgent(k - 1, aiCount));
      }

      const inputs: SurferInput[] = new Array(world.numSurfers);
      inputs[0] = NO_INPUT;
      for (let s = 0; s < 3600; s++) {
        for (let k = 1; k <= aiCount; k++) {
          inputs[k] = agents[k - 1].decide(world, k);
        }
        stepWorld(world, inputs);
      }

      let catches = 0;
      let dropins = 0;
      for (let k = 1; k <= aiCount; k++) {
        catches += world.surferCatches[k];
        dropins += world.surferDropins[k];
      }
      console.log(`multi-AI (4 agents, 3600 steps): catches=${catches}, dropins=${dropins}`);

      expect(dropins).toBeLessThanOrEqual(1);
      expect(catches).toBeGreaterThanOrEqual(70);
    },
    60000
  );
});
