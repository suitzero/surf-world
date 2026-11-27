import { describe, expect, it } from 'vitest';
import { NO_INPUT, SurferState } from '../src/contracts/types.ts';
import type { BathymetryKind, SurferInput, SwellParams, WorldState } from '../src/contracts/types.ts';
import { SeededRng } from '../src/sim/prng.ts';
import {
  createWorld,
  findTakeoffSpot,
  sampleOcean,
  stepWorld,
  takeoffQuality,
} from '../src/sim/world.ts';
import type { OceanSample } from '../src/sim/world.ts';

interface BotObservation {
  x: number;
  y: number;
  state: SurferState;
  ocean: OceanSample;
}

const DELAY_STEPS = 10; // 0.5 s at dt = 0.05
const ANGLE_REROLL_STEPS = 6;
const ANGLE_NOISE_RAD = 0.35;
const DROP_PROB = 0.1;

/**
 * Deliberately clumsy scripted player policy for game-feel verification.
 * Reads ONLY current observable state via sampleOcean(), findTakeoffSpot(),
 * and the surfer's own position/state from world arrays — delayed by 0.5 s
 * through a 10-step ring buffer, with noisy paddle direction and ~10% dropped inputs.
 */
export class BeginnerBot {
  private readonly rng: SeededRng;
  private readonly ring: Array<BotObservation | undefined> = new Array(DELAY_STEPS);
  private stepCount = 0;
  private angleNoise = 0;
  private spot: { x: number; y: number } | null = null;

  constructor(seed: number) {
    this.rng = new SeededRng(seed);
  }

  decide(world: WorldState, surferIndex = 0): SurferInput {
    if (this.spot === null) {
      this.spot = findTakeoffSpot(world, 0.5);
    }

    const ownX = world.surferX[surferIndex];
    const ownY = world.surferY[surferIndex];
    const ownState = world.surferState[surferIndex] as SurferState;
    const currentObs: BotObservation = {
      x: ownX,
      y: ownY,
      state: ownState,
      ocean: sampleOcean(world, ownX, ownY),
    };

    const slot = this.stepCount % DELAY_STEPS;
    const delayed = this.stepCount >= DELAY_STEPS ? this.ring[slot] : undefined;
    this.ring[slot] = currentObs;

    if (this.stepCount % ANGLE_REROLL_STEPS === 0) {
      this.angleNoise = this.rng.range(-ANGLE_NOISE_RAD, ANGLE_NOISE_RAD);
    }
    const dropped = this.rng.next() < DROP_PROB;
    this.stepCount++;

    if (!delayed || dropped) {
      return NO_INPUT;
    }

    const raw = this.computeIntent(delayed, this.spot);
    if (raw.moveX === 0 && raw.moveY === 0 && !raw.takeoff) {
      return NO_INPUT;
    }

    const c = Math.cos(this.angleNoise);
    const s = Math.sin(this.angleNoise);
    const rotX = Math.max(-1, Math.min(1, raw.moveX * c - raw.moveY * s));
    const rotY = Math.max(-1, Math.min(1, raw.moveX * s + raw.moveY * c));
    return { moveX: rotX, moveY: rotY, takeoff: raw.takeoff };
  }

  private computeIntent(
    obs: BotObservation,
    spot: { x: number; y: number }
  ): SurferInput {
    const { state, x, y, ocean } = obs;
    if (
      state === SurferState.TAKEOFF ||
      state === SurferState.RIDE ||
      state === SurferState.WIPEOUT ||
      state === SurferState.KICKOUT
    ) {
      return NO_INPUT;
    }

    const dx = spot.x - x;
    const dy = spot.y - y;
    const dist = Math.hypot(dx, dy);

    if (state === SurferState.RETURN) {
      if (dist <= 6) return NO_INPUT;
      return { moveX: dx / dist, moveY: dy / dist, takeoff: false };
    }

    // Approaching crest on delayed observation: paddle hard shoreward.
    if (ocean.cosPhase >= 0.3 && ocean.ratio >= 0.4) {
      const corrX = Math.max(-0.25, Math.min(0.25, dx * 0.04));
      return { moveX: corrX, moveY: 1, takeoff: false };
    }

    // Wait when within 6 m of the takeoff spot; otherwise paddle toward it.
    if (dist <= 6) {
      return NO_INPUT;
    }
    return { moveX: dx / dist, moveY: dy / dist, takeoff: false };
  }
}

const SWELL: SwellParams = { heightM: 1.8, periodS: 9, directionDeg: 8 };
const TERRAINS: readonly BathymetryKind[] = ['sandbar', 'beach'];
const SEEDS: readonly number[] = [7, 11, 42, 99];
const MAX_STEPS = 1200;

describe('P1 beginner bot game-feel verification', () => {
  for (const terrain of TERRAINS) {
    it(
      `beginner bot catches a wave on at least 3 of 4 seeds (${terrain})`,
      () => {
        let catches = 0;

        for (const seed of SEEDS) {
          const world = createWorld(
            { bathymetry: terrain, swell: SWELL, aiCount: 0 },
            seed
          );
          const bot = new BeginnerBot(seed * 1009 + (terrain === 'sandbar' ? 1 : 2));
          let firstCatchStep = -1;
          let takeoffQualitySteps = 0;
          let wantsAndQualitySteps = 0;
          let minStamina = 100;

          for (let step = 1; step <= MAX_STEPS; step++) {
            const input = bot.decide(world, 0);
            const qual = takeoffQuality(world, world.surferX[0], world.surferY[0]);
            if (qual) {
              takeoffQualitySteps++;
              if (input.takeoff || input.moveY > 0.55) wantsAndQualitySteps++;
            }
            stepWorld(world, [input]);
            if (world.surferStamina[0] < minStamina) {
              minStamina = world.surferStamina[0];
            }
            if (world.surferCatches[0] >= 1) {
              firstCatchStep = step;
              break;
            }
          }

          const wipeouts = world.surferWipeouts[0];
          if (firstCatchStep >= 0) catches++;
          console.log(
            `[beginner] terrain=${terrain} seed=${seed} catchStep=${firstCatchStep >= 0 ? firstCatchStep : 'NONE'} wipeouts=${wipeouts} qualSteps=${takeoffQualitySteps} wantsAndQual=${wantsAndQualitySteps} minStamina=${minStamina.toFixed(1)}`
          );
        }

        expect(catches).toBeGreaterThanOrEqual(3);
      },
      120000
    );
  }
});
