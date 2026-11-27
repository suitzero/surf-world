import {
  CELL_M,
  CONTRACT_VERSION,
  GRID_H,
  GRID_W,
  SurferState,
  WORLD_H_M,
  WORLD_W_M,
} from '../contracts/types.ts';
import type {
  BathymetryKind,
  SurferInput,
  SwellParams,
  WorldConfig,
  WorldState,
} from '../contracts/types.ts';
import { generateBathymetry } from './bathymetry.ts';
import { BREAKING_RATIO_MAX, buildOcean, peelSignAt, updateOcean } from './ocean.ts';
import { normalizeSeed } from './prng.ts';

export const DEFAULT_SWELL: SwellParams = { heightM: 1.8, periodS: 9, directionDeg: 8 };

export const TUNING = {
  /** Base paddling speed in m/s. */
  paddleSpeed: 4.0,
  /** Velocity blend rate toward target paddle velocity (1/s). */
  paddleAccelRate: 8,
  /** Stamina drained per second while actively paddling. */
  staminaDrainPerSec: 7,
  /** Stamina regenerated per second while resting/waiting. */
  staminaRegenPerSec: 5,
  /** Stamina level below which paddle speed is reduced. */
  staminaLowThreshold: 15,
  /** Paddle speed multiplier applied below staminaLowThreshold. */
  staminaLowFactor: 0.5,
  /** Minimum effective water depth (m) required for takeoff. */
  takeoffMinDepth: 0.8,
  /** Minimum wave height-to-depth ratio required for takeoff. */
  takeoffRatioMin: 0.42,
  /** Maximum wave height-to-depth ratio allowed for takeoff. */
  takeoffRatioMax: 1.25,
  /** Minimum cos(phase) (crest proximity) required for takeoff. */
  takeoffCosPhaseMin: 0.0,
  /** Maximum foam level allowed for takeoff. */
  takeoffFoamMax: 0.85,
  /** Foam threshold that counts as being caught inside. */
  insideFoamThreshold: 0.8,
  /** Seconds allowed in breaking/foamy water before forced wipeout. */
  insideGraceTime: 1.0,
  /** Duration of the takeoff pop-up state in seconds. */
  takeoffTime: 0.35,
  /** Duration of the wipeout state in seconds. */
  wipeoutTime: 2.2,
  /** Duration of the kickout state in seconds. */
  kickoutTime: 0.5,
} as const;

export interface CreateWorldOptions {
  bathymetry?: BathymetryKind;
  swell?: SwellParams;
  tideM?: number;
  aiCount?: number;
  dt?: number;
}

export function createWorld(opts: CreateWorldOptions, seed: number): WorldState {
  const config: WorldConfig = {
    version: CONTRACT_VERSION,
    gridW: GRID_W,
    gridH: GRID_H,
    cellM: CELL_M,
    dt: opts.dt ?? 0.05,
    tideM: opts.tideM ?? 0,
    bathymetry: opts.bathymetry ?? 'sandbar',
    aiCount: Math.max(0, Math.min(8, opts.aiCount ?? 3)),
  };
  const swell: SwellParams = { ...(opts.swell ?? DEFAULT_SWELL) };
  const bathy = generateBathymetry(config.bathymetry, seed);
  const ocean = buildOcean(bathy, swell, config.tideM);
  const numSurfers = 1 + config.aiCount;
  const world: WorldState = {
    config,
    seed,
    rngState: normalizeSeed(seed),
    time: 0,
    stepCount: 0,
    bathy,
    swell,
    ocean,
    numSurfers,
    surferX: new Float32Array(numSurfers),
    surferY: new Float32Array(numSurfers),
    surferVX: new Float32Array(numSurfers),
    surferVY: new Float32Array(numSurfers),
    surferHeading: new Float32Array(numSurfers).fill(Math.PI / 2),
    surferState: new Uint8Array(numSurfers),
    surferStateTime: new Float32Array(numSurfers),
    surferStamina: new Float32Array(numSurfers).fill(100),
    surferScore: new Float32Array(numSurfers),
    surferCatches: new Uint16Array(numSurfers),
    surferWipeouts: new Uint16Array(numSurfers),
    surferRideTime: new Float32Array(numSurfers),
    surferBestRide: new Float32Array(numSurfers),
    surferInsideTime: new Float32Array(numSurfers),
    rideDirX: new Float32Array(numSurfers),
    rideDirY: new Float32Array(numSurfers),
  };
  // Spawn the lineup spread along the takeoff contour.
  for (let i = 0; i < numSurfers; i++) {
    const spot = findTakeoffSpot(world, 0.25 + 0.5 * (numSurfers === 1 ? 0.5 : i / (numSurfers - 1 || 1)));
    world.surferX[i] = spot.x;
    world.surferY[i] = Math.max(8, spot.y - 14); // start a little outside
    world.surferState[i] = SurferState.PADDLE;
  }
  return world;
}

/** Rebuild the ocean field in place after swell/tide changes (keeps foam). */
export function rebuildOcean(world: WorldState): void {
  const foam = world.ocean.foam;
  world.ocean = buildOcean(world.bathy, world.swell, world.config.tideM);
  world.ocean.foam.set(foam);
}

export function cellIndexAt(world: WorldState, xM: number, yM: number): number {
  const cx = Math.max(0, Math.min(world.config.gridW - 1, Math.floor(xM / CELL_M)));
  const cy = Math.max(0, Math.min(world.config.gridH - 1, Math.floor(yM / CELL_M)));
  return cy * world.config.gridW + cx;
}

export interface OceanSample {
  depth: number;
  ratio: number;
  cosPhase: number;
  breaking: boolean;
  foam: number;
  speed: number;
}

export function sampleOcean(world: WorldState, xM: number, yM: number): OceanSample {
  const i = cellIndexAt(world, xM, yM);
  const o = world.ocean;
  return {
    depth: o.depthEff[i],
    ratio: o.ratio[i],
    cosPhase: o.cosPhase[i],
    breaking: o.breaking[i] === 1,
    foam: o.foam[i],
    speed: o.speed[i],
  };
}

/**
 * Is this spot a plausible takeoff right now? Uses only current observable
 * field values: a steep face (ratio in the pre-breaking band), crest close,
 * and not already buried in whitewater.
 */
export function takeoffQuality(world: WorldState, xM: number, yM: number): boolean {
  const s = sampleOcean(world, xM, yM);
  return (
    s.depth >= TUNING.takeoffMinDepth &&
    s.ratio >= TUNING.takeoffRatioMin &&
    s.ratio <= TUNING.takeoffRatioMax &&
    s.cosPhase >= TUNING.takeoffCosPhaseMin &&
    s.foam <= TUNING.takeoffFoamMax
  );
}

/**
 * Find a takeoff-contour spot: the cell closest to alongshore fraction
 * `xFrac` whose depth puts it in the pre-breaking band for a set wave.
 */
export function findTakeoffSpot(world: WorldState, xFrac: number): { x: number; y: number } {
  const { gridW: w, gridH: h } = world.config;
  const o = world.ocean;
  const targetX = xFrac * (w - 1);
  let best = -1;
  let bestScore = Infinity;
  for (let y = 4; y < h - 4; y++) {
    for (let x = 2; x < w - 2; x++) {
      const i = y * w + x;
      const d = o.depthEff[i];
      if (d < 1.2 || d > 10) continue;
      const setRatio = o.amp[i] / d; // ratio at envelope peak
      if (setRatio < 0.45 || setRatio > 0.85) continue;
      const score = Math.abs(x - targetX) * 1.5 + Math.abs(setRatio - 0.66) * 60;
      if (score < bestScore) {
        bestScore = score;
        best = i;
      }
    }
  }
  if (best < 0) return { x: WORLD_W_M / 2, y: WORLD_H_M * 0.45 };
  return { x: (best % w) * CELL_M + CELL_M / 2, y: Math.floor(best / w) * CELL_M + CELL_M / 2 };
}

function setState(world: WorldState, i: number, state: SurferState): void {
  world.surferState[i] = state;
  world.surferStateTime[i] = 0;
}

/** Advance the whole world by one fixed step. inputs[0] is the player. */
export function stepWorld(world: WorldState, inputs: readonly SurferInput[]): void {
  const dt = world.config.dt;
  world.time += dt;
  world.stepCount++;
  updateOcean(world.ocean, world.swell, world.time, world.config.gridW, world.config.gridH, world.config.dt);
  for (let i = 0; i < world.numSurfers; i++) {
    stepSurfer(world, i, inputs[i] ?? { moveX: 0, moveY: 0, takeoff: false }, dt);
  }
}

function stepSurfer(world: WorldState, i: number, input: SurferInput, dt: number): void {
  const state = world.surferState[i] as SurferState;
  world.surferStateTime[i] += dt;
  const st = world.surferStateTime[i];
  let x = world.surferX[i];
  let y = world.surferY[i];
  let vx = world.surferVX[i];
  let vy = world.surferVY[i];
  const local = sampleOcean(world, x, y);
  const stamina = world.surferStamina[i];

  switch (state) {
    case SurferState.PADDLE:
    case SurferState.WAIT:
    case SurferState.RETURN: {
      const mag = Math.hypot(input.moveX, input.moveY);
      const moving = mag > 0.05;
      const staminaFactor = stamina < TUNING.staminaLowThreshold ? TUNING.staminaLowFactor : 1;
      const tx = moving ? (input.moveX / Math.max(1, mag)) * TUNING.paddleSpeed * staminaFactor : 0;
      const ty = moving ? (input.moveY / Math.max(1, mag)) * TUNING.paddleSpeed * staminaFactor : 0;
      const blend = Math.min(1, TUNING.paddleAccelRate * dt);
      vx += (tx - vx) * blend;
      vy += (ty - vy) * blend;
      if (moving) {
        world.surferStamina[i] = Math.max(0, stamina - TUNING.staminaDrainPerSec * dt);
      } else {
        world.surferStamina[i] = Math.min(100, stamina + TUNING.staminaRegenPerSec * dt);
      }
      // Caught inside: sitting in the break zone too long washes you out.
      if (local.breaking || local.foam > TUNING.insideFoamThreshold) {
        world.surferInsideTime[i] += dt;
        if (world.surferInsideTime[i] > TUNING.insideGraceTime) {
          world.surferWipeouts[i]++;
          world.surferStamina[i] = Math.max(0, world.surferStamina[i] - 12);
          setState(world, i, SurferState.WIPEOUT);
          break;
        }
      } else {
        world.surferInsideTime[i] = Math.max(0, world.surferInsideTime[i] - 2 * dt);
      }
      // Takeoff: explicit signal or paddling hard toward shore on a face.
      const wantsTakeoff = input.takeoff || input.moveY > 0.55;
      if (wantsTakeoff && takeoffQuality(world, x, y)) {
        setState(world, i, SurferState.TAKEOFF);
        break;
      }
      if (state === SurferState.RETURN && (local.depth >= 6.5 || st > 8)) {
        setState(world, i, SurferState.PADDLE);
      } else if (state !== SurferState.RETURN) {
        const speedNow = Math.hypot(vx, vy);
        world.surferState[i] =
          !moving && speedNow < 0.3 ? SurferState.WAIT : SurferState.PADDLE;
        world.surferStateTime[i] = st; // preserve timer across PADDLE<->WAIT
      }
      break;
    }
    case SurferState.TAKEOFF: {
      // Short pop-up: drift shoreward with the face, then riding.
      vx *= 0.9;
      vy += (2.2 - vy) * Math.min(1, 8 * dt);
      if (st >= TUNING.takeoffTime) {
        const cx = Math.floor(x / CELL_M);
        const cy = Math.floor(y / CELL_M);
        const peel = peelSignAt(world.ocean, world.config.gridW, cx, cy);
        const dirX = peel * 0.9;
        const dirY = 0.45;
        const len = Math.hypot(dirX, dirY) || 1;
        world.rideDirX[i] = dirX / len;
        world.rideDirY[i] = dirY / len;
        world.surferCatches[i]++;
        world.surferRideTime[i] = 0;
        setState(world, i, SurferState.RIDE);
      }
      break;
    }
    case SurferState.RIDE: {
      const rideSpeed = Math.max(2.4, Math.min(6.5, local.speed * 0.8));
      vx = world.rideDirX[i] * rideSpeed;
      vy = world.rideDirY[i] * rideSpeed;
      world.surferRideTime[i] += dt;
      world.surferScore[i] += rideSpeed * dt * 1.5 + dt * 2;
      if (world.surferRideTime[i] > world.surferBestRide[i]) {
        world.surferBestRide[i] = world.surferRideTime[i];
      }
      const ahead = sampleOcean(
        world,
        x + world.rideDirX[i] * 7,
        y + world.rideDirY[i] * 7
      );
      const faceAhead =
        ahead.breaking ||
        (ahead.ratio >= 0.45 && ahead.ratio <= BREAKING_RATIO_MAX && ahead.cosPhase > 0);
      if (local.depth < 0.55 || world.surferRideTime[i] > 25) {
        setState(world, i, SurferState.KICKOUT);
      } else if (!faceAhead) {
        if (local.foam > 0.6) {
          world.surferWipeouts[i]++;
          world.surferStamina[i] = Math.max(0, world.surferStamina[i] - 12);
          setState(world, i, SurferState.WIPEOUT);
        } else {
          setState(world, i, SurferState.KICKOUT);
        }
      }
      break;
    }
    case SurferState.WIPEOUT: {
      // Tumbled shoreward by the whitewater.
      vx *= 0.92;
      vy += (0.9 - vy) * Math.min(1, 3 * dt);
      if (st >= TUNING.wipeoutTime) setState(world, i, SurferState.RETURN);
      break;
    }
    case SurferState.KICKOUT: {
      // Over the back of the wave, then paddle back out.
      vx *= 0.9;
      vy += (-2.0 - vy) * Math.min(1, 6 * dt);
      if (st >= TUNING.kickoutTime) setState(world, i, SurferState.RETURN);
      break;
    }
  }

  // Integrate with land blocking per axis.
  const nx = x + vx * dt;
  if (sampleDepth(world, nx, y) > 0.05) x = nx;
  else vx = 0;
  const ny = y + vy * dt;
  if (sampleDepth(world, x, ny) > 0.05) y = ny;
  else vy = 0;
  x = Math.max(4, Math.min(WORLD_W_M - 4, x));
  y = Math.max(4, Math.min(WORLD_H_M - 4, y));

  world.surferX[i] = x;
  world.surferY[i] = y;
  world.surferVX[i] = vx;
  world.surferVY[i] = vy;
  if (Math.abs(vx) + Math.abs(vy) > 0.05) {
    world.surferHeading[i] = Math.atan2(vy, vx);
  }
}

function sampleDepth(world: WorldState, xM: number, yM: number): number {
  return world.ocean.depthEff[cellIndexAt(world, xM, yM)];
}
