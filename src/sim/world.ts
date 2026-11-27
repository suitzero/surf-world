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
  /** Max seconds after the owner's takeoff that another takeoff counts as the same wave. */
  priorityWindowSec: 1.5,
  /** Max alongshore distance (m) from the owner's takeoff for same-wave grouping. */
  priorityAlongshoreM: 25,
  /** Max cross-shore distance (m) from the owner's takeoff for same-wave grouping. */
  priorityCrossShoreM: 30,
  /** Margin (m) closer to the breaking peak required for a later takeoff to override priority. */
  priorityPeakMarginM: 2,
  /** Face strength at/above which a paddling surfer may take off on the face band. */
  takeoffFaceMin: 0.38,
  /** Max seconds between two takeoffs for them to count as the same wave (in-ride priority). */
  rideGroupWindowSec: 10,
  /** Max alongshore distance (m) between two takeoffs for same-wave grouping (in-ride). */
  rideGroupAlongshoreM: 34,
  /** Max cross-shore distance (m) between two takeoffs for same-wave grouping (in-ride). */
  rideGroupCrossShoreM: 14,
  /** Gap (m) at which a wave owner coming from behind knocks out a drop-in rider ahead. */
  knockGapM: 4,
  /** Gap (m) a non-owner must keep behind a wave owner ahead on the same wave. */
  ownerKeepGapM: 3.5,
  /** Seconds a non-owner may tailgate within knockGapM of the owner before wiping itself out. */
  blockedLimitSec: 1,
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
    surferTakeoffTime: new Float32Array(numSurfers).fill(-Infinity),
    surferTakeoffX: new Float32Array(numSurfers),
    surferTakeoffY: new Float32Array(numSurfers),
    surferPeakDist: new Float32Array(numSurfers).fill(Infinity),
    surferDropins: new Uint16Array(numSurfers),
    surferKnockouts: new Uint16Array(numSurfers),
    surferKnockTime: new Float32Array(numSurfers).fill(-Infinity),
    surferRideScoreStart: new Float32Array(numSurfers),
    surferBlockedTime: new Float32Array(numSurfers),
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
  face: number;
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
    face: o.face[i],
    speed: o.speed[i],
  };
}

/**
 * Is this spot a plausible takeoff right now? Uses only current observable
 * field values: a steep face (ratio in the pre-breaking band), crest close,
 * and not already buried in whitewater — or a strong unbroken wave face,
 * which is the window real surfers actually use (the slope is rideable
 * before the whitewater forms).
 */
export function takeoffQuality(world: WorldState, xM: number, yM: number): boolean {
  const s = sampleOcean(world, xM, yM);
  if (s.depth < TUNING.takeoffMinDepth || s.foam > TUNING.takeoffFoamMax) return false;
  if (
    s.ratio >= TUNING.takeoffRatioMin &&
    s.ratio <= TUNING.takeoffRatioMax &&
    s.cosPhase >= TUNING.takeoffCosPhaseMin
  ) {
    return true;
  }
  // Face-band takeoff: ride the steep face itself as the crest arrives.
  // The ratio floor keeps out the weakest faces, which cannot carry a ride.
  return s.face >= TUNING.takeoffFaceMin && s.cosPhase >= -0.15 && s.ratio >= 0.5;
}

/**
 * Minimum Euclidean distance (m) from (x, y) to the center of any currently
 * breaking cell (ocean.breaking === 1). Returns Infinity if no cell is breaking.
 */
export function distanceToPeak(world: WorldState, x: number, y: number): number {
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
 * Check whether a takeoff by surfer `i` at `(x, y)` and time `t` drops in on
 * another surfer who owns priority on the same wave.
 *
 * Right-of-way rule: the surfer closest to the breaking peak owns the wave.
 * A later takeoff grouped onto the same wave (within priorityWindowSec,
 * priorityAlongshoreM, and priorityCrossShoreM of claimant `j`'s takeoff) while
 * `j` is still riding (or popping up when `includeTakeoff` is true) is a drop-in
 * unless surfer `i` is clearly closer to the peak by more than priorityPeakMarginM.
 *
 * Peak distances are takeoff-time measurements on both sides: `myPeakDist`
 * is surfer `i`'s distance to the peak at `i`'s takeoff moment (the caller
 * passes surferPeakDist[i] when adjudicating a real takeoff, or a live
 * distanceToPeak for a hypothetical one), and each claimant is compared by
 * its recorded surferPeakDist. The breaking mask moves too fast to re-measure
 * fairly at adjudication time.
 *
 * Returns the first qualifying claimant index `j` in increasing index order, or -1.
 */
export function findPriorityConflict(
  world: WorldState,
  i: number,
  x: number,
  y: number,
  t: number,
  includeTakeoff: boolean,
  myPeakDist: number
): number {
  const t32 = Math.fround(t);
  for (let j = 0; j < world.numSurfers; j++) {
    if (j === i) continue;
    const sj = world.surferState[j] as SurferState;
    if (sj !== SurferState.RIDE && !(includeTakeoff && sj === SurferState.TAKEOFF)) {
      continue;
    }
    const dt = t32 - world.surferTakeoffTime[j];
    if (dt < 0 || dt > TUNING.priorityWindowSec) continue;
    if (Math.abs(x - world.surferTakeoffX[j]) > TUNING.priorityAlongshoreM) continue;
    if (Math.abs(y - world.surferTakeoffY[j]) > TUNING.priorityCrossShoreM) continue;
    if (myPeakDist >= world.surferPeakDist[j] - TUNING.priorityPeakMarginM) {
      return j;
    }
  }
  return -1;
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
  enforceRidePriority(world);
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
        world.surferTakeoffTime[i] = world.time;
        world.surferTakeoffX[i] = x;
        world.surferTakeoffY[i] = y;
        world.surferPeakDist[i] = distanceToPeak(world, x, y);
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
        // Adjudicate right-of-way before transitioning to RIDE: dropping in on
        // a surfer who owns the wave wipes out immediately with no catch/score.
        if (
          findPriorityConflict(
            world,
            i,
            world.surferTakeoffX[i],
            world.surferTakeoffY[i],
            world.surferTakeoffTime[i],
            false,
            world.surferPeakDist[i]
          ) >= 0
        ) {
          world.surferDropins[i]++;
          world.surferWipeouts[i]++;
          world.surferStamina[i] = Math.max(0, world.surferStamina[i] - 12);
          setState(world, i, SurferState.WIPEOUT);
          break;
        }
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
        world.surferRideScoreStart[i] = world.surferScore[i];
        world.surferBlockedTime[i] = 0;
        setState(world, i, SurferState.RIDE);
      }
      break;
    }
    case SurferState.RIDE: {
      const rideSpeed = Math.max(2.4, Math.min(6.5, local.speed * 0.8));
      vx = world.rideDirX[i] * rideSpeed + input.moveX * 1.2;
      // Ride the face: track the cross-shore position of the strongest face
      // in this column, so the rider stays on the slope as it translates
      // shoreward instead of sliding along the foam line. Rider input trims
      // position on the face.
      const faceY = faceLineY(world, x, y);
      if (faceY >= 0) {
        vy = clampNum((faceY - y) * 1.5, -3.2, 3.6) + input.moveY * 1.0;
      } else {
        vy = world.rideDirY[i] * rideSpeed + input.moveY * 1.0;
      }
      // Owner protection: a non-owner behind a wave owner on the same wave
      // may not pass through them — bleed off any closing speed that would
      // shrink the gap below ownerKeepGapM within this step.
      const clamped = clampBehindOwner(world, i, vx, vy, dt);
      vx = clamped[0];
      vy = clamped[1];
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
        ahead.face >= 0.18 ||
        (ahead.ratio >= 0.45 && ahead.ratio <= BREAKING_RATIO_MAX && ahead.cosPhase > 0);
      const faceGone =
        world.surferRideTime[i] > 0.8 &&
        local.face < 0.03 &&
        !local.breaking &&
        local.ratio < 0.4;
      if (local.depth < 0.55 || world.surferRideTime[i] > 25) {
        setState(world, i, SurferState.KICKOUT);
      } else if (!faceAhead || faceGone) {
        // Wipe out only when properly buried in whitewater; at the foam
        // edge the rider can still kick out over the back.
        if (local.foam > 0.8) {
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

function clampNum(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Cross-shore position (m) of the strongest wave face in the column at xM,
 * searched within +/-28 m of yM. Returns -1 when no meaningful face is there.
 */
function faceLineY(world: WorldState, xM: number, yM: number): number {
  const { gridW: w, gridH: h } = world.config;
  const cx = Math.max(0, Math.min(w - 1, Math.floor(xM / CELL_M)));
  const cyC = Math.floor(yM / CELL_M);
  let best = -1;
  let bestFace = 0.12;
  for (let cy = Math.max(0, cyC - 7); cy <= Math.min(h - 1, cyC + 7); cy++) {
    // Mild proximity weight: track the rider's own crest, not whichever
    // face in the window happens to be strongest this step.
    const f = world.ocean.face[cy * w + cx] - Math.abs(cy - cyC) * 0.03;
    if (f > bestFace) {
      bestFace = f;
      best = cy;
    }
  }
  return best < 0 ? -1 : best * CELL_M + CELL_M / 2;
}

/**
 * Two riders count as being on the same wave section when their takeoffs
 * were close in time and space (same grouping idea as the takeoff-time
 * priority rule, with a longer window: a peeling wave takes seconds to
 * travel down the line). All enforcement below is additionally gated on
 * the riders' CURRENT proximity, so grouping alone never penalizes anyone.
 */
function sameWave(world: WorldState, i: number, j: number): boolean {
  if (Math.abs(world.surferTakeoffTime[i] - world.surferTakeoffTime[j]) > TUNING.rideGroupWindowSec) {
    return false;
  }
  if (Math.abs(world.surferTakeoffX[i] - world.surferTakeoffX[j]) > TUNING.rideGroupAlongshoreM) {
    return false;
  }
  if (Math.abs(world.surferTakeoffY[i] - world.surferTakeoffY[j]) > TUNING.rideGroupCrossShoreM) {
    return false;
  }
  return true;
}

/** Wave owner of a pair: closest to the peak at takeoff; ties -> earlier takeoff -> lower index. */
function waveOwner(world: WorldState, i: number, j: number): number {
  const di = world.surferPeakDist[i];
  const dj = world.surferPeakDist[j];
  if (di !== dj) return di < dj ? i : j;
  const ti = world.surferTakeoffTime[i];
  const tj = world.surferTakeoffTime[j];
  if (ti !== tj) return ti < tj ? i : j;
  return i < j ? i : j;
}

/**
 * If surfer i is a non-owner riding behind the wave owner on the same wave,
 * remove the radial closing speed that would shrink their gap below
 * ownerKeepGapM this step. Returns the (possibly adjusted) velocity.
 */
function clampBehindOwner(
  world: WorldState,
  i: number,
  vx: number,
  vy: number,
  dt: number
): [number, number] {
  const peel = Math.sign(world.rideDirX[i]);
  if (peel === 0) return [vx, vy];
  const x = world.surferX[i];
  const y = world.surferY[i];
  for (let j = 0; j < world.numSurfers; j++) {
    if (j === i) continue;
    if (world.surferState[j] !== SurferState.RIDE) continue;
    if (!sameWave(world, i, j)) continue;
    if (waveOwner(world, i, j) !== j) continue;
    const dx = world.surferX[j] - x;
    const dy = world.surferY[j] - y;
    if (dx * peel <= 0) continue; // owner is not ahead down the line
    const dist = Math.hypot(dx, dy);
    if (dist >= 6 || dist < 1e-6) continue;
    const ux = dx / dist;
    const uy = dy / dist;
    const closing = (vx - world.surferVX[j]) * ux + (vy - world.surferVY[j]) * uy;
    const allowed = Math.max(0, (dist - TUNING.ownerKeepGapM) / dt) * 0.9;
    if (closing > allowed) {
      const excess = closing - allowed;
      vx -= excess * ux;
      vy -= excess * uy;
    }
  }
  return [vx, vy];
}

/**
 * In-ride right-of-way enforcement (extends the takeoff-time priority rule):
 * - If the wave OWNER closes to within knockGapM behind a non-owner riding
 *   ahead on the same wave, the rider ahead is knocked out: wipeout, the
 *   ride's score is removed and the catch is un-counted (it was a drop-in).
 * - A non-owner stuck within knockGapM behind the owner (the clamp keeps
 *   it from passing through) for more than blockedLimitSec wipes itself out.
 */
function enforceRidePriority(world: WorldState): void {
  const dt = world.config.dt;
  for (let r = 0; r < world.numSurfers; r++) {
    if (world.surferState[r] !== SurferState.RIDE) continue;
    const peel = Math.sign(world.rideDirX[r]);
    if (peel === 0) continue;
    for (let f = 0; f < world.numSurfers; f++) {
      if (f === r) continue;
      if (world.surferState[f] !== SurferState.RIDE) continue;
      if (!sameWave(world, r, f)) continue;
      const dx = world.surferX[f] - world.surferX[r];
      const dy = world.surferY[f] - world.surferY[r];
      if (dx * peel <= 0) continue; // f is not ahead of r down the line
      const dist = Math.hypot(dx, dy);
      if (waveOwner(world, r, f) === r) {
        if (dist <= TUNING.knockGapM) {
          world.surferKnockouts[f]++;
          world.surferKnockTime[f] = world.time;
          world.surferWipeouts[f]++;
          world.surferStamina[f] = Math.max(0, world.surferStamina[f] - 12);
          world.surferScore[f] = world.surferRideScoreStart[f];
          if (world.surferCatches[f] > 0) world.surferCatches[f]--;
          setState(world, f, SurferState.WIPEOUT);
        }
      } else if (dist <= TUNING.knockGapM) {
        world.surferBlockedTime[r] += dt;
        if (world.surferBlockedTime[r] > TUNING.blockedLimitSec) {
          world.surferWipeouts[r]++;
          world.surferStamina[r] = Math.max(0, world.surferStamina[r] - 12);
          setState(world, r, SurferState.WIPEOUT);
          break;
        }
      } else {
        world.surferBlockedTime[r] = Math.max(0, world.surferBlockedTime[r] - 2 * dt);
      }
    }
  }
}
