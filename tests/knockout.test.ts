import { describe, expect, it } from 'vitest';
import { CELL_M, NO_INPUT, SurferState } from '../src/contracts/types.ts';
import type { SwellParams, WorldState } from '../src/contracts/types.ts';
import { peelSignAt } from '../src/sim/ocean.ts';
import { createWorld, sampleOcean, stepWorld } from '../src/sim/world.ts';

const SWELL: SwellParams = { heightM: 1.8, periodS: 9, directionDeg: 8 };

/**
 * In-ride right-of-way enforcement (F1, extends the P4 takeoff rule):
 * scripted two-rider scenarios on a real forming face. Riders are placed
 * directly into RIDE on the same wave section — the rear rider starts
 * ~5.7 m behind the front rider, upwave on the same face line, and the
 * face-tracking ride converges them onto one line, which is exactly the
 * situation the rule exists for. Placement is retried on fresh faces
 * (same scan-forward pattern as the priority tests) because any single
 * face may die before the gap closes.
 */
function placeRider(world: WorldState, i: number, x: number, y: number, peakDist: number): void {
  const cx = Math.floor(x / CELL_M);
  const cy = Math.floor(y / CELL_M);
  const peel = peelSignAt(world.ocean, world.config.gridW, cx, cy);
  const dirX = peel * 0.9;
  const dirY = 0.45;
  const len = Math.hypot(dirX, dirY) || 1;
  world.surferX[i] = x;
  world.surferY[i] = y;
  world.surferVX[i] = 0;
  world.surferVY[i] = 0;
  world.surferState[i] = SurferState.RIDE;
  world.surferStateTime[i] = 0;
  world.surferRideTime[i] = 0;
  world.rideDirX[i] = dirX / len;
  world.rideDirY[i] = dirY / len;
  world.surferTakeoffTime[i] = world.time;
  world.surferTakeoffX[i] = x;
  world.surferTakeoffY[i] = y;
  world.surferPeakDist[i] = peakDist;
  world.surferRideScoreStart[i] = world.surferScore[i];
  world.surferBlockedTime[i] = 0;
  world.surferStamina[i] = 100;
  world.surferInsideTime[i] = 0;
}

function findFaceCell(world: WorldState): { x: number; y: number; peel: number } | null {
  const { gridW: w, gridH: h } = world.config;
  const o = world.ocean;
  for (let cy = 6; cy < h - 6; cy++) {
    for (let cx = 6; cx < w - 6; cx++) {
      const idx = cy * w + cx;
      if (o.face[idx] < 0.55) continue;
      const d = o.depthEff[idx];
      if (d < 1.6 || d > 3.2) continue;
      const peel = peelSignAt(o, w, cx, cy);
      if (peel === 0) continue;
      return { x: cx * CELL_M + CELL_M / 2, y: cy * CELL_M + CELL_M / 2, peel };
    }
  }
  return null;
}

/** Advance the world (both scripted riders idle) until a usable face cell exists. */
function scanForFace(world: WorldState): { x: number; y: number; peel: number } | null {
  for (let s = 0; s < 900; s++) {
    const cell = findFaceCell(world);
    if (cell) return cell;
    stepWorld(world, [NO_INPUT, NO_INPUT]);
  }
  return null;
}

describe('Drop-in knockout (in-ride priority enforcement)', () => {
  it(
    'owner behind closes in and knocks out the drop-in rider ahead: wipeout, score zeroed, catch un-counted',
    () => {
      const world = createWorld({ bathymetry: 'sandbar', swell: SWELL, aiCount: 1 }, 7);
      for (let s = 0; s < 600; s++) stepWorld(world, [NO_INPUT, NO_INPUT]);
      const F = 1; // drop-in, ahead (farther from the peak at takeoff)
      const R = 0; // wave owner, behind
      let knocked = false;
      for (let attempt = 0; attempt < 10 && !knocked; attempt++) {
        const cell = scanForFace(world);
        if (!cell) break;
        const fx = cell.x;
        const fy = cell.y;
        const rx = fx - cell.peel * 3.5;
        const ry = fy - 4.5;
        if (sampleOcean(world, rx, ry).speed < sampleOcean(world, fx, fy).speed) continue;
        placeRider(world, F, fx, fy, 15);
        placeRider(world, R, rx, ry, 2);
        world.surferCatches[F] += 1; // the ride counted when it took off
        world.surferScore[F] = world.surferRideScoreStart[F] + 30; // accrued ride score
        const catchesBefore = world.surferCatches[F];
        for (let s = 0; s < 300; s++) {
          stepWorld(world, [NO_INPUT, NO_INPUT]);
          if (world.surferKnockouts[F] >= 1) {
            knocked = true;
            expect(world.surferState[F]).toBe(SurferState.WIPEOUT);
            expect(world.surferCatches[F]).toBe(catchesBefore - 1);
            expect(world.surferScore[F]).toBe(world.surferRideScoreStart[F]);
            expect(world.surferKnockTime[F]).toBeGreaterThan(0);
            break;
          }
          if (world.surferState[F] !== SurferState.RIDE || world.surferState[R] !== SurferState.RIDE) {
            break;
          }
        }
      }
      expect(knocked).toBe(true);
    },
    180000
  );

  it(
    'owner ahead: the non-owner behind is clamped, never passes through, and the owner is never knocked out',
    () => {
      const world = createWorld({ bathymetry: 'sandbar', swell: SWELL, aiCount: 1 }, 7);
      for (let s = 0; s < 600; s++) stepWorld(world, [NO_INPUT, NO_INPUT]);
      const F = 0; // wave owner, ahead
      const R = 1; // non-owner, behind
      let observed = false;
      let minDist = Infinity;
      let passedThrough = false;
      for (let attempt = 0; attempt < 10 && !observed; attempt++) {
        const cell = scanForFace(world);
        if (!cell) break;
        const fx = cell.x;
        const fy = cell.y;
        const rx = fx - cell.peel * 3.5;
        const ry = fy - 4.5;
        if (sampleOcean(world, rx, ry).speed < sampleOcean(world, fx, fy).speed) continue;
        placeRider(world, F, fx, fy, 2);
        placeRider(world, R, rx, ry, 15);
        minDist = Infinity;
        let bothRidingSteps = 0;
        for (let s = 0; s < 300; s++) {
          stepWorld(world, [NO_INPUT, NO_INPUT]);
          if (world.surferState[F] !== SurferState.RIDE || world.surferState[R] !== SurferState.RIDE) {
            break;
          }
          bothRidingSteps++;
          const dx = world.surferX[F] - world.surferX[R];
          const dy = world.surferY[F] - world.surferY[R];
          const dist = Math.hypot(dx, dy);
          if (dist < minDist) minDist = dist;
          if (dx * cell.peel <= 0) passedThrough = true;
        }
        // Observed = the rear rider genuinely closed into the clamp zone.
        if (bothRidingSteps >= 15 && minDist < 5) observed = true;
      }
      console.log(`[F1 owner-ahead] minDist=${minDist.toFixed(2)} passedThrough=${passedThrough}`);
      expect(observed).toBe(true);
      expect(passedThrough).toBe(false);
      expect(world.surferKnockouts[F]).toBe(0);
      expect(minDist).toBeGreaterThanOrEqual(2.8);
    },
    180000
  );

  it('a lone rider is never knocked out', () => {
    const world = createWorld({ bathymetry: 'sandbar', swell: SWELL, aiCount: 0 }, 7);
    for (let s = 0; s < 600; s++) stepWorld(world, [NO_INPUT]);
    const cell = scanForFace(world as WorldState);
    expect(cell).not.toBeNull();
    if (!cell) return;
    placeRider(world, 0, cell.x, cell.y, 2);
    for (let s = 0; s < 300; s++) {
      stepWorld(world, [NO_INPUT]);
      if (world.surferState[0] !== SurferState.RIDE) break;
    }
    expect(world.surferKnockouts[0]).toBe(0);
  });
});
