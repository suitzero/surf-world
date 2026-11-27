import { CELL_M, NO_INPUT, SurferState } from '../contracts/types.ts';
import type { SurferInput, WorldState } from '../contracts/types.ts';
import { distanceToPeak, findPriorityConflict, sampleOcean, takeoffQuality } from '../sim/world.ts';

const PLANNING_SPEED = 3.2; // m/s, conservative paddle speed
const MAX_HORIZON_S = 40; // ignore crests arriving more than 40 s from now

/**
 * Forecast-based surfer agent (S12). Differs from RuleAgent only in how it
 * picks its paddle target: instead of reacting to the static set-wave ratio,
 * it predicts future crest arrival times and set-envelope amplitudes using
 * the ocean's tau and groupTau fields to plan an intercept.
 */
export class ForecastAgent {
  private targetX = 0;
  private targetY = 0;
  private retargetIn = 0;
  private readonly homeFrac: number;

  constructor(agentIndex: number, agentCount: number) {
    this.homeFrac = agentCount <= 1 ? 0.5 : 0.22 + 0.56 * (agentIndex / (agentCount - 1));
  }

  decide(world: WorldState, i: number): SurferInput {
    const state = world.surferState[i] as SurferState;
    if (
      state === SurferState.TAKEOFF ||
      state === SurferState.RIDE ||
      state === SurferState.WIPEOUT ||
      state === SurferState.KICKOUT
    ) {
      return NO_INPUT;
    }
    const x = world.surferX[i];
    const y = world.surferY[i];
    const dt = world.config.dt;

    // Priority etiquette: identical to RuleAgent.
    const blocked =
      takeoffQuality(world, x, y) &&
      findPriorityConflict(world, i, x, y, world.time, true, distanceToPeak(world, x, y)) >= 0;
    if (blocked) return NO_INPUT;

    this.retargetIn -= dt;
    if (this.retargetIn <= 0) {
      this.retargetIn = 0.8;
      this.pickIntercept(world, x, y);
    }

    // Opportunistic takeoff: a face is under us right now.
    if (takeoffQuality(world, x, y)) {
      return { moveX: 0, moveY: 0.8, takeoff: true };
    }

    const dx = this.targetX - x;
    const dy = this.targetY - y;
    const dist = Math.hypot(dx, dy);
    if (state === SurferState.RETURN) {
      const local = sampleOcean(world, x, y);
      if (local.depth < 5.5) return { moveX: 0, moveY: -1, takeoff: false };
    }
    if (dist < 3) return NO_INPUT; // wait for the crest
    return { moveX: dx / dist, moveY: dy / dist, takeoff: false };
  }

  /**
   * Plan an intercept with a future wave crest using the ocean's travel-time
   * (tau) and set-envelope (groupTau) fields.
   */
  private pickIntercept(world: WorldState, myX: number, myY: number): void {
    const { gridW: w, gridH: h } = world.config;
    const o = world.ocean;
    const now = world.time;
    const T = world.swell.periodS;
    const groupOmega = (2 * Math.PI) / (7 * T);

    let bestScore = Infinity;
    let bestX = w * CELL_M * this.homeFrac;
    let bestY = h * CELL_M * 0.45;

    for (let cy = 4; cy < h - 4; cy += 2) {
      for (let cx = 2; cx < w - 2; cx += 2) {
        const idx = cy * w + cx;
        const d = o.depthEff[idx];
        if (d < 1.2 || d > 10) continue;
        if (o.foam[idx] > 0.5) continue;
        const ampRatio = o.amp[idx] / d;
        if (ampRatio < 0.45 || ampRatio > 1.6) continue;

        const xm = cx * CELL_M + CELL_M / 2;
        const ym = cy * CELL_M + CELL_M / 2;
        const travel = Math.hypot(xm - myX, ym - myY) / PLANNING_SPEED;

        const earliest = now + travel - 0.5;
        const tau = o.tau[idx];
        let tArr = tau + Math.ceil((earliest - tau) / T) * T;
        if (tArr < earliest) tArr += T;
        if (tArr - now > MAX_HORIZON_S) continue;

        const env = 0.6 + 0.4 * Math.cos(groupOmega * (tArr - o.groupTau[idx]));
        const predRatio = ampRatio * env;
        if (predRatio < 0.5 || predRatio > 1.1) continue;

        const wait = Math.max(0, tArr - travel - now);
        const homeDist = Math.abs(xm / (w * CELL_M) - this.homeFrac) * 300;
        const score = Math.abs(predRatio - 0.78) * 100 + homeDist + wait * 2;
        if (score < bestScore) {
          bestScore = score;
          bestX = xm;
          bestY = ym;
        }
      }
    }

    this.targetX = bestX;
    this.targetY = bestY;
  }
}
