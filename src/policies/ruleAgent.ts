import { CELL_M, NO_INPUT, SurferState } from '../contracts/types.ts';
import type { SurferInput, WorldState } from '../contracts/types.ts';
import { sampleOcean, takeoffQuality } from '../sim/world.ts';

/**
 * Rule-based surfer agent (S06). Reads ONLY the current observable ocean
 * field (phase, breaking mask, foam, ratio) — never future teacher results.
 * Loop: paddle to a predicted peak on the takeoff contour -> wait -> take
 * off when the face arrives -> ride (sim-driven) -> kick out -> paddle back.
 */
export class RuleAgent {
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

    this.retargetIn -= dt;
    if (this.retargetIn <= 0) {
      this.retargetIn = 0.8;
      this.pickTarget(world);
    }

    // Opportunistic takeoff: a face is under us right now.
    if (takeoffQuality(world, x, y)) {
      return { moveX: 0, moveY: 0.8, takeoff: true };
    }

    const dx = this.targetX - x;
    const dy = this.targetY - y;
    const dist = Math.hypot(dx, dy);
    if (state === SurferState.RETURN) {
      // Head back out to deeper water first, then re-join the lineup.
      const local = sampleOcean(world, x, y);
      if (local.depth < 5.5) return { moveX: 0, moveY: -1, takeoff: false };
    }
    if (dist < 3) return NO_INPUT; // at the peak: wait for the wave
    return { moveX: dx / dist, moveY: dy / dist, takeoff: false };
  }

  /**
   * Scan the field for the best takeoff cell: set-wave ratio near the
   * pre-breaking band, low foam, close to this agent's home alongshore
   * fraction and not too far away.
   */
  private pickTarget(world: WorldState): void {
    const { gridW: w, gridH: h } = world.config;
    const o = world.ocean;
    let bestScore = Infinity;
    let bestX = world.config.gridW * CELL_M * this.homeFrac;
    let bestY = world.config.gridH * CELL_M * 0.45;
    for (let cy = 4; cy < h - 4; cy += 2) {
      for (let cx = 2; cx < w - 2; cx += 2) {
        const idx = cy * w + cx;
        const d = o.depthEff[idx];
        if (d < 1.2 || d > 10) continue;
        const setRatio = o.amp[idx] / d;
        if (setRatio < 0.48 || setRatio > 0.9) continue;
        if (o.foam[idx] > 0.5) continue;
        const xm = cx * CELL_M + CELL_M / 2;
        const ym = cy * CELL_M + CELL_M / 2;
        const homeDist = Math.abs(xm / (w * CELL_M) - this.homeFrac) * 300;
        const score = Math.abs(setRatio - 0.68) * 100 + homeDist;
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
