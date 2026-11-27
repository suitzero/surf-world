import { CELL_M, NO_INPUT, SurferState } from '../contracts/types.ts';
import type { SurferInput, WorldState } from '../contracts/types.ts';
import { distanceToPeak, findPriorityConflict, sampleOcean, takeoffQuality } from '../sim/world.ts';

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
  /** Energy-management state: resting regenerates stamina (see decide). */
  private resting = false;
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

    // Priority etiquette: yield the wave completely if taking off here would
    // drop in on another surfer (note: paddling shoreward with moveY > 0.55
    // also triggers takeoff, so returning NO_INPUT avoids any accidental drop-in).
    const blocked =
      takeoffQuality(world, x, y) &&
      findPriorityConflict(world, i, x, y, world.time, true, distanceToPeak(world, x, y)) >= 0;
    // Yield by drifting offshore out of the wash, not by sitting in it.
    if (blocked) return { moveX: 0, moveY: -0.6, takeoff: false };

    this.retargetIn -= dt;
    if (this.retargetIn <= 0) {
      this.retargetIn = 0.8;
      this.pickTarget(world);
      // Face chase: if a real face is forming within reach, intercept it
      // (waiting just upwave of it) instead of only waiting on the static
      // contour. The static contour says where waves CAN break; the face
      // field says where one is breaking NOW.
      const faceTarget = this.scanFace(world, x, y);
      if (faceTarget) {
        this.targetX = faceTarget.x;
        this.targetY = faceTarget.y;
      }
    }

    // Opportunistic takeoff: a face is under us right now.
    if (takeoffQuality(world, x, y)) {
      return { moveX: 0, moveY: 0.8, takeoff: true };
    }

    // Energy management: below the low-stamina threshold paddling runs at
    // half speed, so an agent that never rests is slow everywhere (measured:
    // ~93% of the time below threshold without this). Rest in place to
    // regenerate when tired — unless the wash makes sitting still dangerous
    // or a face is about to arrive here.
    const stamina = world.surferStamina[i];
    if (this.resting && stamina > 55) this.resting = false;
    if (!this.resting && stamina < 16) this.resting = true;
    if (this.resting) {
      const localR = sampleOcean(world, x, y);
      if (!localR.breaking && localR.foam <= 0.5) {
        const thetaR = (world.swell.directionDeg * Math.PI) / 180;
        const probe = sampleOcean(
          world,
          x - Math.sin(thetaR) * 10,
          y - Math.cos(thetaR) * 10
        );
        if (probe.face < 0.5 && probe.ratio < 0.6) {
          return NO_INPUT;
        }
      }
    }

    const dx = this.targetX - x;
    const dy = this.targetY - y;
    const dist = Math.hypot(dx, dy);
    if (state === SurferState.RETURN) {
      // Paddle straight out only while actually in the wash; once clear of
      // the foam, head for the lineup instead of touring out to deep water
      // (on a shallow reef platform that tour eats the whole cycle).
      const local = sampleOcean(world, x, y);
      if (local.depth < 5.5 && local.foam > 0.35) return { moveX: 0, moveY: -1, takeoff: false };
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
        if (setRatio < 0.5 || setRatio > 0.95) continue;
        if (o.foam[idx] > 0.5) continue;
        const xm = cx * CELL_M + CELL_M / 2;
        const ym = cy * CELL_M + CELL_M / 2;
        const homeDist = Math.abs(xm / (w * CELL_M) - this.homeFrac) * 200;
        const score = Math.abs(setRatio - 0.7) * 100 + homeDist;
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

  /**
   * Find the strongest current wave-face cell within ~32 m of (x, y) on
   * water deep enough to ride. The face band is where takeoffs happen now,
   * so head for it rather than for where waves broke a minute ago.
   */
  private scanFace(world: WorldState, x: number, y: number): { x: number; y: number } | null {
    const { gridW: w, gridH: h } = world.config;
    const o = world.ocean;
    const cxC = Math.floor(x / CELL_M);
    const cyC = Math.floor(y / CELL_M);
    const R = 8; // cells = 32 m
    // Intercept point: wait ~8 m upwave (offshore) of the face cell, so the
    // face arrives at the agent — waiting ON the face cell means waiting in
    // the foam once the crest passes.
    const theta = (world.swell.directionDeg * Math.PI) / 180;
    const backX = Math.sin(theta) * 8;
    const backY = Math.cos(theta) * 8;
    let best: { x: number; y: number } | null = null;
    let bestScore = 0.5;
    for (let cy = Math.max(2, cyC - R); cy <= Math.min(h - 3, cyC + R); cy++) {
      for (let cx = Math.max(2, cxC - R); cx <= Math.min(w - 3, cxC + R); cx++) {
        const idx = cy * w + cx;
        const f = o.face[idx];
        if (f < 0.55) continue;
        if (o.depthEff[idx] < 1.2) continue;
        const xm = cx * CELL_M + CELL_M / 2;
        const ym = cy * CELL_M + CELL_M / 2;
        // Stay near home: four agents chasing the same strongest face just
        // congest one section and yield/wash each other out.
        const homePenalty = Math.abs(xm / (w * CELL_M) - this.homeFrac) * 0.6;
        const score = f - Math.hypot(xm - x, ym - y) * 0.004 - homePenalty;
        if (score > bestScore) {
          bestScore = score;
          best = { x: xm - backX, y: ym - backY };
        }
      }
    }
    return best;
  }
}
