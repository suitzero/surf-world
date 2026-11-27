import { CELL_M, NO_INPUT, SurferState } from '../contracts/types.ts';
import type { SurferInput, WorldState } from '../contracts/types.ts';
import {
  cellIndexAt,
  distanceToPeak,
  findPriorityConflict,
  findTakeoffSpot,
  sampleOcean,
  takeoffQuality,
  TUNING,
} from '../sim/world.ts';

const REPLAN_INTERVAL_S = 0.5;

interface Plan {
  x: number;
  y: number;
  tArr: number;
  departAt: number;
  score: number;
}

interface CandidateCell {
  idx: number;
  xm: number;
  ym: number;
  ampRatio: number;
  homeDist: number;
  preScore: number;
}

interface TravelEstimate {
  time: number;
  staminaAfter: number;
}

/**
 * Multi-step rollout planning surfer agent. Unlike ForecastAgent (which
 * greedily picks one intercept target), PlannerAgent enumerates discrete plans
 * across (target cell x future wave index x departure timing variant), rolls
 * each plan forward with an analytic mini-model (crest arrival from tau, set
 * envelope from groupTau, two-segment stamina-aware paddle travel time),
 * commits to the lowest-cost plan with hysteresis, and executes its phases:
 * wait-for-departure -> paddle -> hold at spot -> takeoff (via the sim's own
 * takeoffQuality check).
 *
 * Reads ONLY world-provided fields (tau, groupTau, amp, depthEff, foam,
 * swell parameters, and surfer states/positions/stamina) — never future world
 * state or RNG internals.
 */
export class PlannerAgent {
  public readonly candidates: number;
  public readonly horizonS: number;

  private readonly homeFrac: number;
  private plan: Plan | null = null;
  private fallbackTarget: { x: number; y: number } = { x: 0, y: 0 };
  private nextReplanAt = -Infinity;

  constructor(
    agentIndex: number,
    agentCount: number,
    opts?: { candidates?: number; horizonS?: number }
  ) {
    this.candidates = opts?.candidates ?? 24;
    this.horizonS = opts?.horizonS ?? 20;
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

    // Priority etiquette: yield if taking off here would drop in on another surfer.
    if (
      takeoffQuality(world, x, y) &&
      findPriorityConflict(world, i, x, y, world.time, true, distanceToPeak(world, x, y)) >= 0
    ) {
      return NO_INPUT;
    }

    // Opportunistic takeoff: a face is under us right now.
    if (takeoffQuality(world, x, y)) {
      return { moveX: 0, moveY: 0.8, takeoff: true };
    }

    // Head back out to deeper water first when returning from a ride/wipeout.
    if (state === SurferState.RETURN && sampleOcean(world, x, y).depth < 5.5) {
      return { moveX: 0, moveY: -1, takeoff: false };
    }

    // Plan maintenance: replan on schedule, or immediately when an active
    // plan becomes invalid. With no active plan, replans stay on the 0.5 s
    // schedule so a fruitless search cannot rerun every step.
    const planInvalid = this.plan !== null && !this.isPlanValid(world, i);
    if (world.time >= this.nextReplanAt || planInvalid) {
      this.replan(world, i);
      this.nextReplanAt = world.time + REPLAN_INTERVAL_S;
    }

    // Follow active plan (wait -> paddle -> hold) or paddle to fallback spot.
    if (this.plan !== null) {
      if (world.time < this.plan.departAt - 0.1) {
        return NO_INPUT;
      }
      const dx = this.plan.x - x;
      const dy = this.plan.y - y;
      const dist = Math.hypot(dx, dy);
      if (dist < 3) return NO_INPUT;
      return { moveX: dx / dist, moveY: dy / dist, takeoff: false };
    }

    const dx = this.fallbackTarget.x - x;
    const dy = this.fallbackTarget.y - y;
    const dist = Math.hypot(dx, dy);
    if (dist < 3) return NO_INPUT;
    return { moveX: dx / dist, moveY: dy / dist, takeoff: false };
  }

  private isPlanValid(world: WorldState, i: number): boolean {
    if (this.plan === null) return false;
    if (this.plan.tArr < world.time - 0.25) return false;
    const dist = Math.hypot(this.plan.x - world.surferX[i], this.plan.y - world.surferY[i]);
    const { time } = this.travelTime(dist, world.surferStamina[i]);
    return time <= this.plan.tArr - world.time + 0.5;
  }

  /**
   * Two-segment stamina-aware paddle travel model:
   * full speed while stamina >= staminaLowThreshold, reduced speed below it.
   */
  private travelTime(distM: number, staminaStart: number): TravelEstimate {
    if (distM <= 0) {
      return { time: 0, staminaAfter: staminaStart };
    }
    const fullSpeed = TUNING.paddleSpeed;
    const lowSpeed = fullSpeed * TUNING.staminaLowFactor;
    const drain = TUNING.staminaDrainPerSec;
    const threshold = TUNING.staminaLowThreshold;

    if (staminaStart <= threshold) {
      const time = distM / lowSpeed;
      const staminaAfter = Math.max(0, staminaStart - drain * time);
      return { time, staminaAfter };
    }

    const tFull = (staminaStart - threshold) / drain;
    const fullDist = fullSpeed * tFull;
    if (distM <= fullDist) {
      const time = distM / fullSpeed;
      const staminaAfter = staminaStart - drain * time;
      return { time, staminaAfter };
    }

    const rest = distM - fullDist;
    const time = tFull + rest / lowSpeed;
    const staminaAfter = Math.max(0, threshold - drain * (time - tFull));
    return { time, staminaAfter };
  }

  /**
   * Enumerate candidate takeoff cells, roll forward crest arrivals within the
   * planning horizon for 'now' and 'late' departure variants, and commit to
   * the best plan subject to hysteresis.
   */
  private replan(world: WorldState, i: number): void {
    const { gridW: w, gridH: h } = world.config;
    const o = world.ocean;
    const now = world.time;
    const myX = world.surferX[i];
    const myY = world.surferY[i];
    const staminaNow = world.surferStamina[i];
    const T = world.swell.periodS;
    const groupOmega = (2 * Math.PI) / (7 * T);
    const worldWM = w * CELL_M;

    const pool: CandidateCell[] = [];
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
        const homeDist = Math.abs(xm / worldWM - this.homeFrac) * 300;
        const preScore = Math.abs(ampRatio - 0.78) * 100 + homeDist;
        pool.push({ idx, xm, ym, ampRatio, homeDist, preScore });
      }
    }

    pool.sort((a, b) => a.preScore - b.preScore);
    const keepCount = Math.min(pool.length, this.candidates);

    let best: Plan | null = null;
    const horizonEnd = now + this.horizonS;

    for (let c = 0; c < keepCount; c++) {
      const cand = pool[c];
      const { idx, xm, ym, ampRatio, homeDist } = cand;
      const tau = o.tau[idx];
      const groupTau = o.groupTau[idx];
      let t0 = tau + Math.ceil((now - tau) / T) * T;
      if (t0 < now) t0 += T;

      const dist = Math.hypot(xm - myX, ym - myY);
      const nowTrip = this.travelTime(dist, staminaNow);
      let peakDist: number | null = null;

      for (let tArr = t0; tArr <= horizonEnd; tArr += T) {
        const env = 0.6 + 0.4 * Math.cos(groupOmega * (tArr - groupTau));
        const predRatio = ampRatio * env;
        if (predRatio < 0.5 || predRatio > 1.1) continue;

        if (peakDist === null) {
          peakDist = distanceToPeak(world, xm, ym);
        }
        if (findPriorityConflict(world, i, xm, ym, tArr, true, peakDist) >= 0) {
          continue;
        }

        const ratioTerm = Math.abs(predRatio - 0.8) * 100;
        const homeTerm = homeDist * 0.5;

        // Variant (a): depart immediately ('now')
        const waitNowRaw = tArr - (now + nowTrip.time);
        if (waitNowRaw >= -0.5) {
          const waitNow = Math.max(0, waitNowRaw);
          const scoreNow =
            ratioTerm +
            waitNow * 3 +
            (100 - nowTrip.staminaAfter) * 0.3 +
            homeTerm +
            (nowTrip.staminaAfter < 10 ? 50 : 0);
          if (best === null || scoreNow < best.score) {
            best = { x: xm, y: ym, tArr, departAt: now, score: scoreNow };
          }
        }

        // Variant (b): wait offshore to regen stamina, then depart ('late')
        const restWindow = Math.max(0, tArr - nowTrip.time - 0.3 - now);
        const staminaAtDepart = Math.min(
          100,
          staminaNow + TUNING.staminaRegenPerSec * restWindow
        );
        const lateTrip = this.travelTime(dist, staminaAtDepart);
        let departAtLate = tArr - lateTrip.time - 0.3;
        if (departAtLate >= now - 0.05) {
          if (departAtLate < now) departAtLate = now;
          const waitLate = 0.3;
          const scoreLate =
            ratioTerm +
            waitLate * 3 +
            (100 - lateTrip.staminaAfter) * 0.3 +
            homeTerm +
            (lateTrip.staminaAfter < 10 ? 50 : 0);
          if (best === null || scoreLate < best.score) {
            best = { x: xm, y: ym, tArr, departAt: departAtLate, score: scoreLate };
          }
        }
      }
    }

    if (best === null) {
      this.plan = null;
      this.fallbackTarget = findTakeoffSpot(world, this.homeFrac);
      return;
    }

    if (this.plan !== null && this.isPlanValid(world, i)) {
      const currentIdx = cellIndexAt(world, this.plan.x, this.plan.y);
      const d = o.depthEff[currentIdx];
      const ampRatioCur = d > 0 ? o.amp[currentIdx] / d : 0;
      const envCur = 0.6 + 0.4 * Math.cos(groupOmega * (this.plan.tArr - o.groupTau[currentIdx]));
      const predRatioCur = ampRatioCur * envCur;
      const distCur = Math.hypot(this.plan.x - myX, this.plan.y - myY);
      const curTrip = this.travelTime(distCur, staminaNow);
      const waitCur = Math.max(0, this.plan.tArr - (now + curTrip.time));
      const homeDistCur = Math.abs(this.plan.x / worldWM - this.homeFrac) * 300;
      const currentRescore =
        Math.abs(predRatioCur - 0.8) * 100 +
        waitCur * 3 +
        (100 - curTrip.staminaAfter) * 0.3 +
        homeDistCur * 0.5;
      if (best.score >= currentRescore * 0.85) {
        return;
      }
    }

    this.plan = best;
  }
}
