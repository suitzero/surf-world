import { planTick } from '../sim/governor.ts';
import type { CostTable, TickPlan, TileStateInput } from '../sim/governor.ts';
import { normalizeSeed } from '../sim/prng.ts';
import { predictPeakDeviation } from './fast.ts';
import { createTeacher } from './teacher.ts';
import {
  BASE_PENDULUM_LENGTH_M,
  DEFAULT_GUST_AMP,
  LINE_LENGTH_M,
  PROBE_HORIZON_S,
  PROBE_INTERNAL_STEPS,
  PROBE_OMEGA0,
  PROBE_THETA0,
  SEGMENT_COUNT,
  ZERO_ACTION,
  clamp,
  segmentCenter,
  segmentWindScale,
} from './types.ts';

/**
 * Versioned hybrid fast + nonlinear-teacher risk estimator for the 1D slackline.
 *
 * Mirrors the versioned tile-correction semantics of the surf HybridOcean
 * (startCorrection / finishCorrection, version check, drop+count stale) and
 * uses the shared pure compute-budget governor `planTick`.
 */

/** Sentinel correctedAt value representing a segment that has never been corrected. */
export const NEVER_CORRECTED_AT = -1e9;

/** Default abstract governor cost table (ms). */
export const SLACK_GOVERNOR_COSTS: Readonly<CostTable> = {
  teacherStepMs: 0.01,
  tileApplyMs: 0.05,
};

export type SlackProvenance = 'fast' | 'corrected';

export interface SlackHybridOptions {
  /** Wind gust amplitude passed to nonlinear segment probes. */
  gustAmp?: number;
}

export interface SlackProbeResult {
  /** Predicted peak lateral deviation |y + lEff * sin(theta)| over 3 s (m). */
  value: number;
  /** Internal teacher integration steps consumed (= 1500 for 3 s at dt=0.002). */
  steps: number;
}

export interface SlackCorrectionJob {
  readonly segIndex: number;
  readonly version: number;
  readonly fastValue: number;
}

export interface SlackRiskQuery {
  readonly value: number;
  readonly provenance: SlackProvenance;
  readonly version: number;
  readonly segIndex: number;
}

export interface SlackSegmentSnapshot {
  readonly id: number;
  readonly centerM: number;
  readonly fastValue: number;
  readonly correctedValue: number | null;
  readonly correctedVersion: number;
  readonly correctedAt: number;
  readonly queries: number;
}

interface InternalSegment {
  id: number;
  centerM: number;
  fastValue: number;
  correctedValue: number | null;
  correctedVersion: number;
  correctedAt: number;
  queries: number;
}

export interface SlackHybrid {
  readonly seed: number;
  readonly segmentCount: number;
  readonly version: number;
  readonly tickCount: number;
  readonly totalSpendMs: number;
  readonly bankedMs: number;
  readonly appliedCount: number;
  readonly droppedCount: number;

  /** Increment scenario version (e.g. when wind/walker params change), invalidating in-flight/old corrections. */
  bumpVersion(): number;
  /** Instant linearized fast prediction for a segment index, adjusted by sagFactor only. */
  fastValueFor(segIndex: number): number;
  /** Run deterministic nonlinear teacher probe over 3 s (1500 internal steps) at segment `segIndex`. */
  runAccurateProbe(segIndex: number): SlackProbeResult;
  /** Snapshot current version and fast baseline for a segment correction job. */
  startCorrection(segIndex: number): SlackCorrectionJob;
  /** Apply accurate value only if job.version matches current version; else drop and increment droppedCount. */
  finishCorrection(job: SlackCorrectionJob, accurateValue: number): boolean;
  /** Query predicted peak lateral deviation at line position s (m) and increment segment query counter. */
  riskAt(s: number): SlackRiskQuery;
  /** Map line position s (m) to segment index in [0, segmentCount - 1]. */
  segmentIndexAt(s: number): number;
  /** Advance one governor-budgeted correction tick. */
  advanceTick(budgetMs: number): TickPlan;
  /** Read-only snapshot of a segment's current state. */
  getSegment(segIndex: number): SlackSegmentSnapshot;
}

export function createSlackHybrid(seed: number, opts: SlackHybridOptions = {}): SlackHybrid {
  const normSeed = normalizeSeed(seed);
  const gustAmp = opts.gustAmp ?? DEFAULT_GUST_AMP;

  let version = 1;
  let tickCount = 0;
  let bankedMs = 0;
  let teacherStepPool = 0;
  let totalSpendMs = 0;
  let appliedCount = 0;
  let droppedCount = 0;

  const segments: InternalSegment[] = [];
  for (let i = 0; i < SEGMENT_COUNT; i++) {
    const centerM = segmentCenter(i, SEGMENT_COUNT);
    const fastValue = predictPeakDeviation(
      { theta: PROBE_THETA0, omega: PROBE_OMEGA0 },
      PROBE_HORIZON_S,
      centerM
    );
    segments.push({
      id: i,
      centerM,
      fastValue,
      correctedValue: null,
      correctedVersion: 0,
      correctedAt: NEVER_CORRECTED_AT,
      queries: 0,
    });
  }

  function segmentIndexAt(s: number): number {
    const raw = Math.floor((s / LINE_LENGTH_M) * SEGMENT_COUNT);
    return clamp(raw, 0, SEGMENT_COUNT - 1);
  }

  function fastValueFor(segIndex: number): number {
    const idx = clamp(Math.trunc(segIndex), 0, SEGMENT_COUNT - 1);
    return segments[idx].fastValue;
  }

  function runAccurateProbe(segIndex: number): SlackProbeResult {
    const idx = clamp(Math.trunc(segIndex), 0, SEGMENT_COUNT - 1);
    const centerM = segments[idx].centerM;
    const wScale = segmentWindScale(idx, SEGMENT_COUNT);

    // Always run over the deterministic wind window starting at t = 0.
    const probe = createTeacher(normSeed, {
      s0: centerM,
      theta0: PROBE_THETA0,
      omega0: PROBE_OMEGA0,
      y0: 0,
      yVel0: 0,
      gustAmp,
      windScale: wScale,
    });

    const lEff = BASE_PENDULUM_LENGTH_M;
    let peak = Math.abs(probe.y + lEff * Math.sin(probe.theta));

    for (let step = 0; step < PROBE_INTERNAL_STEPS; step++) {
      probe.stepInternal(ZERO_ACTION);
      const lat = Math.abs(probe.y + lEff * Math.sin(probe.theta));
      if (lat > peak) {
        peak = lat;
      }
    }

    return {
      value: peak,
      steps: PROBE_INTERNAL_STEPS,
    };
  }

  function bumpVersion(): number {
    version++;
    for (let i = 0; i < segments.length; i++) {
      segments[i].correctedAt = NEVER_CORRECTED_AT;
    }
    return version;
  }

  function startCorrection(segIndex: number): SlackCorrectionJob {
    const idx = clamp(Math.trunc(segIndex), 0, SEGMENT_COUNT - 1);
    return {
      segIndex: idx,
      version,
      fastValue: segments[idx].fastValue,
    };
  }

  function finishCorrection(job: SlackCorrectionJob, accurateValue: number): boolean {
    if (job.version !== version) {
      droppedCount++;
      return false;
    }
    const seg = segments[job.segIndex];
    seg.correctedValue = accurateValue;
    seg.correctedVersion = version;
    seg.correctedAt = tickCount;
    appliedCount++;
    return true;
  }

  function riskAt(s: number): SlackRiskQuery {
    const idx = segmentIndexAt(s);
    const seg = segments[idx];
    seg.queries++;
    if (seg.correctedValue !== null && seg.correctedVersion === version) {
      return {
        value: seg.correctedValue,
        provenance: 'corrected',
        version,
        segIndex: idx,
      };
    }
    return {
      value: seg.fastValue,
      provenance: 'fast',
      version,
      segIndex: idx,
    };
  }

  function advanceTick(budgetMs: number): TickPlan {
    const tileInputs: TileStateInput[] = segments.map((seg) => ({
      id: seg.id,
      correctedAt: seg.correctedAt,
      queries: seg.queries,
      eligible: true,
    }));

    const plan = planTick({
      budgetMs,
      bankedMs,
      now: tickCount,
      tiles: tileInputs,
      costs: SLACK_GOVERNOR_COSTS,
    });

    bankedMs = plan.bankedMs;
    totalSpendMs += plan.spendMs;
    teacherStepPool += plan.teacherSteps;

    for (let i = 0; i < plan.tilesToCorrect.length; i++) {
      if (teacherStepPool < PROBE_INTERNAL_STEPS) {
        break;
      }
      const segId = plan.tilesToCorrect[i];
      const job = startCorrection(segId);
      const probe = runAccurateProbe(segId);
      teacherStepPool -= probe.steps;
      finishCorrection(job, probe.value);
    }

    tickCount++;
    return plan;
  }

  function getSegment(segIndex: number): SlackSegmentSnapshot {
    const idx = clamp(Math.trunc(segIndex), 0, SEGMENT_COUNT - 1);
    return { ...segments[idx] };
  }

  return {
    seed: normSeed,
    segmentCount: SEGMENT_COUNT,
    get version(): number {
      return version;
    },
    get tickCount(): number {
      return tickCount;
    },
    get totalSpendMs(): number {
      return totalSpendMs;
    },
    get bankedMs(): number {
      return bankedMs;
    },
    get appliedCount(): number {
      return appliedCount;
    },
    get droppedCount(): number {
      return droppedCount;
    },
    bumpVersion,
    fastValueFor,
    runAccurateProbe,
    startCorrection,
    finishCorrection,
    riskAt,
    segmentIndexAt,
    advanceTick,
    getSegment,
  };
}
