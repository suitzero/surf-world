import { describe, expect, it } from 'vitest';
import { predictPeakDeviation } from '../src/slack/fast.ts';
import { createSlackHybrid } from '../src/slack/hybrid.ts';
import {
  PROBE_HORIZON_S,
  PROBE_INTERNAL_STEPS,
  PROBE_OMEGA0,
  PROBE_THETA0,
  SEGMENT_COUNT,
  segmentCenter,
} from '../src/slack/types.ts';

describe('slackline fast predictor, versioned hybrid, and governor budget', () => {
  it('shows a genuine difference (> 0.005 m) between fast and accurate probes on mid-line', () => {
    const hybrid = createSlackHybrid(7);
    const midSeg = Math.floor(SEGMENT_COUNT / 2); // segment 4, center 11.25 m
    const midCenter = segmentCenter(midSeg, SEGMENT_COUNT);

    const fastDirect = predictPeakDeviation(
      { theta: PROBE_THETA0, omega: PROBE_OMEGA0 },
      PROBE_HORIZON_S,
      midCenter
    );
    const fastHybrid = hybrid.fastValueFor(midSeg);
    const accurate = hybrid.runAccurateProbe(midSeg);

    expect(fastHybrid).toBeCloseTo(fastDirect, 12);
    expect(accurate.steps).toBe(PROBE_INTERNAL_STEPS);
    expect(Math.abs(accurate.value - fastHybrid)).toBeGreaterThan(0.005);
  });

  it('brings riskAt closer to accurate probe than fast value on >=6 of 8 segments after correction', () => {
    const hybrid = createSlackHybrid(7);

    const fastValues: number[] = [];
    const truthValues: number[] = [];

    for (let seg = 0; seg < SEGMENT_COUNT; seg++) {
      const center = segmentCenter(seg, SEGMENT_COUNT);
      const before = hybrid.riskAt(center);
      expect(before.provenance).toBe('fast');
      fastValues.push(before.value);

      const job = hybrid.startCorrection(seg);
      const probe = hybrid.runAccurateProbe(seg);
      truthValues.push(probe.value);
      expect(hybrid.finishCorrection(job, probe.value)).toBe(true);
    }

    let closerCount = 0;
    for (let seg = 0; seg < SEGMENT_COUNT; seg++) {
      const center = segmentCenter(seg, SEGMENT_COUNT);
      const after = hybrid.riskAt(center);
      expect(after.provenance).toBe('corrected');

      const errFast = Math.abs(fastValues[seg] - truthValues[seg]);
      const errCorrected = Math.abs(after.value - truthValues[seg]);
      if (errCorrected < errFast) {
        closerCount++;
      }
    }

    expect(closerCount).toBeGreaterThanOrEqual(6);
  });

  it('drops stale corrections when version bumps and applies fresh corrections', () => {
    const hybrid = createSlackHybrid(42);
    const seg = 3;
    const center = segmentCenter(seg, SEGMENT_COUNT);

    const initial = hybrid.riskAt(center);
    expect(initial.provenance).toBe('fast');
    expect(initial.version).toBe(1);

    const staleJob = hybrid.startCorrection(seg);
    const probe = hybrid.runAccurateProbe(seg);

    // Scenario changes mid-flight -> version bumps to 2.
    hybrid.bumpVersion();
    expect(hybrid.version).toBe(2);

    const appliedStale = hybrid.finishCorrection(staleJob, probe.value);
    expect(appliedStale).toBe(false);
    expect(hybrid.droppedCount).toBe(1);
    expect(hybrid.appliedCount).toBe(0);

    const afterDrop = hybrid.riskAt(center);
    expect(afterDrop.provenance).toBe('fast');
    expect(afterDrop.value).toBe(initial.value);
    expect(afterDrop.version).toBe(2);

    // Fresh job started at version 2 applies cleanly.
    const freshJob = hybrid.startCorrection(seg);
    const appliedFresh = hybrid.finishCorrection(freshJob, probe.value);
    expect(appliedFresh).toBe(true);
    expect(hybrid.appliedCount).toBe(1);
    expect(hybrid.droppedCount).toBe(1);

    const afterApply = hybrid.riskAt(center);
    expect(afterApply.provenance).toBe('corrected');
    expect(afterApply.value).toBe(probe.value);
    expect(afterApply.version).toBe(2);
  });

  it('respects governor compute budget over 600 ticks and scales applied corrections with budget', () => {
    const lowBudget = createSlackHybrid(7);
    const highBudget = createSlackHybrid(7);
    const lowBudgetRepeat = createSlackHybrid(7);

    const ticks = 600;
    for (let t = 0; t < ticks; t++) {
      lowBudget.advanceTick(0.2);
      highBudget.advanceTick(2.0);
      lowBudgetRepeat.advanceTick(0.2);
    }

    // 600 * 0.2 * 1.05 = 126.0 + small epsilon -> <= 126.1
    expect(lowBudget.totalSpendMs).toBeLessThanOrEqual(126.1);
    expect(lowBudget.appliedCount).toBeGreaterThan(0);
    expect(highBudget.appliedCount).toBeGreaterThan(lowBudget.appliedCount);

    // Determinism check for governor driver.
    expect(lowBudgetRepeat.totalSpendMs).toBe(lowBudget.totalSpendMs);
    expect(lowBudgetRepeat.appliedCount).toBe(lowBudget.appliedCount);
  });
});
