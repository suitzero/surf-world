/**
 * Pure, deterministic compute-budget governor (S16) for scheduling teacher
 * stepping and tile correction overlays within a per-tick millisecond budget.
 *
 * No imports from hybrid.ts, no side effects, no mutation of inputs, and no
 * wall-clock or PRNG calls.
 */

export interface TileStateInput {
  id: number;
  correctedAt: number;
  queries: number;
  eligible: boolean;
}

export interface CostTable {
  teacherStepMs: number;
  tileApplyMs: number;
}

export interface GovernorInput {
  budgetMs: number;
  bankedMs: number;
  now: number;
  tiles: readonly TileStateInput[];
  costs: CostTable;
  maxTeacherStepsPerTick?: number;
}

export interface TickPlan {
  teacherSteps: number;
  tilesToCorrect: number[];
  spendMs: number;
  bankedMs: number;
}

interface ScoredTile {
  id: number;
  score: number;
}

export function planTick(input: GovernorInput): TickPlan {
  const { budgetMs, bankedMs, now, tiles, costs, maxTeacherStepsPerTick } = input;
  const available = budgetMs + bankedMs;

  if (available <= 0) {
    return {
      teacherSteps: 0,
      tilesToCorrect: [],
      spendMs: 0,
      bankedMs: available,
    };
  }

  const eligibleTiles: ScoredTile[] = [];
  for (let i = 0; i < tiles.length; i++) {
    const t = tiles[i];
    if (!t.eligible) continue;
    const age = Number.isFinite(t.correctedAt) ? now - t.correctedAt : 1e9 + (now - 0);
    const score = age * (1 + t.queries);
    eligibleTiles.push({ id: t.id, score });
  }

  eligibleTiles.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.id - b.id;
  });

  const { teacherStepMs, tileApplyMs } = costs;
  let remaining = available;
  const tilesToCorrect: number[] = [];
  let nextIdx = 0;

  // 1. Guarantee rotation/starvation freedom: reserve the single highest-priority
  //    eligible tile first when we can afford at least one tile apply.
  if (eligibleTiles.length > 0 && tileApplyMs >= 0 && remaining >= tileApplyMs) {
    tilesToCorrect.push(eligibleTiles[0].id);
    remaining -= tileApplyMs;
    nextIdx = 1;
  }

  // 2. Allocate remaining budget to teacher steps, respecting optional cap.
  let teacherSteps = 0;
  if (teacherStepMs > 0 && remaining >= teacherStepMs) {
    let steps = Math.floor(remaining / teacherStepMs);
    if (maxTeacherStepsPerTick !== undefined) {
      const cap = Math.max(0, Math.floor(maxTeacherStepsPerTick));
      if (steps > cap) steps = cap;
    }
    teacherSteps = steps;
    remaining -= teacherSteps * teacherStepMs;
  }

  // 3. Spend leftover budget on additional eligible tiles in priority order.
  while (nextIdx < eligibleTiles.length && tileApplyMs >= 0 && remaining >= tileApplyMs) {
    tilesToCorrect.push(eligibleTiles[nextIdx].id);
    remaining -= tileApplyMs;
    nextIdx++;
  }

  const spendMs = teacherSteps * teacherStepMs + tilesToCorrect.length * tileApplyMs;
  const rawBanked = available - spendMs;
  const nextBanked = rawBanked < 0 && rawBanked > -1e-12 ? 0 : rawBanked;

  return {
    teacherSteps,
    tilesToCorrect,
    spendMs,
    bankedMs: nextBanked,
  };
}
