/**
 * Pure, deterministic Mulberry32 PRNG operating on a 32-bit integer state.
 * Designed so that `WorldState.rngState` stores the exact PRNG state and can be
 * replicated bit-for-bit in Python / CUDA reference tests.
 */

export function normalizeSeed(seed: number): number {
  const s = (Math.trunc(seed) >>> 0) || 0x6d2b79f5;
  return s >>> 0;
}

/**
 * Advances a 32-bit state and returns `[nextFloat01, nextRngState]`.
 */
export function nextRandom(rngState: number): [number, number] {
  const nextState = (rngState + 0x6d2b79f5) >>> 0;
  let t = nextState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  const uint32 = (t ^ (t >>> 14)) >>> 0;
  return [uint32 / 4294967296, nextState];
}

/**
 * Helper class wrapping an integer state or operating directly on an object holding `rngState`.
 */
export class SeededRng {
  public state: number;

  constructor(seed: number) {
    this.state = normalizeSeed(seed);
  }

  next(): number {
    const [val, nextState] = nextRandom(this.state);
    this.state = nextState;
    return val;
  }

  range(min: number, max: number): number {
    return min + this.next() * (max - min);
  }

  int(minInclusive: number, maxExclusive: number): number {
    return Math.floor(this.range(minInclusive, maxExclusive));
  }
}
