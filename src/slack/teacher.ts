import { SeededRng, normalizeSeed } from '../sim/prng.ts';
import {
  BASE_PENDULUM_LENGTH_M,
  CROUCH_LENGTH_FACTOR,
  CROUCH_SPEED_FACTOR,
  DEFAULT_GUST_AMP,
  FALL_ANGLE_RAD,
  FINISH_MARGIN_M,
  GRAVITY_MPS2,
  INITIAL_POS_M,
  INTERNAL_DT_S,
  LINE_BASE_FREQ_HZ,
  LINE_COUPLE,
  LINE_LENGTH_M,
  LINE_ZETA,
  MIN_POS_M,
  PENDULUM_DAMPING,
  STEP_SPEED_MPS,
  SUBSTEPS_PER_CONTROL,
  U_MAX,
  clamp,
  sagFactor,
} from './types.ts';
import type { SlackAction, SlackTeacher, SlackTeacherOptions } from './types.ts';

/**
 * Nonlinear 1D slackline teacher dynamics.
 *
 * Physics constants (all documented in types.ts):
 * - Line length L = 20 m, walker starts at s = 1.0 m, clamped to [0.5, L].
 * - Effective pendulum length: lEff = 1.0 * (1 - 0.3 * crouch).
 * - Sag stiffness factor: sagFactor(s) = 1 - 0.5 * sin(pi * s / L).
 * - Line natural frequency: wLine = 2 * pi * 0.9 * sqrt(sagFactor(s)),
 *   so stiffness wLine^2 scales directly with sagFactor(s).
 * - Line damping ratio: zeta = 0.08; walker-line coupling: couple = 0.6.
 * - Pendulum damping: cP = 0.15; gravity g = 9.81 m/s^2; max lean authority uMax = 6.0 rad/s^2.
 * - Wind acceleration:
 *     windAcc(t) = gustAmp * (sin(0.9*t + phase1) + 0.6*sin(2.3*t + phase2)) * gustEnvelope(t)
 *   where gustEnvelope(t) is smoothly interpolated seeded value noise at 0.25 Hz
 *   precomputed once at setup so windAt(t) is a pure function of t.
 */

/** Value-noise sampling rate for the gust envelope (Hz -> 1 knot every 4 s). */
const WIND_NOISE_FREQ_HZ = 0.25;

/** Precomputed noise table length (256 knots = 1024 s before periodic wrap). */
const WIND_NOISE_TABLE_SIZE = 256;

/** Minimum gust envelope floor so wind stays active and smooth. */
const WIND_ENVELOPE_MIN = 0.25;

/** Maximum gust envelope ceiling. */
const WIND_ENVELOPE_MAX = 1.0;

/**
 * Create a deterministic nonlinear slackline teacher instance.
 */
export function createTeacher(seed: number, opts: SlackTeacherOptions = {}): SlackTeacher {
  const normSeed = normalizeSeed(seed);
  const rng = new SeededRng(normSeed);

  // Draw deterministic wind parameters at setup only.
  const phase1 = rng.range(0, 2 * Math.PI);
  const phase2 = rng.range(0, 2 * Math.PI);

  const noiseTable = new Float64Array(WIND_NOISE_TABLE_SIZE);
  for (let i = 0; i < WIND_NOISE_TABLE_SIZE; i++) {
    noiseTable[i] = rng.range(WIND_ENVELOPE_MIN, WIND_ENVELOPE_MAX);
  }
  const setupRngState = rng.state;

  const gustAmp = opts.gustAmp ?? DEFAULT_GUST_AMP;
  const windScale = opts.windScale ?? 1.0;

  let s = clamp(opts.s0 ?? INITIAL_POS_M, MIN_POS_M, LINE_LENGTH_M);
  let theta = opts.theta0 ?? 0;
  let omega = opts.omega0 ?? 0;
  let y = opts.y0 ?? 0;
  let yVel = opts.yVel0 ?? 0;
  let internalStepCount = 0;
  let time = 0;
  let fallen = Math.abs(theta) > FALL_ANGLE_RAD;
  let finished = !fallen && s >= LINE_LENGTH_M - FINISH_MARGIN_M;

  function gustEnvelope(t: number): number {
    const u = t * WIND_NOISE_FREQ_HZ;
    const i0 = Math.floor(u);
    const frac = u - i0;
    // Cubic Hermite smoothstep for C1 continuity between 0.25 Hz noise knots.
    const smooth = frac * frac * (3 - 2 * frac);
    const idx0 = ((i0 % WIND_NOISE_TABLE_SIZE) + WIND_NOISE_TABLE_SIZE) % WIND_NOISE_TABLE_SIZE;
    const idx1 = (idx0 + 1) % WIND_NOISE_TABLE_SIZE;
    return noiseTable[idx0] + (noiseTable[idx1] - noiseTable[idx0]) * smooth;
  }

  function windAt(t: number): number {
    const carrier = Math.sin(0.9 * t + phase1) + 0.6 * Math.sin(2.3 * t + phase2);
    return gustAmp * carrier * gustEnvelope(t);
  }

  function stepInternal(action: SlackAction): void {
    const lean = clamp(action.lean, -1, 1);
    const stepRate = clamp(action.stepRate, -1, 1);
    const crouch = clamp(action.crouch, 0, 1);

    const lEff = BASE_PENDULUM_LENGTH_M * (1 - CROUCH_LENGTH_FACTOR * crouch);
    const sf = sagFactor(s);
    const wLine = 2 * Math.PI * LINE_BASE_FREQ_HZ * Math.sqrt(sf);
    const wLine2 = wLine * wLine;

    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);

    // 1. Line lateral acceleration computed from previous state (explicit).
    const yAcc =
      -wLine2 * y -
      2 * LINE_ZETA * wLine * yVel +
      LINE_COUPLE * (lEff * sinTheta - y) * wLine2;

    // 2. Semi-implicit Euler update for the slackline contact point.
    yVel += yAcc * INTERNAL_DT_S;
    y += yVel * INTERNAL_DT_S;

    // 3. Nonlinear inverted pendulum acceleration using that yAcc and pure windAt(t).
    const wind = windScale * windAt(time);
    const alpha =
      (GRAVITY_MPS2 / lEff) * sinTheta -
      (yAcc / lEff) * cosTheta -
      PENDULUM_DAMPING * omega +
      lean * U_MAX +
      wind;

    // 4. Semi-implicit Euler update for the walker pendulum.
    omega += alpha * INTERNAL_DT_S;
    theta += omega * INTERNAL_DT_S;

    // 5. Advance walker along the line while upright and not yet finished.
    if (!fallen && !finished) {
      const dsDt = stepRate * STEP_SPEED_MPS * (1 - CROUCH_SPEED_FACTOR * crouch);
      s = clamp(s + dsDt * INTERNAL_DT_S, MIN_POS_M, LINE_LENGTH_M);
    }

    internalStepCount++;
    time = internalStepCount * INTERNAL_DT_S;

    if (Math.abs(theta) > FALL_ANGLE_RAD) {
      fallen = true;
    }
    if (!fallen && s >= LINE_LENGTH_M - FINISH_MARGIN_M) {
      finished = true;
    }
  }

  function step(action: SlackAction): void {
    for (let i = 0; i < SUBSTEPS_PER_CONTROL; i++) {
      stepInternal(action);
    }
  }

  return {
    seed: normSeed,
    rngState: setupRngState,
    phase1,
    phase2,
    gustAmp,
    windScale,
    get s(): number {
      return s;
    },
    get theta(): number {
      return theta;
    },
    get omega(): number {
      return omega;
    },
    get y(): number {
      return y;
    },
    get yVel(): number {
      return yVel;
    },
    get time(): number {
      return time;
    },
    get fallen(): boolean {
      return fallen;
    },
    get finished(): boolean {
      return finished;
    },
    windAt,
    step,
    stepInternal,
  };
}
