import {
  BASE_PENDULUM_LENGTH_M,
  CONTROL_DT_S,
  GRAVITY_MPS2,
  LINE_LENGTH_M,
  PENDULUM_DAMPING,
  PROBE_HORIZON_S,
  sagFactor,
} from './types.ts';

/**
 * Fast linearized slackline balance predictor.
 *
 * Ignores line coupling (y = 0), uses small-angle linearization sin(theta) ~ theta,
 * upright effective length lEff = 1.0, zero control, and mean wind 0.
 * Rolls the linear ODE at dt = 0.05 s and applies a sagFactor-only compliance
 * adjustment for segment position s. Pure function.
 */

export interface FastPredictState {
  theta: number;
  omega: number;
}

/**
 * Predict peak lateral deviation |x = lEff * sin(theta)| over `horizonS` seconds
 * from `state` at line position `s`, adjusted by `sagFactor(s)` only.
 */
export function predictPeakDeviation(
  state: FastPredictState,
  horizonS: number = PROBE_HORIZON_S,
  s: number = LINE_LENGTH_M * 0.5
): number {
  const lEff = BASE_PENDULUM_LENGTH_M;
  const dt = CONTROL_DT_S;
  const steps = Math.max(0, Math.round(horizonS / dt));

  let theta = state.theta;
  let omega = state.omega;
  let peak = Math.abs(lEff * Math.sin(theta));

  for (let i = 0; i < steps; i++) {
    const alpha = (GRAVITY_MPS2 / lEff) * theta - PENDULUM_DAMPING * omega;
    omega += alpha * dt;
    theta += omega * dt;
    const lat = Math.abs(lEff * Math.sin(theta));
    if (lat > peak) {
      peak = lat;
    }
  }

  // Adjust by sagFactor(s) only: slacker mid-line (sagFactor = 0.5) amplifies predicted sway.
  const sf = sagFactor(s);
  const sagScale = 1 + 0.2 * (1 - sf);
  return peak * sagScale;
}
