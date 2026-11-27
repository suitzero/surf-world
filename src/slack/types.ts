/**
 * Types and physical constants for the 1D slackline balance domain.
 * Self-contained domain types; no imports from surf-specific modules.
 */

/** Total slackline anchor-to-anchor length (m). */
export const LINE_LENGTH_M = 20;

/** Minimum clamped walker position along the line (m). */
export const MIN_POS_M = 0.5;

/** Default initial walker position along the line (m). */
export const INITIAL_POS_M = 1.0;

/** Distance from the far anchor considered finished (m). */
export const FINISH_MARGIN_M = 0.5;

/** Lateral lean angle threshold beyond which the walker falls (rad). */
export const FALL_ANGLE_RAD = 0.6;

/** Gravitational acceleration (m/s^2). */
export const GRAVITY_MPS2 = 9.81;

/** Standing center-of-mass effective pendulum length (m). */
export const BASE_PENDULUM_LENGTH_M = 1.0;

/** Crouch reduction fraction on effective pendulum length (lEff = 1*(1 - 0.3*crouch)). */
export const CROUCH_LENGTH_FACTOR = 0.3;

/** Pendulum angular damping coefficient (1/s). */
export const PENDULUM_DAMPING = 0.15;

/** Maximum lateral control angular acceleration authority (rad/s^2). */
export const U_MAX = 6.0;

/** Base lateral natural frequency of the taut line near anchors (Hz). */
export const LINE_BASE_FREQ_HZ = 0.9;

/** Line lateral damping ratio (dimensionless). */
export const LINE_ZETA = 0.08;

/** Walker-to-line lateral coupling coefficient (dimensionless). */
export const LINE_COUPLE = 0.6;

/** Maximum forward/backward walking speed along the line when upright (m/s). */
export const STEP_SPEED_MPS = 0.45;

/** Crouch walking speed penalty factor (speed scales by 1 - 0.5*crouch). */
export const CROUCH_SPEED_FACTOR = 0.5;

/** Default wind gust amplitude (rad/s^2). */
export const DEFAULT_GUST_AMP = 0.8;

/** Control step duration (s). */
export const CONTROL_DT_S = 0.05;

/** Internal nonlinear integration timestep (s). */
export const INTERNAL_DT_S = 0.002;

/** Number of internal substeps per control step (0.05 / 0.002 = 25). */
export const SUBSTEPS_PER_CONTROL = 25;

/** Number of spatial segments used by the hybrid risk estimator. */
export const SEGMENT_COUNT = 8;

/** Canonical probe initial lean angle (rad). */
export const PROBE_THETA0 = 0.12;

/** Canonical probe initial angular velocity (rad/s). */
export const PROBE_OMEGA0 = 0;

/** Canonical probe horizon (s). */
export const PROBE_HORIZON_S = 3;

/** Number of internal steps in one 3 s accurate probe (3 / 0.002 = 1500). */
export const PROBE_INTERNAL_STEPS = 1500;

/** Control action applied by a walker policy or replay tape. */
export interface SlackAction {
  /** Lateral balance lean effort, clamped to [-1, 1]. */
  lean: number;
  /** Walking pace along the line, clamped to [-1, 1]. */
  stepRate: number;
  /** Crouch depth, clamped to [0, 1]. */
  crouch: number;
}

/** Zero-control action constant. */
export const ZERO_ACTION: Readonly<SlackAction> = {
  lean: 0,
  stepRate: 0,
  crouch: 0,
};

/** Observable instantaneous walker and line state. */
export interface SlackState {
  /** Position along the slackline (m), in [0.5, L]. */
  s: number;
  /** Lateral lean angle from vertical (rad). */
  theta: number;
  /** Lateral lean angular velocity dtheta/dt (rad/s). */
  omega: number;
  /** Contact-point lateral line deflection (m). */
  y: number;
  /** Contact-point lateral line velocity dy/dt (m/s). */
  yVel: number;
  /** Elapsed simulation time (s). */
  time: number;
  /** True once |theta| has exceeded FALL_ANGLE_RAD. */
  fallen: boolean;
  /** True once s >= L - FINISH_MARGIN_M without falling. */
  finished: boolean;
}

/** Initial state and disturbance overrides when creating a slackline teacher. */
export interface SlackTeacherOptions {
  /** Initial position along the line (m), default 1.0. */
  s0?: number;
  /** Initial lateral lean angle (rad), default 0. */
  theta0?: number;
  /** Initial lateral angular velocity (rad/s), default 0. */
  omega0?: number;
  /** Initial line lateral deflection (m), default 0. */
  y0?: number;
  /** Initial line lateral velocity (m/s), default 0. */
  yVel0?: number;
  /** Wind gust amplitude (rad/s^2), default DEFAULT_GUST_AMP. */
  gustAmp?: number;
  /** Multiplicative wind scale (e.g. segment wind exposure), default 1.0. */
  windScale?: number;
}

/** Nonlinear slackline teacher simulator instance. */
export interface SlackTeacher extends SlackState {
  /** Normalized 32-bit seed used to initialize the wind model. */
  readonly seed: number;
  /** PRNG state immediately after wind setup (for deterministic hashing). */
  readonly rngState: number;
  /** Primary wind sinusoid phase drawn from seed. */
  readonly phase1: number;
  /** Secondary wind sinusoid phase drawn from seed. */
  readonly phase2: number;
  /** Wind gust amplitude (rad/s^2). */
  readonly gustAmp: number;
  /** Multiplicative wind scale factor. */
  readonly windScale: number;

  /** Pure deterministic wind acceleration at arbitrary simulation time t (s). */
  windAt(t: number): number;

  /**
   * Advance ONE control step (CONTROL_DT_S = 0.05 s, 25 internal substeps of 0.002 s).
   */
  step(action: SlackAction): void;

  /**
   * Advance a single internal substep of INTERNAL_DT_S (0.002 s) with clamped action.
   * Exposed for fine-grained probe tracking without duplicating physics.
   */
  stepInternal(action: SlackAction): void;
}

/** Clamp a numeric value to [min, max]. */
export function clamp(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

/**
 * Mid-line sag stiffness factor: 1.0 at anchors (s=0, L), 0.5 at mid-line (s=L/2).
 * Effective line stiffness scales linearly with sagFactor(s).
 */
export function sagFactor(s: number): number {
  const sc = clamp(s, 0, LINE_LENGTH_M);
  return 1 - 0.5 * Math.sin((Math.PI * sc) / LINE_LENGTH_M);
}

/**
 * Segment wind exposure multiplier: 0.7 near anchors, up to 1.3 at mid-line.
 */
export function segmentCenter(segIndex: number, segmentCount: number = SEGMENT_COUNT): number {
  const segWidth = LINE_LENGTH_M / segmentCount;
  return (segIndex + 0.5) * segWidth;
}

/**
 * Segment wind scale: windScale(seg) = 0.7 + 0.6 * sin(pi * segCenter / L).
 */
export function segmentWindScale(segIndex: number, segmentCount: number = SEGMENT_COUNT): number {
  const center = segmentCenter(segIndex, segmentCount);
  return 0.7 + 0.6 * Math.sin((Math.PI * center) / LINE_LENGTH_M);
}
