import { clamp } from './types.ts';
import type { SlackAction, SlackState } from './types.ts';

/**
 * PD balance + gated walking policy for the 1D slackline walker.
 *
 * Mirrors the repo's agent-class shape (`decide(state) -> Action`) and reads
 * ONLY the current observable state fields (never future wind or teacher internals).
 */

export interface SlackPolicyOptions {
  kp?: number;
  kd?: number;
  kyVel?: number;
  walkThetaLimit?: number;
  walkStepRate?: number;
  crouchThetaThreshold?: number;
  crouchAmount?: number;
}

export class SlackPolicy {
  private readonly kp: number;
  private readonly kd: number;
  private readonly kyVel: number;
  private readonly walkThetaLimit: number;
  private readonly walkStepRate: number;
  private readonly crouchThetaThreshold: number;
  private readonly crouchAmount: number;

  constructor(opts: SlackPolicyOptions = {}) {
    this.kp = opts.kp ?? 14.0;
    this.kd = opts.kd ?? 3.2;
    this.kyVel = opts.kyVel ?? 0.8;
    this.walkThetaLimit = opts.walkThetaLimit ?? 0.18;
    this.walkStepRate = opts.walkStepRate ?? 0.7;
    this.crouchThetaThreshold = opts.crouchThetaThreshold ?? 0.3;
    this.crouchAmount = opts.crouchAmount ?? 0.7;
  }

  decide(state: Pick<SlackState, 'theta' | 'omega' | 'yVel'>): SlackAction {
    const absTheta = Math.abs(state.theta);
    const lean = clamp(
      -(this.kp * state.theta + this.kd * state.omega) - this.kyVel * state.yVel,
      -1,
      1
    );
    const stepRate = absTheta < this.walkThetaLimit ? this.walkStepRate : 0;
    const crouch = absTheta > this.crouchThetaThreshold ? this.crouchAmount : 0;

    return { lean, stepRate, crouch };
  }
}
