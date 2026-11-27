import type { SlackAction, SlackTeacherOptions } from './types.ts';
import { createSlackWorld, stepSlackWorld } from './world.ts';
import type { SlackWorld } from './world.ts';

/**
 * Deterministic policy-independent action recorder and replay runner
 * for the 1D slackline domain.
 */

export interface SlackRecording {
  readonly seed: number;
  readonly options?: SlackTeacherOptions;
  readonly actions: readonly SlackAction[];
}

export class SlackRecorder {
  private readonly seed: number;
  private readonly options?: SlackTeacherOptions;
  private readonly actions: SlackAction[];

  constructor(worldOrSeed: SlackWorld | number, options?: SlackTeacherOptions) {
    if (typeof worldOrSeed === 'number') {
      this.seed = worldOrSeed;
      this.options = options ? { ...options } : undefined;
    } else {
      this.seed = worldOrSeed.seed;
      this.options = options ? { ...options } : { ...worldOrSeed.options };
    }
    this.actions = [];
  }

  record(action: SlackAction): void {
    this.actions.push({
      lean: action.lean,
      stepRate: action.stepRate,
      crouch: action.crouch,
    });
  }

  get recording(): SlackRecording {
    return {
      seed: this.seed,
      options: this.options ? { ...this.options } : undefined,
      actions: this.actions.map((a) => ({ ...a })),
    };
  }
}

/** Rebuild a slackline world from a recording and replay every recorded action. */
export function replaySlackRecording(recording: SlackRecording): SlackWorld {
  const world = createSlackWorld(recording.seed, recording.options);
  for (let i = 0; i < recording.actions.length; i++) {
    stepSlackWorld(world, recording.actions[i]);
  }
  return world;
}
