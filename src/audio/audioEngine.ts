/**
 * Procedural 8-bit Web Audio engine for AI Surf World.
 *
 * - Pure Web Audio API synthesis (square, triangle, LCG white-noise buffer).
 * - BGM: original generated I–I–vi–vi–IV–IV–V–V swell arpeggio loop at 108 BPM.
 * - Lookahead scheduler (35 ms interval, 0.12 s lookahead window).
 * - Pure note/SFX data and helpers exported for headless unit tests.
 */

export const MUTE_STORAGE_KEY = 'surf-world-muted';
export const BGM_TEMPO_BPM = 108;
export const BGM_STEP_DURATION_SEC = 60 / BGM_TEMPO_BPM / 2; // Eighth-note step (~0.2778s)
export const BGM_STEPS_PER_BAR = 8;
export const BGM_BAR_COUNT = 8;
export const BGM_TOTAL_STEPS = BGM_STEPS_PER_BAR * BGM_BAR_COUNT; // 64 steps
export const PADDLE_MIN_INTERVAL_MS = 240;

export type WaveformType = 'square' | 'triangle' | 'noise';

export type SfxName =
  | 'paddle'
  | 'takeoff'
  | 'ride'
  | 'wipeout'
  | 'kickout'
  | 'click';

export interface BarChordSpec {
  readonly chord: string;
  readonly rootMidi: number;
  readonly tones: readonly [number, number, number];
}

export interface SfxEventSpec {
  readonly waveform: WaveformType;
  readonly startTime: number;
  readonly duration: number;
  readonly gain: number;
  readonly midi?: number;
  readonly freqStart?: number;
  readonly freqEnd?: number;
  readonly filterFreq?: number;
}

export interface SfxDefinition {
  readonly name: SfxName;
  readonly totalDuration: number;
  readonly events: readonly SfxEventSpec[];
}

/**
 * Convert a MIDI note number to frequency in Hz (A4 = MIDI 69 = 440 Hz).
 */
export function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/**
 * 8-bar chord progression: C, C, Am, Am, F, F, G, G (I–I–vi–vi–IV–IV–V–V).
 * Roots sit in the low octave for triangle bass; tones sit in octaves 4–5 for lead arpeggios.
 */
export const BGM_CHORDS: readonly BarChordSpec[] = [
  { chord: 'C', rootMidi: 36, tones: [60, 64, 67] },
  { chord: 'C', rootMidi: 36, tones: [60, 64, 67] },
  { chord: 'Am', rootMidi: 33, tones: [57, 60, 64] },
  { chord: 'Am', rootMidi: 33, tones: [57, 60, 64] },
  { chord: 'F', rootMidi: 41, tones: [65, 69, 72] },
  { chord: 'F', rootMidi: 41, tones: [65, 69, 72] },
  { chord: 'G', rootMidi: 43, tones: [67, 71, 74] },
  { chord: 'G', rootMidi: 43, tones: [67, 71, 74] },
] as const;

/**
 * Bass bar roots (1 root per bar across 8 bars): C2=36, A1=33, F2=41, G2=43.
 */
export const BGM_BASS_ROOTS: readonly number[] = BGM_CHORDS.map((c) => c.rootMidi);

/**
 * Up-down "swell" arpeggio pattern of chord-tone indices per eighth step within a bar.
 */
export const BGM_LEAD_PATTERN: readonly number[] = [0, 1, 2, 1, 0, 1, 2, 1] as const;

/**
 * Compute the lead MIDI note for a given eighth-note step (0..63).
 * Follows the swell arpeggio pattern over each bar's chord tones, lifting the final
 * peak of the second bar in each chord pair by one octave (+12 semitones).
 */
export function getLeadMidiForStep(stepIndex: number): number | null {
  const normalizedStep =
    ((stepIndex % BGM_TOTAL_STEPS) + BGM_TOTAL_STEPS) % BGM_TOTAL_STEPS;
  const barIndex = Math.floor(normalizedStep / BGM_STEPS_PER_BAR);
  const stepInBar = normalizedStep % BGM_STEPS_PER_BAR;
  const chord = BGM_CHORDS[barIndex];
  const toneIndex = BGM_LEAD_PATTERN[stepInBar];
  const baseMidi = chord.tones[toneIndex];
  const isLastBarOfPair = barIndex % 2 === 1;
  if (isLastBarOfPair && stepInBar === 6) {
    return baseMidi + 12;
  }
  return baseMidi;
}

/**
 * Compute the bass MIDI note for a given eighth-note step (0..63).
 * Alternates root on even steps and root+7 on odd steps within each bar.
 */
export function getBassMidiForStep(stepIndex: number): number {
  const normalizedStep =
    ((stepIndex % BGM_TOTAL_STEPS) + BGM_TOTAL_STEPS) % BGM_TOTAL_STEPS;
  const barIndex = Math.floor(normalizedStep / BGM_STEPS_PER_BAR);
  const stepInBar = normalizedStep % BGM_STEPS_PER_BAR;
  const root = BGM_CHORDS[barIndex].rootMidi;
  return stepInBar % 2 === 0 ? root : root + 7;
}

/**
 * Whether an eighth-note step index (0..63) has an off-beat percussion tick (steps 1,3,5,7 of each bar).
 */
export function isPercussionStep(stepIndex: number): boolean {
  const stepInBar =
    ((stepIndex % BGM_STEPS_PER_BAR) + BGM_STEPS_PER_BAR) % BGM_STEPS_PER_BAR;
  return stepInBar % 2 === 1;
}

/**
 * Pure SFX definition table for all named sound effects.
 */
export const SFX_DEFINITIONS: Readonly<Record<SfxName, SfxDefinition>> = {
  paddle: {
    name: 'paddle',
    totalDuration: 0.07,
    events: [
      {
        waveform: 'noise',
        startTime: 0,
        duration: 0.07,
        gain: 0.06,
        filterFreq: 900,
      },
    ],
  },
  takeoff: {
    name: 'takeoff',
    totalDuration: 0.18,
    events: [
      {
        waveform: 'square',
        startTime: 0,
        duration: 0.18,
        gain: 0.16,
        freqStart: 300,
        freqEnd: 920,
      },
    ],
  },
  ride: {
    name: 'ride',
    totalDuration: 0.17,
    events: [
      {
        waveform: 'square',
        startTime: 0,
        duration: 0.07,
        gain: 0.15,
        midi: 76, // E5
      },
      {
        waveform: 'square',
        startTime: 0.07,
        duration: 0.1,
        gain: 0.15,
        midi: 84, // C6
      },
    ],
  },
  wipeout: {
    name: 'wipeout',
    totalDuration: 0.32,
    events: [
      {
        waveform: 'square',
        startTime: 0,
        duration: 0.32,
        gain: 0.18,
        freqStart: 220,
        freqEnd: 55,
      },
      {
        waveform: 'noise',
        startTime: 0,
        duration: 0.25,
        gain: 0.1,
        filterFreq: 700,
      },
    ],
  },
  kickout: {
    name: 'kickout',
    totalDuration: 0.14,
    events: [
      {
        waveform: 'square',
        startTime: 0,
        duration: 0.06,
        gain: 0.1,
        midi: 79, // G5
      },
      {
        waveform: 'square',
        startTime: 0.06,
        duration: 0.08,
        gain: 0.1,
        midi: 84, // C6
      },
    ],
  },
  click: {
    name: 'click',
    totalDuration: 0.05,
    events: [
      {
        waveform: 'square',
        startTime: 0,
        duration: 0.05,
        gain: 0.12,
        freqStart: 880,
        freqEnd: 660,
      },
    ],
  },
};

// --- Runtime Web Audio Engine State ---

const LOOKAHEAD_INTERVAL_MS = 35;
const SCHEDULE_AHEAD_SEC = 0.12;
const BGM_LEAD_GAIN = 0.1;
const BGM_BASS_GAIN = 0.11;
const BGM_NOISE_GAIN = 0.02;

function readInitialMutedState(): boolean {
  try {
    if (typeof localStorage !== 'undefined') {
      return localStorage.getItem(MUTE_STORAGE_KEY) === 'true';
    }
  } catch {
    // Ignore storage access errors in restricted/headless contexts
  }
  return false;
}

let muted: boolean = readInitialMutedState();
let audioCtx: AudioContext | null = null;
let masterGain: GainNode | null = null;
let noiseBuffer: AudioBuffer | null = null;

let schedulerTimer: ReturnType<typeof setInterval> | null = null;
let currentStep = 0;
let nextNoteTime = 0;
let lastPaddleTimeMs = -Infinity;

function getAudioContextConstructor(): typeof AudioContext | null {
  if (typeof globalThis === 'undefined') return null;
  const g = globalThis as unknown as {
    AudioContext?: typeof AudioContext;
    webkitAudioContext?: typeof AudioContext;
  };
  return g.AudioContext ?? g.webkitAudioContext ?? null;
}

function createWhiteNoiseBuffer(ctx: AudioContext): AudioBuffer {
  const durationSec = 0.5;
  const length = Math.max(1, Math.floor(ctx.sampleRate * durationSec));
  const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  // Deterministic LCG noise so buffer creation is fast and consistent
  let seed = 0x12345678;
  for (let i = 0; i < length; i++) {
    seed = (1664525 * seed + 1013904223) >>> 0;
    data[i] = (seed / 0xffffffff) * 2 - 1;
  }
  return buffer;
}

/**
 * Lazily create and resume the Web Audio API context and master gain node.
 * Safe to call multiple times or in headless environments without Web Audio.
 */
export function initAudio(): AudioContext | null {
  if (!audioCtx) {
    const Ctor = getAudioContextConstructor();
    if (!Ctor) return null;
    try {
      audioCtx = new Ctor();
      masterGain = audioCtx.createGain();
      masterGain.gain.value = muted ? 0 : 1;
      masterGain.connect(audioCtx.destination);
      noiseBuffer = createWhiteNoiseBuffer(audioCtx);
    } catch {
      audioCtx = null;
      masterGain = null;
      return null;
    }
  }

  if (audioCtx.state === 'suspended') {
    audioCtx.resume().catch(() => {
      // Ignore autoplay rejection until a valid user gesture occurs
    });
  }

  if (masterGain) {
    masterGain.gain.setValueAtTime(muted ? 0 : 1, audioCtx.currentTime);
  }

  return audioCtx;
}

/**
 * Return whether audio is currently muted.
 */
export function isMuted(): boolean {
  return muted;
}

/**
 * Set the mute state, persist it in localStorage (`surf-world-muted`),
 * and start/stop BGM accordingly.
 */
export function setMuted(m: boolean): void {
  muted = Boolean(m);
  try {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem(MUTE_STORAGE_KEY, String(muted));
    }
  } catch {
    // Ignore localStorage quota/privacy errors
  }

  if (muted) {
    if (masterGain && audioCtx) {
      masterGain.gain.setValueAtTime(0, audioCtx.currentTime);
    }
    stopBgm();
  } else {
    initAudio();
    if (masterGain && audioCtx) {
      masterGain.gain.setValueAtTime(1, audioCtx.currentTime);
    }
    startBgm();
  }
}

function scheduleOscNote(
  ctx: AudioContext,
  dest: AudioNode,
  waveform: 'square' | 'triangle',
  freqStart: number,
  freqEnd: number | undefined,
  startTime: number,
  duration: number,
  peakGain: number
): void {
  const osc = ctx.createOscillator();
  const gainNode = ctx.createGain();

  osc.type = waveform;
  osc.frequency.setValueAtTime(freqStart, startTime);
  if (typeof freqEnd === 'number' && freqEnd !== freqStart) {
    osc.frequency.exponentialRampToValueAtTime(
      Math.max(20, freqEnd),
      startTime + duration
    );
  }

  const attack = Math.min(0.008, duration * 0.15);
  const releaseTime = startTime + duration;
  gainNode.gain.setValueAtTime(0.0001, startTime);
  gainNode.gain.linearRampToValueAtTime(peakGain, startTime + attack);
  gainNode.gain.exponentialRampToValueAtTime(0.0001, releaseTime);

  osc.connect(gainNode);
  gainNode.connect(dest);

  osc.start(startTime);
  osc.stop(releaseTime + 0.005);
}

function scheduleNoiseBurst(
  ctx: AudioContext,
  dest: AudioNode,
  startTime: number,
  duration: number,
  peakGain: number,
  filterFreq = 1800
): void {
  if (!noiseBuffer) {
    noiseBuffer = createWhiteNoiseBuffer(ctx);
  }

  const source = ctx.createBufferSource();
  source.buffer = noiseBuffer;
  source.loop = true;

  const filter = ctx.createBiquadFilter();
  filter.type = 'bandpass';
  filter.frequency.setValueAtTime(filterFreq, startTime);
  filter.Q.setValueAtTime(1.2, startTime);

  const gainNode = ctx.createGain();
  const attack = Math.min(0.005, duration * 0.15);
  const releaseTime = startTime + duration;
  gainNode.gain.setValueAtTime(0.0001, startTime);
  gainNode.gain.linearRampToValueAtTime(peakGain, startTime + attack);
  gainNode.gain.exponentialRampToValueAtTime(0.0001, releaseTime);

  source.connect(filter);
  filter.connect(gainNode);
  gainNode.connect(dest);

  source.start(startTime);
  source.stop(releaseTime + 0.005);
}

function scheduleBgmStep(stepIndex: number, stepTime: number): void {
  if (!audioCtx || !masterGain || muted) return;

  // 1. Lead arpeggio (square wave, quiet swell pattern)
  const leadMidi = getLeadMidiForStep(stepIndex);
  if (leadMidi !== null) {
    const freq = midiToFreq(leadMidi);
    scheduleOscNote(
      audioCtx,
      masterGain,
      'square',
      freq,
      undefined,
      stepTime,
      BGM_STEP_DURATION_SEC * 0.85,
      BGM_LEAD_GAIN
    );
  }

  // 2. Bass (triangle wave, alternating root / root+7 each eighth step)
  const bassMidi = getBassMidiForStep(stepIndex);
  const bassFreq = midiToFreq(bassMidi);
  scheduleOscNote(
    audioCtx,
    masterGain,
    'triangle',
    bassFreq,
    undefined,
    stepTime,
    BGM_STEP_DURATION_SEC * 0.78,
    BGM_BASS_GAIN
  );

  // 3. Soft off-beat percussion noise tick (steps 1,3,5,7 of each bar)
  if (isPercussionStep(stepIndex)) {
    scheduleNoiseBurst(
      audioCtx,
      masterGain,
      stepTime,
      0.03,
      BGM_NOISE_GAIN,
      2800
    );
  }
}

function schedulerTick(): void {
  if (!audioCtx || !masterGain || muted) return;
  if (audioCtx.state !== 'running') return;

  const now = audioCtx.currentTime;
  if (nextNoteTime < now - 0.05) {
    nextNoteTime = now + 0.01;
  }

  while (nextNoteTime < now + SCHEDULE_AHEAD_SEC) {
    scheduleBgmStep(currentStep, nextNoteTime);
    nextNoteTime += BGM_STEP_DURATION_SEC;
    currentStep = (currentStep + 1) % BGM_TOTAL_STEPS;
  }
}

/**
 * Start the looping chiptune BGM using a lookahead scheduler.
 */
export function startBgm(): void {
  if (muted) return;
  const ctx = initAudio();
  if (!ctx) return;

  if (schedulerTimer !== null) {
    return;
  }

  currentStep = 0;
  nextNoteTime = ctx.currentTime + 0.02;
  schedulerTick();
  schedulerTimer = setInterval(schedulerTick, LOOKAHEAD_INTERVAL_MS);
}

/**
 * Stop the BGM lookahead scheduler.
 */
export function stopBgm(): void {
  if (schedulerTimer !== null) {
    clearInterval(schedulerTimer);
    schedulerTimer = null;
  }
}

/**
 * Play one of the named 8-bit sound effects (`paddle`, `takeoff`, `ride`, `wipeout`, `kickout`, `click`).
 * Does NOT create the AudioContext; no-ops until initAudio() has initialized it on a user gesture.
 */
export function playSfx(name: SfxName): void {
  if (muted) return;
  if (!audioCtx || !masterGain) return;

  const def = SFX_DEFINITIONS[name];
  if (!def) return;

  const nowMs =
    typeof performance !== 'undefined' ? performance.now() : Date.now();
  if (name === 'paddle') {
    if (nowMs - lastPaddleTimeMs < PADDLE_MIN_INTERVAL_MS) {
      return;
    }
    lastPaddleTimeMs = nowMs;
  }

  const ctx = audioCtx;
  if (ctx.state === 'suspended') {
    ctx.resume().catch(() => {
      // Ignore autoplay rejection until a valid user gesture occurs
    });
  }

  const baseTime = ctx.currentTime + 0.005;
  for (const ev of def.events) {
    const start = baseTime + ev.startTime;
    if (ev.waveform === 'noise') {
      scheduleNoiseBurst(
        ctx,
        masterGain,
        start,
        ev.duration,
        ev.gain,
        ev.filterFreq ?? 1200
      );
    } else {
      const freqStart =
        typeof ev.midi === 'number'
          ? midiToFreq(ev.midi)
          : (ev.freqStart ?? 440);
      scheduleOscNote(
        ctx,
        masterGain,
        ev.waveform,
        freqStart,
        ev.freqEnd,
        start,
        ev.duration,
        ev.gain
      );
    }
  }
}
