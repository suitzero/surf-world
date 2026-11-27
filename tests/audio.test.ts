import { describe, expect, it } from 'vitest';
import {
  BGM_BAR_COUNT,
  BGM_CHORDS,
  BGM_STEPS_PER_BAR,
  BGM_TOTAL_STEPS,
  MUTE_STORAGE_KEY,
  SFX_DEFINITIONS,
  getBassMidiForStep,
  getLeadMidiForStep,
  initAudio,
  isMuted,
  midiToFreq,
  playSfx,
  setMuted,
  startBgm,
  stopBgm,
} from '../src/audio/audioEngine.ts';
import type { SfxName } from '../src/audio/audioEngine.ts';

describe('P6 audio engine', () => {
  it('converts MIDI notes to frequency accurately', () => {
    expect(midiToFreq(69)).toBe(440);
    expect(midiToFreq(60)).toBeCloseTo(261.63, 2);
  });

  it('generates deterministic lead notes from each bar chord across all 8 bars', () => {
    expect(BGM_TOTAL_STEPS).toBe(64);
    expect(BGM_BAR_COUNT).toBe(8);
    expect(BGM_STEPS_PER_BAR).toBe(8);
    expect(BGM_CHORDS).toHaveLength(8);

    const sampleStepsInBar = [0, 1, 2, 3, 6, 7];
    for (let bar = 0; bar < BGM_BAR_COUNT; bar++) {
      const chord = BGM_CHORDS[bar];
      const allowedMidis = new Set<number>([
        ...chord.tones,
        chord.tones[2] + 12,
      ]);
      const allowedPitchClasses = new Set<number>(
        chord.tones.map((tone) => tone % 12)
      );

      for (const offset of sampleStepsInBar) {
        const step = bar * BGM_STEPS_PER_BAR + offset;
        const midi = getLeadMidiForStep(step);
        expect(midi).not.toBeNull();
        expect(allowedMidis.has(midi!)).toBe(true);
        expect(allowedPitchClasses.has(midi! % 12)).toBe(true);
        // Deterministic across repeated calls and loop wrap-around
        expect(getLeadMidiForStep(step)).toBe(midi);
        expect(getLeadMidiForStep(step + BGM_TOTAL_STEPS)).toBe(midi);
      }
    }
  });

  it('alternates bass notes between root and root+7 within every bar', () => {
    for (let bar = 0; bar < BGM_BAR_COUNT; bar++) {
      const root = BGM_CHORDS[bar].rootMidi;
      for (let stepInBar = 0; stepInBar < BGM_STEPS_PER_BAR; stepInBar++) {
        const step = bar * BGM_STEPS_PER_BAR + stepInBar;
        const expected = stepInBar % 2 === 0 ? root : root + 7;
        expect(getBassMidiForStep(step)).toBe(expected);
      }
    }
  });

  it('defines all six SFX with positive durations and gains', () => {
    const names: readonly SfxName[] = [
      'paddle',
      'takeoff',
      'ride',
      'wipeout',
      'kickout',
      'click',
    ];
    expect(Object.keys(SFX_DEFINITIONS).sort()).toEqual([...names].sort());

    for (const name of names) {
      const def = SFX_DEFINITIONS[name];
      expect(def.name).toBe(name);
      expect(def.totalDuration).toBeGreaterThan(0);
      expect(def.events.length).toBeGreaterThan(0);
      for (const ev of def.events) {
        expect(ev.duration).toBeGreaterThan(0);
        expect(ev.gain).toBeGreaterThan(0);
        expect(ev.startTime).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('is safe to invoke in headless node environments without Web Audio or localStorage', () => {
    expect(MUTE_STORAGE_KEY).toBe('surf-world-muted');
    expect(initAudio()).toBeNull();
    expect(() => playSfx('wipeout')).not.toThrow();
    expect(() => playSfx('paddle')).not.toThrow();
    expect(() => startBgm()).not.toThrow();
    expect(() => stopBgm()).not.toThrow();
    expect(() => setMuted(true)).not.toThrow();
    expect(isMuted()).toBe(true);
    expect(() => setMuted(false)).not.toThrow();
    expect(isMuted()).toBe(false);
  });
});
