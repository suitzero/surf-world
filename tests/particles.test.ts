import { describe, expect, it } from 'vitest';
import { ParticleSystem } from '../src/render/particles.ts';

describe('ParticleSystem', () => {
  it('emits wake, spray, and streak particles and updates 60 frames without throwing', () => {
    const ps = new ParticleSystem();
    ps.emitWake(100, 120, 0.8, 0.6, 5);
    ps.emitSpray(100, 120, 0.8, 0.6, 5);
    ps.emitStreak(101, 119, 4.0, 3.0);

    ps.update(1 / 60);
    expect(ps.activeCount).toBeGreaterThan(0);

    for (let i = 1; i < 60; i++) {
      ps.update(1 / 60);
    }
  });

  it('expires all particles after repeated updates', () => {
    const ps = new ParticleSystem();
    for (let i = 0; i < 10; i++) {
      ps.emitWake(50, 60, 1, 0, 4);
      ps.emitSpray(50, 60, 1, 0, 4);
      ps.emitStreak(50, 60, 4, 0);
    }
    expect(ps.activeCount).toBeGreaterThan(0);

    for (let i = 0; i < 40; i++) {
      ps.update(0.25);
    }
    expect(ps.activeCount).toBe(0);
  });

  it('never exceeds capacity when over-emitting', () => {
    const ps = new ParticleSystem(720);
    for (let i = 0; i < 5000; i++) {
      ps.emitWake(i * 0.1, 100, 1, 0, 5);
    }
    expect(ps.activeCount).toBeLessThanOrEqual(ps.capacity);
  });

  it('keeps all active particle coordinates and velocities finite', () => {
    const ps = new ParticleSystem();
    for (let step = 0; step < 30; step++) {
      ps.emitWake(120 + step, 80, 0.6, 0.8, 6);
      ps.emitSpray(120 + step, 80, 0.6, 0.8, 6);
      ps.emitStreak(120 + step, 80, 3.6, 4.8);
      ps.update(1 / 60);
    }
    expect(ps.activeCount).toBeGreaterThan(0);
    for (let i = 0; i < ps.capacity; i++) {
      if (ps.life[i] > 0) {
        expect(Number.isFinite(ps.x[i])).toBe(true);
        expect(Number.isFinite(ps.y[i])).toBe(true);
        expect(Number.isFinite(ps.vx[i])).toBe(true);
        expect(Number.isFinite(ps.vy[i])).toBe(true);
      }
    }
  });

  it('is deterministic for identical seed and emit/update sequence', () => {
    const a = new ParticleSystem(128, 424242);
    const b = new ParticleSystem(128, 424242);

    for (let step = 0; step < 20; step++) {
      a.emitWake(10 + step, 20, 0.89, 0.45, 5.2);
      b.emitWake(10 + step, 20, 0.89, 0.45, 5.2);
      a.emitSpray(10 + step, 20, 0.89, 0.45, 5.2);
      b.emitSpray(10 + step, 20, 0.89, 0.45, 5.2);
      a.emitStreak(11 + step, 21, 4.6, 2.3);
      b.emitStreak(11 + step, 21, 4.6, 2.3);
      a.update(1 / 60);
      b.update(1 / 60);
    }

    expect([...a.x]).toEqual([...b.x]);
    expect([...a.y]).toEqual([...b.y]);
    expect([...a.life]).toEqual([...b.life]);
  });
});
