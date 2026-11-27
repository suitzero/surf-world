import { describe, expect, it } from 'vitest';
import { WORLD_H_M, WORLD_W_M } from '../src/contracts/types.ts';
import {
  createCamera,
  screenToWorld,
  updateCamera,
  viewRect,
  worldToScreen,
} from '../src/render/camera.ts';
import type { CameraState } from '../src/render/camera.ts';

describe('P2 camera', () => {
  it('round-trips worldToScreen and screenToWorld across camera modes and corners', () => {
    const canvasW = 768;
    const canvasH = 576;
    const cameras: CameraState[] = [
      createCamera('full'),
      { mode: 'follow', cx: 256, cy: 192 },
      { mode: 'follow', cx: 20, cy: 370 },
    ];
    const points: [number, number][] = [
      [0, 0],
      [256, 192],
      [511, 383],
      [100.5, 40.25],
    ];

    for (const cam of cameras) {
      for (const [wx, wy] of points) {
        const s = worldToScreen(cam, canvasW, canvasH, wx, wy);
        const w = screenToWorld(cam, canvasW, canvasH, s.x, s.y);
        expect(Math.abs(w.x - wx)).toBeLessThan(1e-6);
        expect(Math.abs(w.y - wy)).toBeLessThan(1e-6);
      }
    }
  });

  it('returns the full world rectangle in full mode at 768x576 aspect', () => {
    const cam = createCamera('full');
    expect(viewRect(cam, 768 / 576)).toEqual({
      x: 0,
      y: 0,
      w: 512,
      h: 384,
    });
  });

  it('clamps follow camera viewRect inside world bounds when tracking corner target', () => {
    const aspect = 768 / 576;
    const cam = createCamera('follow');
    for (let i = 0; i < 600; i++) {
      updateCamera(cam, 0, 0, 0, 0, 1 / 60, aspect);
      const rect = viewRect(cam, aspect);
      expect(rect.x).toBeGreaterThanOrEqual(0);
      expect(rect.y).toBeGreaterThanOrEqual(0);
      expect(rect.x + rect.w).toBeLessThanOrEqual(WORLD_W_M);
      expect(rect.y + rect.h).toBeLessThanOrEqual(WORLD_H_M);
    }
  });

  it('converges within 0.5 m of clamped goal under exponential damping', () => {
    const aspect = 768 / 576;
    const cam = createCamera('follow');
    const targetX = 400;
    const targetY = 100;

    for (let i = 0; i < 600; i++) {
      updateCamera(cam, targetX, targetY, 0, 0, 1 / 60, aspect);
    }

    const rect = viewRect(cam, aspect);
    const clampedGoalX = Math.max(rect.w / 2, Math.min(WORLD_W_M - rect.w / 2, targetX));
    const clampedGoalY = Math.max(rect.h / 2, Math.min(WORLD_H_M - rect.h / 2, targetY));

    expect(Math.hypot(cam.cx - clampedGoalX, cam.cy - clampedGoalY)).toBeLessThan(0.5);
  });
});
