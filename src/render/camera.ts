import { WORLD_H_M, WORLD_W_M } from '../contracts/types.ts';

export type CameraMode = 'follow' | 'full';

export interface CameraState {
  mode: CameraMode;
  cx: number;
  cy: number;
}

export const FOLLOW_VIEW_H_M = WORLD_H_M * 0.45;

export function createCamera(mode: CameraMode): CameraState {
  return {
    mode,
    cx: WORLD_W_M / 2,
    cy: WORLD_H_M / 2,
  };
}

export function viewHeightM(cam: CameraState): number {
  return cam.mode === 'full' ? WORLD_H_M : FOLLOW_VIEW_H_M;
}

export function viewRect(
  cam: CameraState,
  aspect: number
): { x: number; y: number; w: number; h: number } {
  let h = viewHeightM(cam);
  let w = h * aspect;
  if (w > WORLD_W_M) {
    w = WORLD_W_M;
    h = w / aspect;
  }
  let x = cam.cx - w / 2;
  let y = cam.cy - h / 2;
  if (x < 0) x = 0;
  else if (x + w > WORLD_W_M) x = WORLD_W_M - w;
  if (y < 0) y = 0;
  else if (y + h > WORLD_H_M) y = WORLD_H_M - h;
  return { x, y, w, h };
}

export function worldToScreen(
  cam: CameraState,
  canvasW: number,
  canvasH: number,
  wx: number,
  wy: number
): { x: number; y: number } {
  const rect = viewRect(cam, canvasW / canvasH);
  return {
    x: ((wx - rect.x) / rect.w) * canvasW,
    y: ((wy - rect.y) / rect.h) * canvasH,
  };
}

export function screenToWorld(
  cam: CameraState,
  canvasW: number,
  canvasH: number,
  sx: number,
  sy: number
): { x: number; y: number } {
  const rect = viewRect(cam, canvasW / canvasH);
  return {
    x: rect.x + (sx / canvasW) * rect.w,
    y: rect.y + (sy / canvasH) * rect.h,
  };
}

export function updateCamera(
  cam: CameraState,
  targetX: number,
  targetY: number,
  lookX: number,
  lookY: number,
  dt: number,
  aspect: number
): void {
  const goalX = targetX + lookX;
  const goalY = targetY + lookY;
  if (dt <= 0) {
    cam.cx = goalX;
    cam.cy = goalY;
  } else {
    const k = 1 - Math.exp(-dt / 0.18);
    cam.cx += (goalX - cam.cx) * k;
    cam.cy += (goalY - cam.cy) * k;
  }

  const rect = viewRect(cam, aspect);
  const halfW = rect.w / 2;
  const halfH = rect.h / 2;

  if (rect.w >= WORLD_W_M) {
    cam.cx = WORLD_W_M / 2;
  } else {
    cam.cx = Math.max(halfW, Math.min(WORLD_W_M - halfW, cam.cx));
  }

  if (rect.h >= WORLD_H_M) {
    cam.cy = WORLD_H_M / 2;
  } else {
    cam.cy = Math.max(halfH, Math.min(WORLD_H_M - halfH, cam.cy));
  }
}
