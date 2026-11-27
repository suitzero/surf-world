import { CELL_M, SurferState, WORLD_H_M, WORLD_W_M } from '../contracts/types.ts';
import type { WorldState } from '../contracts/types.ts';
import {
  createCamera,
  screenToWorld,
  updateCamera,
  viewRect,
  worldToScreen,
} from './camera.ts';
import type { CameraMode, CameraState } from './camera.ts';
import { ParticleKind, ParticleSystem } from './particles.ts';

/**
 * Top-down Canvas renderer (S04). The ocean is painted into a grid-sized
 * ImageData (one pixel per cell) and blitted scaled with bilinear smoothing;
 * surfers are drawn as vector shapes on top.
 * Depth colormap: deep blue -> blue -> turquoise -> shallow -> sand.
 * Crests are lightened by surface elevation; breaking cells and foam are
 * blended toward white.
 */

export interface Rgb {
  r: number;
  g: number;
  b: number;
}

const DEPTH_STOPS: { d: number; c: Rgb }[] = [
  { d: -1.5, c: { r: 232, g: 205, b: 148 } }, // dry sand
  { d: 0, c: { r: 214, g: 196, b: 150 } }, // wet sand / waterline
  { d: 1.2, c: { r: 74, g: 198, b: 196 } }, // shallow turquoise
  { d: 3.5, c: { r: 36, g: 158, b: 205 } },
  { d: 7, c: { r: 22, g: 106, b: 182 } },
  { d: 11, c: { r: 14, g: 66, b: 148 } },
  { d: 15, c: { r: 9, g: 40, b: 112 } }, // deep
];

export function depthColor(d: number, out: Rgb): void {
  const stops = DEPTH_STOPS;
  if (d <= stops[0].d) {
    out.r = stops[0].c.r;
    out.g = stops[0].c.g;
    out.b = stops[0].c.b;
    return;
  }
  for (let s = 1; s < stops.length; s++) {
    if (d <= stops[s].d) {
      const a = stops[s - 1];
      const b = stops[s];
      const t = (d - a.d) / (b.d - a.d);
      out.r = a.c.r + (b.c.r - a.c.r) * t;
      out.g = a.c.g + (b.c.g - a.c.g) * t;
      out.b = a.c.b + (b.c.b - a.c.b) * t;
      return;
    }
  }
  const last = stops[stops.length - 1].c;
  out.r = last.r;
  out.g = last.g;
  out.b = last.b;
}

const AI_COLORS = ['#ffd166', '#ef476f', '#06d6a0', '#c77dff', '#ff9f1c', '#4cc9f0', '#f72585', '#80ed99'];

export class SurfRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly offscreen: HTMLCanvasElement;
  private readonly offCtx: CanvasRenderingContext2D;
  private readonly image: ImageData;
  private readonly col: Rgb = { r: 0, g: 0, b: 0 };
  private readonly camera: CameraState;
  private readonly particles = new ParticleSystem();
  private readonly wakeAcc = new Float32Array(9);
  private readonly sprayAcc = new Float32Array(9);
  private readonly streakAcc = new Float32Array(9);
  private readonly knockSeen = new Float32Array(9);
  private streakSide = 1;

  constructor(
    canvas: HTMLCanvasElement,
    gridW: number,
    gridH: number,
    mode: CameraMode = 'follow'
  ) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2d context unavailable');
    this.ctx = ctx;
    this.offscreen = document.createElement('canvas');
    this.offscreen.width = gridW;
    this.offscreen.height = gridH;
    const offCtx = this.offscreen.getContext('2d');
    if (!offCtx) throw new Error('2d context unavailable');
    this.offCtx = offCtx;
    this.image = offCtx.createImageData(gridW, gridH);
    this.camera = createCamera(mode);
  }

  get cameraMode(): CameraMode {
    return this.camera.mode;
  }

  get activeParticleCount(): number {
    return this.particles.activeCount;
  }

  setCameraMode(mode: CameraMode): void {
    this.camera.mode = mode;
  }

  toggleCameraMode(): CameraMode {
    this.camera.mode = this.camera.mode === 'follow' ? 'full' : 'follow';
    return this.camera.mode;
  }

  snapToPlayer(world: WorldState): void {
    updateCamera(
      this.camera,
      world.surferX[0],
      world.surferY[0],
      0,
      0,
      0,
      this.canvas.width / this.canvas.height
    );
  }

  clientToWorld(clientX: number, clientY: number): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    const sx = ((clientX - rect.left) / rect.width) * this.canvas.width;
    const sy = ((clientY - rect.top) / rect.height) * this.canvas.height;
    return screenToWorld(this.camera, this.canvas.width, this.canvas.height, sx, sy);
  }

  render(world: WorldState, dtSeconds: number): void {
    const { gridW: w, gridH: h } = world.config;
    const o = world.ocean;
    const data = this.image.data;
    const col = this.col;
    for (let i = 0; i < w * h; i++) {
      const d = o.depthEff[i];
      depthColor(d, col);
      let r = col.r;
      let g = col.g;
      let b = col.b;
      if (d > 0.25) {
        // Surface elevation shading: crests lighter, troughs darker.
        const elev = o.cosPhase[i] * o.amp[i];
        const shade = elev * 14;
        r += shade;
        g += shade;
        b += shade;
        if (o.cosPhase[i] > 0.82 && o.breaking[i] === 0) {
          r += 22; // crest highlight line
          g += 22;
          b += 22;
        }
        // Wave face: the steep unbroken slope catches the light. Blend
        // toward a bright glassy teal, stronger toward the lip (crest);
        // the trough just behind the face sits in shadow.
        const faceAmt = o.face[i];
        if (faceAmt > 0.02) {
          const sheen = faceAmt * (0.5 + 0.3 * Math.max(0, o.cosPhase[i]));
          r += (150 - r) * sheen;
          g += (242 - g) * sheen;
          b += (216 - b) * sheen;
        }
        if (o.cosPhase[i] < -0.55) {
          const dk = (-o.cosPhase[i] - 0.55) * 30;
          r -= dk;
          g -= dk;
          b -= dk * 0.6;
        }
        const foamAmt = o.breaking[i] === 1 ? 1 : o.foam[i] * 0.85;
        if (foamAmt > 0.01) {
          r += (255 - r) * foamAmt;
          g += (255 - g) * foamAmt;
          b += (255 - b) * foamAmt;
        }
      }
      const p = i * 4;
      data[p] = r < 0 ? 0 : r > 255 ? 255 : r;
      data[p + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
      data[p + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
      data[p + 3] = 255;
    }
    this.offCtx.putImageData(this.image, 0, 0);
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const aspect = cw / ch;
    if (this.camera.mode === 'follow') {
      let lx = world.surferVX[0] * 1.1;
      let ly = world.surferVY[0] * 1.1;
      const lookLen = Math.hypot(lx, ly);
      if (lookLen > 30) {
        const scale = 30 / lookLen;
        lx *= scale;
        ly *= scale;
      }
      updateCamera(this.camera, world.surferX[0], world.surferY[0], lx, ly, dtSeconds, aspect);
    } else {
      this.camera.cx = WORLD_W_M / 2;
      this.camera.cy = WORLD_H_M / 2;
    }
    const rect = viewRect(this.camera, aspect);
    // Bilinear interpolation between cell colors: the field is continuous,
    // and flat per-cell blocks read as rendering errors (squares) on zoom.
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.drawImage(
      this.offscreen,
      rect.x / CELL_M,
      rect.y / CELL_M,
      rect.w / CELL_M,
      rect.h / CELL_M,
      0,
      0,
      cw,
      ch
    );
    const k = WORLD_H_M / rect.h;
    const pdt = Math.min(dtSeconds, 0.05);
    for (let i = 0; i < world.numSurfers; i++) {
      if (world.surferState[i] !== SurferState.RIDE) continue;
      const vx = world.surferVX[i];
      const vy = world.surferVY[i];
      const speed = Math.hypot(vx, vy);
      if (speed < 0.5) continue;
      const heading = world.surferHeading[i];
      const hx = Math.cos(heading);
      const hy = Math.sin(heading);
      const x = world.surferX[i];
      const y = world.surferY[i];
      const tx = x - hx * 2.3;
      const ty = y - hy * 2.3;

      this.wakeAcc[i] += (5 + speed * 5) * pdt;
      while (this.wakeAcc[i] >= 1) {
        this.wakeAcc[i] -= 1;
        this.particles.emitWake(tx, ty, hx, hy, speed);
      }

      const sprayRate = speed > 3 ? (speed - 3) * 6 : 0;
      this.sprayAcc[i] += sprayRate * pdt;
      while (this.sprayAcc[i] >= 1) {
        this.sprayAcc[i] -= 1;
        this.particles.emitSpray(tx, ty, hx, hy, speed);
      }

      if (i === 0) {
        const streakRate = speed > 4.2 ? (speed - 4.2) * 5 : 0;
        this.streakAcc[i] += streakRate * pdt;
        while (this.streakAcc[i] >= 1) {
          this.streakAcc[i] -= 1;
          const side = (this.streakSide = -this.streakSide);
          const sx = x + -hy * side * 1.1;
          const sy = y + hx * side * 1.1;
          this.particles.emitStreak(sx, sy, vx, vy);
        }
      }
    }
    // Knockout splash: when a rider gets knocked out by the wave owner,
    // burst spray at their position once per knockout event.
    for (let i = 0; i < world.numSurfers; i++) {
      const kc = world.surferKnockouts[i];
      if (kc > this.knockSeen[i]) {
        this.knockSeen[i] = kc;
        const hx = Math.cos(world.surferHeading[i]);
        const hy = Math.sin(world.surferHeading[i]);
        for (let n = 0; n < 12; n++) {
          this.particles.emitSpray(world.surferX[i], world.surferY[i], hx, hy, 5);
        }
      }
    }
    this.particles.update(pdt);

    const pxPerM = ch / rect.h;
    const p = this.particles;
    this.ctx.save();
    for (let i = 0; i < p.capacity; i++) {
      const life = p.life[i];
      if (life <= 0) continue;
      const kind = p.kind[i];
      if (kind !== ParticleKind.WAKE && kind !== ParticleKind.STREAK) continue;
      const lifeFrac = life / p.maxLife[i];
      const pos = worldToScreen(this.camera, cw, ch, p.x[i], p.y[i]);
      if (kind === ParticleKind.WAKE) {
        const radius = p.size[i] * (1.7 - lifeFrac * 1.0) * pxPerM;
        this.ctx.fillStyle = `rgba(238, 250, 252, ${(0.40 * lifeFrac).toFixed(3)})`;
        this.ctx.beginPath();
        this.ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        this.ctx.fill();
      } else {
        this.ctx.strokeStyle = `rgba(255, 255, 255, ${(0.32 * lifeFrac).toFixed(3)})`;
        this.ctx.lineWidth = Math.max(1, 0.09 * pxPerM);
        this.ctx.beginPath();
        this.ctx.moveTo(pos.x, pos.y);
        this.ctx.lineTo(
          pos.x - p.vx[i] * pxPerM * 0.085,
          pos.y - p.vy[i] * pxPerM * 0.085
        );
        this.ctx.stroke();
      }
    }
    this.ctx.restore();

    for (let i = 0; i < world.numSurfers; i++) {
      const pos = worldToScreen(this.camera, cw, ch, world.surferX[i], world.surferY[i]);
      this.drawSurfer(world, i, pos.x, pos.y, k);
    }

    this.ctx.save();
    for (let i = 0; i < p.capacity; i++) {
      const life = p.life[i];
      if (life <= 0) continue;
      if (p.kind[i] !== ParticleKind.SPRAY) continue;
      const lifeFrac = life / p.maxLife[i];
      const pos = worldToScreen(this.camera, cw, ch, p.x[i], p.y[i]);
      const radius = Math.max(0.8, p.size[i] * pxPerM);
      this.ctx.fillStyle = `rgba(255, 255, 255, ${(0.75 * lifeFrac).toFixed(3)})`;
      this.ctx.beginPath();
      this.ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      this.ctx.fill();
    }
    this.ctx.restore();
  }

  private drawSurfer(world: WorldState, i: number, x: number, y: number, k: number): void {
    const ctx = this.ctx;
    const heading = world.surferHeading[i];
    const state = world.surferState[i] as SurferState;
    const isPlayer = i === 0;
    const boardColor = isPlayer ? '#ffffff' : AI_COLORS[(i - 1) % AI_COLORS.length];
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(heading);
    // Knocked-out riders pop up off the board briefly.
    if (
      state === SurferState.WIPEOUT &&
      world.surferKnockTime[i] > -Infinity &&
      world.time - world.surferKnockTime[i] < 0.7
    ) {
      const p = (world.time - world.surferKnockTime[i]) / 0.7;
      const pop = Math.sin(p * Math.PI);
      ctx.translate(0, -pop * 9 * k);
      ctx.rotate(pop * 0.7);
    }
    // Board: elongated ellipse pointing along heading.
    ctx.fillStyle = state === SurferState.WIPEOUT ? '#9aa5b1' : boardColor;
    ctx.beginPath();
    ctx.ellipse(0, 0, 7 * k, 3.1 * k, 0, 0, Math.PI * 2);
    ctx.fill();
    if (isPlayer) {
      ctx.strokeStyle = '#ff5964';
      ctx.lineWidth = 1.6 * k;
      ctx.stroke();
    }
    // Body.
    ctx.fillStyle = isPlayer ? '#ff5964' : '#173042';
    ctx.beginPath();
    ctx.arc((state === SurferState.RIDE ? 1.5 : -0.5) * k, 0, 2.3 * k, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    if (isPlayer) {
      ctx.fillStyle = '#ffffff';
      ctx.font = `bold ${9 * k}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText('YOU', x, y - 10 * k);
    }
  }
}
