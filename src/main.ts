import './style.css';
import { SurferState, WORLD_H_M, WORLD_W_M } from './contracts/types.ts';
import type { BathymetryKind, SurferInput, WorldState } from './contracts/types.ts';
import { RuleAgent } from './policies/ruleAgent.ts';
import { SurfRenderer } from './render/canvasRenderer.ts';
import { createWorld, stepWorld } from './sim/world.ts';

function getEl<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing DOM element #${id}`);
  return el as T;
}

// --- Settings (mirrors the HUD controls) ---
let seed = 7;
let bathyKind: BathymetryKind = 'sandbar';
let swellHeight = 1.8;
let swellPeriod = 9;
let swellDirection = 8;
let tide = 0;
let aiCount = 3;

let world!: WorldState; // assigned by buildWorld() before any use
let agents: RuleAgent[] = [];

function buildWorld(): void {
  world = createWorld(
    {
      bathymetry: bathyKind,
      swell: { heightM: swellHeight, periodS: swellPeriod, directionDeg: swellDirection },
      tideM: tide,
      aiCount,
    },
    seed
  );
  agents = [];
  for (let i = 1; i < world.numSurfers; i++) {
    agents.push(new RuleAgent(i - 1, aiCount));
  }
  seedLabel.textContent = `seed ${seed}`;
  setStatus('Paddle out. WASD / arrows / drag.');
}

// --- DOM ---
const canvas = getEl<HTMLCanvasElement>('surf-canvas');
const statusMsg = getEl<HTMLElement>('status-msg');
const hudScore = getEl<HTMLElement>('hud-score');
const hudCatches = getEl<HTMLElement>('hud-catches');
const hudBest = getEl<HTMLElement>('hud-best');
const hudStaminaFill = getEl<HTMLElement>('hud-stamina-fill');
const hudFps = getEl<HTMLElement>('hud-fps');
const hudSimMs = getEl<HTMLElement>('hud-simms');
const seedLabel = getEl<HTMLElement>('seed-label');

function setStatus(msg: string): void {
  statusMsg.textContent = msg;
}

buildWorld();
const renderer = new SurfRenderer(canvas, world.config.gridW, world.config.gridH);

// --- Player input: keyboard + pointer drag ---
const heldKeys = new Set<string>();
let pointerActive = false;
let pointerX = 0;
let pointerY = 0;
let takeoffQueued = false;

function canvasToWorld(clientX: number, clientY: number): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((clientX - rect.left) / rect.width) * WORLD_W_M,
    y: ((clientY - rect.top) / rect.height) * WORLD_H_M,
  };
}

canvas.addEventListener('pointerdown', (e) => {
  e.preventDefault();
  canvas.setPointerCapture(e.pointerId);
  pointerActive = true;
  const p = canvasToWorld(e.clientX, e.clientY);
  pointerX = p.x;
  pointerY = p.y;
});
canvas.addEventListener('pointermove', (e) => {
  if (!pointerActive) return;
  const p = canvasToWorld(e.clientX, e.clientY);
  pointerX = p.x;
  pointerY = p.y;
});
const endPointer = (): void => {
  pointerActive = false;
};
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') {
    e.preventDefault();
    takeoffQueued = true;
    return;
  }
  if (
    ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'KeyW', 'KeyA', 'KeyS', 'KeyD'].includes(e.code)
  ) {
    e.preventDefault();
    heldKeys.add(e.code);
  }
});
window.addEventListener('keyup', (e) => {
  heldKeys.delete(e.code);
});

function playerInput(): SurferInput {
  let moveX = 0;
  let moveY = 0;
  if (heldKeys.has('ArrowLeft') || heldKeys.has('KeyA')) moveX -= 1;
  if (heldKeys.has('ArrowRight') || heldKeys.has('KeyD')) moveX += 1;
  if (heldKeys.has('ArrowUp') || heldKeys.has('KeyW')) moveY -= 1;
  if (heldKeys.has('ArrowDown') || heldKeys.has('KeyS')) moveY += 1;
  if (moveX === 0 && moveY === 0 && pointerActive) {
    const dx = pointerX - world.surferX[0];
    const dy = pointerY - world.surferY[0];
    const dist = Math.hypot(dx, dy);
    if (dist > 6) {
      moveX = dx / dist;
      moveY = dy / dist;
    }
  }
  const input: SurferInput = { moveX, moveY, takeoff: takeoffQueued };
  takeoffQueued = false;
  return input;
}

// --- Controls wiring ---
const selBathy = getEl<HTMLSelectElement>('sel-bathy');
selBathy.addEventListener('change', () => {
  bathyKind = selBathy.value as BathymetryKind;
  buildWorld();
});

function bindSlider(
  id: string,
  valId: string,
  fmt: (v: number) => string,
  apply: (v: number) => void
): void {
  const el = getEl<HTMLInputElement>(id);
  const val = getEl<HTMLElement>(valId);
  el.addEventListener('input', () => {
    val.textContent = fmt(parseFloat(el.value));
  });
  el.addEventListener('change', () => {
    apply(parseFloat(el.value));
    buildWorld();
  });
}

bindSlider('rng-height', 'val-height', (v) => `${v.toFixed(1)} m`, (v) => (swellHeight = v));
bindSlider('rng-period', 'val-period', (v) => `${v.toFixed(1)} s`, (v) => (swellPeriod = v));
bindSlider('rng-direction', 'val-direction', (v) => `${v.toFixed(0)}°`, (v) => (swellDirection = v));
bindSlider('rng-tide', 'val-tide', (v) => `${v.toFixed(2)} m`, (v) => (tide = v));
bindSlider('rng-ai', 'val-ai', (v) => `${v.toFixed(0)}`, (v) => (aiCount = Math.round(v)));

getEl<HTMLButtonElement>('btn-restart').addEventListener('click', () => {
  seed = (seed + 1) >>> 0;
  buildWorld();
});

// --- Fixed-step loop with measured FPS / sim ms ---
let lastFrame = performance.now();
let accumulator = 0;
let fpsFrames = 0;
let fpsTime = 0;
let simMsAvg = 0;
let lastPlayerState: SurferState = SurferState.PADDLE;

function updateHud(): void {
  hudScore.textContent = String(Math.round(world.surferScore[0]));
  hudCatches.textContent = String(world.surferCatches[0]);
  hudBest.textContent = `${world.surferBestRide[0].toFixed(1)}s`;
  const stam = Math.round(world.surferStamina[0]);
  hudStaminaFill.style.width = `${stam}%`;
  hudStaminaFill.style.backgroundColor = stam > 30 ? 'var(--accent)' : 'var(--danger)';
  const state = world.surferState[0] as SurferState;
  if (state !== lastPlayerState) {
    if (state === SurferState.RIDE) setStatus('🏄 Riding! Stay ahead of the foam.');
    else if (state === SurferState.WIPEOUT) setStatus('💥 Wiped out! Paddling back out…');
    else if (state === SurferState.TAKEOFF) setStatus('⚡ Takeoff!');
    lastPlayerState = state;
  }
}

function frame(now: number): void {
  const elapsed = Math.min(0.25, (now - lastFrame) / 1000);
  lastFrame = now;
  accumulator += elapsed;
  const dt = world.config.dt;
  let steps = 0;
  const t0 = performance.now();
  while (accumulator >= dt && steps < 12) {
    const inputs: SurferInput[] = new Array(world.numSurfers);
    inputs[0] = playerInput();
    for (let i = 1; i < world.numSurfers; i++) {
      inputs[i] = agents[i - 1].decide(world, i);
    }
    stepWorld(world, inputs);
    accumulator -= dt;
    steps++;
  }
  if (steps > 0) {
    const ms = performance.now() - t0;
    simMsAvg = simMsAvg * 0.9 + ms * 0.1;
  }
  renderer.render(world);
  updateHud();
  fpsFrames++;
  fpsTime += elapsed;
  if (fpsTime >= 0.5) {
    hudFps.textContent = String(Math.round(fpsFrames / fpsTime));
    hudSimMs.textContent = simMsAvg.toFixed(2);
    fpsFrames = 0;
    fpsTime = 0;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
