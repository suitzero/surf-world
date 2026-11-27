import type { Bathymetry, BathymetryKind, OceanField, SwellParams } from '../contracts/types.ts';
import { generateBathymetry } from '../sim/bathymetry.ts';
import { planTick } from '../sim/governor.ts';
import type { CostTable, TileStateInput } from '../sim/governor.ts';
import { createHybrid, SOLVER_DT } from '../sim/hybrid.ts';
import type { HybridSim } from '../sim/hybrid.ts';
import { buildOcean, updateOcean } from '../sim/ocean.ts';
import { createTeacher } from '../sim/teacher.ts';
import type { TeacherSolver } from '../sim/teacher.ts';
import {
  computeOnsetRows,
  etaPixels,
  foldTimeMax,
  hybridPixels,
  physicsPixels,
} from './labRender.ts';
import type { TeacherConfigureMsg, TeacherSnapshotMsg } from './teacherWorker.ts';

export interface LabSettings {
  kind: BathymetryKind;
  seed: number;
  swell: SwellParams;
  tideM: number;
}

type LabMode = 'physics' | 'accurate' | 'hybrid';

interface SnapshotState {
  eta: Float32Array;
  onsetRow: Int16Array;
  time: number;
  stepMsAvg: number;
  receivedAt: number;
}

const CANVAS_W = 768;
const CANVAS_H = 576;
const CELL_PX = 6;

const BLURBS: Record<LabMode, string> = {
  physics:
    'PHYSICS — eikonal approximation: instant static ratio plus a live breaking mask. This is the field the game above actually plays on.',
  accurate:
    "ACCURATE — the linear shallow-water teacher solving the wave equation live. Yellow line: breaking onset read from the teacher's own field.",
  hybrid:
    'HYBRID — fast field now, teacher corrections as the budget allows. Pink tiles are teacher-corrected; the number in a tile is its correction age in seconds. A correction computed against an older world version is dropped, never applied.',
};

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function getEl<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing DOM element #${id}`);
  return el as T;
}

function calibrate(bathy: Bathymetry, s: LabSettings): CostTable {
  try {
    const teacher = createTeacher(bathy, {
      amplitudeM: s.swell.heightM / 2,
      periodS: s.swell.periodS,
      directionDeg: s.swell.directionDeg,
    });
    const t0 = performance.now();
    for (let i = 0; i < 40; i++) {
      teacher.step(0.05);
    }
    const teacherStepMs = clamp((performance.now() - t0) / 40, 0.05, 5);

    const hyb = createHybrid(bathy, s.swell, s.tideM);
    hyb.pumpAccurate(15);
    const t1 = performance.now();
    hyb.correctTiles();
    const tileApplyMs = clamp((performance.now() - t1) / 16, 0.005, 2);

    return { teacherStepMs, tileApplyMs };
  } catch {
    return { teacherStepMs: 0.56, tileApplyMs: 0.06 };
  }
}

export function initOceanLab(getSettings: () => LabSettings): void {
  const canvasEl = document.getElementById('lab-canvas') as HTMLCanvasElement | null;
  if (!canvasEl) return;
  const ctx = canvasEl.getContext('2d');
  if (!ctx) return;

  const blurbEl = getEl<HTMLElement>('lab-blurb');
  const stateEl = getEl<HTMLElement>('lab-state');
  const coverageEl = getEl<HTMLElement>('lab-coverage');
  const appliedEl = getEl<HTMLElement>('lab-applied');
  const droppedEl = getEl<HTMLElement>('lab-dropped');
  const frameEl = getEl<HTMLElement>('lab-frame');
  const budgetEl = getEl<HTMLInputElement>('lab-budget');
  const budgetValEl = getEl<HTMLElement>('lab-budget-val');
  const sourceBtn = getEl<HTMLButtonElement>('lab-source');
  const perturbBtn = getEl<HTMLButtonElement>('lab-perturb');
  const modeBtns = Array.from(document.querySelectorAll<HTMLButtonElement>('.lab-mode-btn'));

  let mode: LabMode = 'physics';
  let source: 'worker' | 'main' = 'worker';
  let workerFailed = false;
  let budgetMs = 5;
  let banked = 0;
  let settings!: LabSettings;
  let bathy!: Bathymetry;
  let fastField!: OceanField;
  let labTime = 0;
  let hybrid!: HybridSim;
  let costs: CostTable = { teacherStepMs: 0.56, tileApplyMs: 0.06 };
  let mainTeacher!: TeacherSolver;
  let mainTimeMax: Float32Array = new Float32Array(0);
  let mainOnset: Int16Array = new Int16Array(0);
  let stepDebt = 0;
  let worker: Worker | null = null;
  let latestSnapshot: SnapshotState | null = null;
  let perturbCounter = 0;
  let govAcc = 0;

  const offCanvas = document.createElement('canvas');
  let offCtx: CanvasRenderingContext2D | null = null;
  let imgData: ImageData | null = null;

  let lastTs: number | null = null;
  let workEma = 0;
  let frameEma = 16.7;
  let textAcc = 1;

  function syncSourceButton(): void {
    if (workerFailed) {
      sourceBtn.textContent = 'TEACHER: MAIN THREAD (WORKER UNAVAILABLE)';
    } else {
      sourceBtn.textContent = source === 'worker' ? 'TEACHER: WORKER' : 'TEACHER: MAIN THREAD';
    }
  }

  function ensureWorker(): void {
    if (worker || workerFailed) return;
    try {
      worker = new Worker(new URL('./teacherWorker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = (e: MessageEvent) => {
        const m = e.data as TeacherSnapshotMsg;
        if (m && m.type === 'snapshot') {
          latestSnapshot = {
            eta: new Float32Array(m.eta),
            onsetRow: m.onsetRow,
            time: m.time,
            stepMsAvg: m.stepMsAvg,
            receivedAt: performance.now(),
          };
        }
      };
      worker.onerror = () => {
        workerFailed = true;
        source = 'main';
        syncSourceButton();
      };
    } catch {
      workerFailed = true;
      source = 'main';
      syncSourceButton();
    }
  }

  function blit(pixels: Uint8ClampedArray): void {
    if (!offCtx || !imgData) return;
    imgData.data.set(pixels);
    offCtx.putImageData(imgData, 0, 0);
    ctx!.imageSmoothingEnabled = true;
    ctx!.drawImage(offCanvas, 0, 0, CANVAS_W, CANVAS_H);
  }

  function drawOnset(onset: Int16Array): void {
    const w = bathy.width;
    ctx!.beginPath();
    let pen = false;
    for (let x = 0; x < w; x++) {
      const row = onset[x];
      if (row < 0) {
        pen = false;
        continue;
      }
      const px = (x + 0.5) * CELL_PX;
      const py = (row + 0.5) * CELL_PX;
      if (!pen) {
        ctx!.moveTo(px, py);
        pen = true;
      } else {
        ctx!.lineTo(px, py);
      }
    }
    ctx!.strokeStyle = '#ffd166';
    ctx!.lineWidth = 2;
    ctx!.stroke();
  }

  function drawTiles(): void {
    const snap = hybrid.tilesSnapshot();
    ctx!.lineWidth = 1;
    ctx!.font = '11px system-ui';
    ctx!.textAlign = 'center';
    for (const tile of snap) {
      const x0 = tile.x0 * CELL_PX;
      const y0 = tile.y0 * CELL_PX;
      const tw = (tile.x1 - tile.x0) * CELL_PX;
      const th = (tile.y1 - tile.y0) * CELL_PX;
      if (tile.correctedVersion >= 0) {
        ctx!.fillStyle = 'rgba(255,110,199,0.10)';
        ctx!.fillRect(x0, y0, tw, th);
        ctx!.strokeStyle = 'rgba(53,196,220,0.85)';
        ctx!.strokeRect(x0 + 0.5, y0 + 0.5, tw - 1, th - 1);
        ctx!.fillStyle = '#e8f4f8';
        ctx!.fillText(
          `${Math.max(0, hybrid.time - tile.correctedAt).toFixed(0)}s`,
          x0 + tw / 2,
          y0 + th / 2 + 4
        );
      } else {
        ctx!.strokeStyle = 'rgba(255,255,255,0.14)';
        ctx!.strokeRect(x0 + 0.5, y0 + 0.5, tw - 1, th - 1);
      }
    }
  }

  function governorSecond(): void {
    const snap = hybrid.tilesSnapshot();
    const tiles: TileStateInput[] = snap.map((t) => ({
      id: t.id,
      correctedAt: t.correctedAt,
      queries: t.queries,
      eligible: t.eligible,
    }));
    const plan = planTick({
      budgetMs,
      bankedMs: banked,
      now: hybrid.time,
      tiles,
      costs,
    });
    banked = plan.bankedMs;
    const job = hybrid.startCorrection(plan.tilesToCorrect);
    if (plan.teacherSteps > 0) {
      hybrid.pumpAccurate(plan.teacherSteps * SOLVER_DT);
    }
    hybrid.finishCorrection(job);
  }

  function updateReadouts(): void {
    blurbEl.textContent = BLURBS[mode];

    if (mode === 'physics') {
      stateEl.textContent = `eikonal field · t = ${labTime.toFixed(1)} s`;
    } else if (mode === 'accurate') {
      if (source === 'worker') {
        if (latestSnapshot) {
          const ageMs = Math.max(0, Math.round(performance.now() - latestSnapshot.receivedAt));
          stateEl.textContent = `teacher t = ${latestSnapshot.time.toFixed(1)} s · worker · step ${latestSnapshot.stepMsAvg.toFixed(2)} ms · snapshot age ${ageMs} ms`;
        } else {
          stateEl.textContent = 'teacher · worker · waiting for first snapshot…';
        }
      } else {
        stateEl.textContent = `teacher t = ${mainTeacher.time.toFixed(1)} s · main thread · step ${costs.teacherStepMs.toFixed(2)} ms`;
      }
    } else {
      stateEl.textContent = `world t = ${hybrid.time.toFixed(1)} s · teacher t = ${hybrid.teacher.time.toFixed(1)} s · v${hybrid.stateVersion}`;
    }

    if (mode === 'hybrid') {
      const snap = hybrid.tilesSnapshot();
      let correctedCount = 0;
      for (const t of snap) {
        if (t.correctedVersion >= 0) correctedCount++;
      }
      const pct = Math.round((100 * correctedCount) / hybrid.tileCount);
      coverageEl.textContent = `${pct}% of tiles`;
      appliedEl.textContent = String(hybrid.appliedCorrections);
      droppedEl.textContent = String(hybrid.droppedStaleCorrections);
    } else {
      coverageEl.textContent = '—';
      appliedEl.textContent = '—';
      droppedEl.textContent = '—';
    }

    frameEl.textContent = `${workEma.toFixed(2)} ms work · ${frameEma.toFixed(1)} ms frame`;
  }

  function rebuild(): void {
    settings = getSettings();
    bathy = generateBathymetry(settings.kind, settings.seed);
    const w = bathy.width;
    const h = bathy.height;
    if (offCanvas.width !== w || offCanvas.height !== h || !offCtx || !imgData) {
      offCanvas.width = w;
      offCanvas.height = h;
      offCtx = offCanvas.getContext('2d');
      imgData = offCtx ? offCtx.createImageData(w, h) : null;
    }

    fastField = buildOcean(bathy, settings.swell, settings.tideM);
    labTime = 0;
    hybrid = createHybrid(bathy, settings.swell, settings.tideM);
    mainTeacher = createTeacher(bathy, {
      amplitudeM: settings.swell.heightM / 2,
      periodS: settings.swell.periodS,
      directionDeg: settings.swell.directionDeg,
    });
    mainTimeMax = new Float32Array(w * h);
    mainOnset = new Int16Array(w).fill(-1);
    stepDebt = 0;
    costs = calibrate(bathy, settings);
    banked = 0;
    govAcc = 0;
    latestSnapshot = null;

    ensureWorker();
    worker?.postMessage({
      type: 'configure',
      kind: settings.kind,
      seed: settings.seed,
      swell: settings.swell,
    } satisfies TeacherConfigureMsg);

    updateReadouts();
  }

  function loop(ts: number): void {
    const rawDeltaMs = lastTs === null ? 0 : ts - lastTs;
    const elapsed = lastTs === null ? 0 : Math.min(0.25, rawDeltaMs / 1000);
    lastTs = ts;
    if (rawDeltaMs > 0) {
      frameEma = frameEma * 0.9 + rawDeltaMs * 0.1;
    }

    const w = bathy.width;
    const h = bathy.height;
    const amp = settings.swell.heightM / 2;

    const t0 = performance.now();
    if (mode === 'physics') {
      labTime += elapsed;
      updateOcean(fastField, settings.swell, labTime, w, h, elapsed);
      blit(physicsPixels(fastField, bathy.depth, w, h));
    } else if (mode === 'accurate') {
      if (source === 'worker') {
        if (latestSnapshot) {
          blit(etaPixels(latestSnapshot.eta, bathy.depth, w, h, amp));
          drawOnset(latestSnapshot.onsetRow);
        } else {
          blit(etaPixels(new Float32Array(w * h), bathy.depth, w, h, amp));
        }
      } else {
        stepDebt += elapsed;
        const steps = Math.min(40, Math.floor(stepDebt / 0.05));
        for (let i = 0; i < steps; i++) {
          mainTeacher.step(0.05);
          foldTimeMax(mainTimeMax, mainTeacher.eta);
        }
        stepDebt -= steps * 0.05;
        mainOnset = computeOnsetRows(mainTimeMax, bathy.depth, w, h);
        blit(etaPixels(mainTeacher.eta, bathy.depth, w, h, amp));
        drawOnset(mainOnset);
      }
    } else {
      hybrid.advanceWorld(Math.min(elapsed, 0.1));
      govAcc += elapsed;
      while (govAcc >= 1) {
        govAcc -= 1;
        governorSecond();
      }
      blit(hybridPixels(hybrid, bathy.depth, w, h));
      drawTiles();
    }
    const workMs = performance.now() - t0;
    workEma = workEma * 0.9 + workMs * 0.1;

    textAcc += elapsed;
    if (textAcc >= 0.2) {
      textAcc = 0;
      updateReadouts();
    }

    requestAnimationFrame(loop);
  }

  for (const btn of modeBtns) {
    btn.addEventListener('click', () => {
      const m = btn.dataset.mode as LabMode | undefined;
      if (m !== 'physics' && m !== 'accurate' && m !== 'hybrid') return;
      mode = m;
      for (const b of modeBtns) {
        b.classList.toggle('active', b === btn);
      }
      updateReadouts();
    });
  }

  budgetEl.addEventListener('input', () => {
    budgetMs = Math.pow(10, parseFloat(budgetEl.value));
    budgetValEl.textContent = `${budgetMs.toFixed(1)} ms/s`;
  });

  sourceBtn.addEventListener('click', () => {
    if (workerFailed) return;
    source = source === 'worker' ? 'main' : 'worker';
    syncSourceButton();
    updateReadouts();
  });

  perturbBtn.addEventListener('click', () => {
    const snap = hybrid.tilesSnapshot();
    const ids = snap.map((t) => t.id);
    const job = hybrid.startCorrection(ids);
    perturbCounter = (perturbCounter + 1) % hybrid.tilesX;
    const bumped = ids.filter((id) => id % hybrid.tilesX === perturbCounter);
    hybrid.bumpVersion(bumped);
    hybrid.finishCorrection(job);
    updateReadouts();
  });

  for (const id of ['sel-bathy', 'rng-height', 'rng-period', 'rng-direction', 'rng-tide']) {
    document.getElementById(id)?.addEventListener('change', () => rebuild());
  }
  document.getElementById('btn-restart')?.addEventListener('click', () => rebuild());

  syncSourceButton();
  rebuild();
  requestAnimationFrame(loop);
}
