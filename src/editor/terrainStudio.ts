import { GRID_H, GRID_W } from '../contracts/types.ts';
import type { Bathymetry, BathymetryKind } from '../contracts/types.ts';
import {
  applyBrush,
  blocksAt,
  createCustomBathymetry,
  deserializeBathymetry,
  fromPreset,
  serializeBathymetry,
  smoothPass,
} from './terrainModel.ts';
import { renderTerrainPixels } from './terrainRender.ts';

export interface TerrainStudioCallbacks {
  onSurfIt: (bathy: Bathymetry) => void;
}

const WORK_KEY = 'surf-world-terrain-work';
const CUSTOM_SLOT_KEY = 'surf-world-terrain-custom';
const DEFAULT_READOUT = 'Hover the map to inspect a cell.';

function getEl<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing DOM element #${id}`);
  return el as T;
}

function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key: string, value: string): boolean {
  try {
    localStorage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

export function initTerrainStudio(cb: TerrainStudioCallbacks): void {
  const canvas = document.getElementById('studio-canvas') as HTMLCanvasElement | null;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const readoutEl = getEl<HTMLElement>('studio-readout');
  const statusEl = getEl<HTMLElement>('studio-status');
  const raiseBtn = getEl<HTMLButtonElement>('studio-raise');
  const lowerBtn = getEl<HTMLButtonElement>('studio-lower');
  const brush1Btn = getEl<HTMLButtonElement>('studio-brush-1');
  const brush3Btn = getEl<HTMLButtonElement>('studio-brush-3');
  const brush5Btn = getEl<HTMLButtonElement>('studio-brush-5');
  const presetSelect = getEl<HTMLSelectElement>('studio-preset');
  const applyPresetBtn = getEl<HTMLButtonElement>('studio-apply-preset');
  const smoothBtn = getEl<HTMLButtonElement>('studio-smooth');
  const saveBtn = getEl<HTMLButtonElement>('studio-save');
  const loadBtn = getEl<HTMLButtonElement>('studio-load');
  const exportBtn = getEl<HTMLButtonElement>('studio-export');
  const importBtn = getEl<HTMLButtonElement>('studio-import-btn');
  const importInput = getEl<HTMLInputElement>('studio-import');
  const surfBtn = getEl<HTMLButtonElement>('studio-surf');

  const offCanvas = document.createElement('canvas');
  offCanvas.width = GRID_W;
  offCanvas.height = GRID_H;
  const offCtx = offCanvas.getContext('2d');
  if (!offCtx) return;
  const imgData = offCtx.createImageData(GRID_W, GRID_H);

  let working: Bathymetry;
  const savedWork = readStorage(WORK_KEY);
  if (savedWork !== null) {
    try {
      working = deserializeBathymetry(savedWork);
    } catch {
      working = fromPreset('sandbar', 7);
    }
  } else {
    working = fromPreset('sandbar', 7);
  }

  let activeTool: 'raise' | 'lower' = 'raise';
  let brushRadius = 0;
  let hoverX = -1;
  let hoverY = -1;
  let painting = false;

  function setStatus(msg: string): void {
    statusEl.textContent = msg;
  }

  function autosave(): void {
    writeStorage(WORK_KEY, serializeBathymetry(working));
  }

  function updateReadout(): void {
    if (hoverX < 0 || hoverY < 0) {
      readoutEl.textContent = DEFAULT_READOUT;
      return;
    }
    const d = working.depth[hoverY * GRID_W + hoverX];
    const blocks = blocksAt(d);
    readoutEl.textContent = `cell (${hoverX}, ${hoverY}) · depth ${d.toFixed(1)} m · ${blocks} blocks`;
  }

  function redraw(): void {
    renderTerrainPixels(working, imgData.data);
    offCtx!.putImageData(imgData, 0, 0);
    ctx!.imageSmoothingEnabled = true;
    ctx!.drawImage(offCanvas, 0, 0, canvas!.width, canvas!.height);

    if (hoverX >= 0 && hoverY >= 0) {
      const x0 = Math.max(0, hoverX - brushRadius);
      const x1 = Math.min(GRID_W - 1, hoverX + brushRadius);
      const y0 = Math.max(0, hoverY - brushRadius);
      const y1 = Math.min(GRID_H - 1, hoverY + brushRadius);
      const scaleX = canvas!.width / GRID_W;
      const scaleY = canvas!.height / GRID_H;
      const rx = x0 * scaleX;
      const ry = y0 * scaleY;
      const rw = (x1 - x0 + 1) * scaleX;
      const rh = (y1 - y0 + 1) * scaleY;
      ctx!.strokeStyle = '#ffd166';
      ctx!.lineWidth = 2;
      ctx!.strokeRect(rx + 1, ry + 1, Math.max(1, rw - 2), Math.max(1, rh - 2));
    }
  }

  function eventToCell(e: PointerEvent): { x: number; y: number } {
    const rect = canvas!.getBoundingClientRect();
    const nx = rect.width > 0 ? (e.clientX - rect.left) / rect.width : 0;
    const ny = rect.height > 0 ? (e.clientY - rect.top) / rect.height : 0;
    const x = Math.max(0, Math.min(GRID_W - 1, Math.floor(nx * GRID_W)));
    const y = Math.max(0, Math.min(GRID_H - 1, Math.floor(ny * GRID_H)));
    return { x, y };
  }

  function paintAt(cx: number, cy: number): void {
    const delta: 1 | -1 = activeTool === 'raise' ? 1 : -1;
    const changed = applyBrush(working.depth, cx, cy, brushRadius, delta);
    if (changed > 0) {
      autosave();
    }
  }

  canvas.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    painting = true;
    const cell = eventToCell(e);
    hoverX = cell.x;
    hoverY = cell.y;
    paintAt(hoverX, hoverY);
    updateReadout();
    redraw();
  });

  canvas.addEventListener('pointermove', (e) => {
    const cell = eventToCell(e);
    hoverX = cell.x;
    hoverY = cell.y;
    if (painting && (e.buttons & 1) !== 0) {
      paintAt(hoverX, hoverY);
    }
    updateReadout();
    redraw();
  });

  const stopPainting = (): void => {
    painting = false;
  };
  canvas.addEventListener('pointerup', stopPainting);
  canvas.addEventListener('pointercancel', stopPainting);
  canvas.addEventListener('pointerleave', () => {
    painting = false;
    hoverX = -1;
    hoverY = -1;
    updateReadout();
    redraw();
  });

  const toolBtns: [HTMLButtonElement, 'raise' | 'lower'][] = [
    [raiseBtn, 'raise'],
    [lowerBtn, 'lower'],
  ];
  for (const [btn, tool] of toolBtns) {
    btn.addEventListener('click', () => {
      activeTool = tool;
      for (const [b] of toolBtns) {
        b.classList.toggle('active', b === btn);
      }
    });
  }

  const brushBtns: [HTMLButtonElement, number][] = [
    [brush1Btn, 0],
    [brush3Btn, 1],
    [brush5Btn, 2],
  ];
  for (const [btn, radius] of brushBtns) {
    btn.addEventListener('click', () => {
      brushRadius = radius;
      for (const [b] of brushBtns) {
        b.classList.toggle('active', b === btn);
      }
      redraw();
    });
  }

  applyPresetBtn.addEventListener('click', () => {
    const preset = presetSelect.value as BathymetryKind | 'flat';
    working = fromPreset(preset, 7);
    redraw();
    autosave();
    updateReadout();
    setStatus(`Preset loaded: ${preset}`);
  });

  smoothBtn.addEventListener('click', () => {
    working.depth = smoothPass(working.depth);
    redraw();
    autosave();
    updateReadout();
    setStatus('Smoothed');
  });

  saveBtn.addEventListener('click', () => {
    const ok = writeStorage(CUSTOM_SLOT_KEY, serializeBathymetry(working));
    setStatus(ok ? 'Saved to slot "custom"' : 'Could not save: storage unavailable');
  });

  loadBtn.addEventListener('click', () => {
    const raw = readStorage(CUSTOM_SLOT_KEY);
    if (raw === null) {
      setStatus('No saved terrain yet');
      return;
    }
    try {
      working = deserializeBathymetry(raw);
      redraw();
      autosave();
      updateReadout();
      setStatus('Loaded slot "custom"');
    } catch (err) {
      setStatus(`Failed to load saved terrain: ${err instanceof Error ? err.message : String(err)}`);
    }
  });

  exportBtn.addEventListener('click', () => {
    const json = serializeBathymetry(working);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'custom-terrain.json';
    a.click();
    URL.revokeObjectURL(url);
    setStatus('Exported custom-terrain.json');
  });

  importBtn.addEventListener('click', () => {
    importInput.value = '';
    importInput.click();
  });

  importInput.addEventListener('change', async () => {
    const file = importInput.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      working = deserializeBathymetry(text);
      redraw();
      autosave();
      updateReadout();
      setStatus(`Imported ${file.name}`);
    } catch (err) {
      setStatus(`Import error: ${err instanceof Error ? err.message : String(err)}`);
    }
  });

  surfBtn.addEventListener('click', () => {
    cb.onSurfIt(createCustomBathymetry(new Float32Array(working.depth)));
  });

  updateReadout();
  redraw();
}
