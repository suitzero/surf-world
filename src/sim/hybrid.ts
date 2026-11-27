import type { Bathymetry, OceanField, SwellParams } from '../contracts/types.ts';
import { BREAKING_RATIO, buildOcean, updateOcean } from './ocean.ts';
import { createTeacher } from './teacher.ts';
import type { TeacherOptions, TeacherSolver } from './teacher.ts';

/**
 * Hybrid fast + accurate ocean simulator (plan S15 / M3 S12 semantics).
 *
 * S07e STATE CONTRACT AXES:
 * - snapshot: the bathymetry + swell + tide field configuration backing a tile.
 * - simulation time: the continuous world clock (`time`), advanced by `advanceWorld(dt)`.
 * - version: monotonic identity of the field inputs (`stateVersion` globally, `tile.version`
 *   per tile). `advanceWorld(dt)` advances simulation time and dynamic phase/foam in the fast
 *   eikonal field, and increments `stateVersion`, but does NOT change tile versions because
 *   dynamic phase evolution is not a state-snapshot change; tile versions track field-input
 *   identity (e.g. a swell shift rolling into a region via `bumpVersion` or `setSwell`).
 * - model id: fast eikonal approximation (`src/sim/ocean.ts`) composed with the linear
 *   shallow-water teacher (`src/sim/teacher.ts`, `linear-shallow-water-teacher-v1`).
 *
 * VERSION RULE & CORRECTION SEMANTICS:
 * - A correction job snapshots each target tile's local version `v` at `startCorrection`.
 * - At `finishCorrection`, a correction applies ONLY to tiles whose local version is still `v`.
 * - Stale corrections (`tile.version !== baseVersion`) are dropped and counted
 *   (`droppedStaleCorrections`), never applied.
 * - No learning or backpropagation occurs on this path; corrections are deterministic
 *   tile-scoped overlays derived from the teacher's accumulated `timeMaxAbsEta`.
 * - Teacher is its own ground truth (subject to S07D caveats: non-dispersive phase speed
 *   lead, offshore boundary reflection R = 0.23744, no post-onset breaking dissipation,
 *   linear reef amplification, and onset-line tail variance).
 */

export const TILE_W = 32;
export const TILE_H = 24;
export const SOLVER_DT = 0.05;
export const SAMPLE_DT = 0.5;

const SAMPLE_EVERY_STEPS = Math.round(SAMPLE_DT / SOLVER_DT);

export type Provenance = 'fast' | 'corrected';

export interface HybridOptions {
  tileW?: number;
  tileH?: number;
  cflSafety?: TeacherOptions['cflSafety'];
  spongeCells?: TeacherOptions['spongeCells'];
  cellM?: TeacherOptions['cellM'];
}

export interface HybridTile {
  id: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  version: number;
  correctedVersion: number;
  correctedAt: number;
  queries: number;
}

export interface TileSnapshot extends HybridTile {
  eligible: boolean;
}

export interface CorrectionJob {
  readonly baseVersions: ReadonlyMap<number, number>;
  readonly tileIds: readonly number[];
}

export interface CorrectionResult {
  applied: number[];
  dropped: number[];
  skipped: number[];
}

export interface RatioQueryResult {
  value: number;
  provenance: Provenance;
  correctionAgeS: number;
}

export interface OnsetQueryResult {
  row: number;
  provenance: Provenance;
  correctionAgeS: number;
}

export interface HybridCounters {
  appliedCorrections: number;
  droppedStaleCorrections: number;
  skippedIneligibleTiles: number;
}

export interface HybridSim {
  readonly width: number;
  readonly height: number;
  readonly tilesX: number;
  readonly tilesY: number;
  readonly tileCount: number;
  readonly fast: OceanField;
  readonly teacher: TeacherSolver;
  readonly time: number;
  readonly stateVersion: number;
  readonly corrRatio: Float32Array;
  readonly corrValid: Uint8Array;
  readonly timeMaxAbsEta: Float32Array;
  readonly arrivedAt: Float32Array;
  readonly appliedCorrections: number;
  readonly droppedStaleCorrections: number;
  readonly skippedIneligibleTiles: number;
  readonly counters: HybridCounters;

  advanceWorld(dt: number): void;
  bumpVersion(tileIds?: readonly number[]): void;
  setSwell(newSwell: SwellParams): void;
  pumpAccurate(dt: number): number;
  startCorrection(tileIds?: readonly number[]): CorrectionJob;
  finishCorrection(job: CorrectionJob): CorrectionResult;
  correctTiles(tileIds?: readonly number[]): CorrectionResult;
  hybridRatioAt(x: number, y: number): RatioQueryResult;
  hybridOnsetRow(x: number): OnsetQueryResult;
  fastOnsetRow(x: number): number;
  tileIdAt(x: number, y: number): number;
  tileEligible(id: number): boolean;
  cellMature(x: number, y: number): boolean;
  getTile(id: number): HybridTile;
  tilesSnapshot(): TileSnapshot[];
}

export function createHybrid(
  bathy: Bathymetry,
  swell: SwellParams,
  tideM: number = 0,
  opts: HybridOptions = {}
): HybridSim {
  const w = bathy.width;
  const h = bathy.height;
  const n = w * h;
  const tileW = opts.tileW ?? TILE_W;
  const tileH = opts.tileH ?? TILE_H;
  const tilesX = Math.ceil(w / tileW);
  const tilesY = Math.ceil(h / tileH);
  const tileCount = tilesX * tilesY;

  let currentSwell: SwellParams = { ...swell };
  let fast: OceanField = buildOcean(bathy, currentSwell, tideM);

  const makeTeacher = (s: SwellParams): TeacherSolver =>
    createTeacher(bathy, {
      amplitudeM: s.heightM / 2,
      periodS: s.periodS,
      directionDeg: s.directionDeg,
      cflSafety: opts.cflSafety,
      spongeCells: opts.spongeCells,
      cellM: opts.cellM,
    });

  let teacher: TeacherSolver = makeTeacher(currentSwell);
  let time = 0;
  let stateVersion = 0;
  let teacherStepCount = 0;

  const tiles: HybridTile[] = [];
  for (let ty = 0; ty < tilesY; ty++) {
    const y0 = ty * tileH;
    const y1 = Math.min(h, y0 + tileH);
    for (let tx = 0; tx < tilesX; tx++) {
      const x0 = tx * tileW;
      const x1 = Math.min(w, x0 + tileW);
      tiles.push({
        id: ty * tilesX + tx,
        x0,
        y0,
        x1,
        y1,
        version: 0,
        correctedVersion: -1,
        correctedAt: -Infinity,
        queries: 0,
      });
    }
  }

  const corrRatio = new Float32Array(n);
  const corrValid = new Uint8Array(n);
  const timeMaxAbsEta = new Float32Array(n);
  const arrivedAt = new Float32Array(n).fill(-1);

  let appliedCorrections = 0;
  let droppedStaleCorrections = 0;
  let skippedIneligibleTiles = 0;

  function tileIdAt(x: number, y: number): number {
    const cx = x < 0 ? 0 : x >= w ? w - 1 : x;
    const cy = y < 0 ? 0 : y >= h ? h - 1 : y;
    const tx = Math.floor(cx / tileW);
    const ty = Math.floor(cy / tileH);
    return ty * tilesX + tx;
  }

  function isCellMatureIdx(idx: number): boolean {
    const arr = arrivedAt[idx];
    return arr >= 0 && teacher.time >= arr + currentSwell.periodS;
  }

  function cellMature(x: number, y: number): boolean {
    return isCellMatureIdx(y * w + x);
  }

  function tileEligible(id: number): boolean {
    const tile = tiles[id];
    if (!tile) return false;
    for (let y = tile.y0; y < tile.y1; y++) {
      const row = y * w;
      for (let x = tile.x0; x < tile.x1; x++) {
        if (isCellMatureIdx(row + x)) return true;
      }
    }
    return false;
  }

  function foldTeacherSample(): void {
    const eta = teacher.eta;
    const tNow = teacher.time;
    const ampThresh = 0.1 * (currentSwell.heightM / 2);
    for (let i = 0; i < n; i++) {
      const absEta = Math.abs(eta[i]);
      if (absEta > timeMaxAbsEta[i]) {
        timeMaxAbsEta[i] = absEta;
      }
      if (arrivedAt[i] < 0 && absEta > 0 && absEta >= ampThresh) {
        arrivedAt[i] = tNow;
      }
    }
  }

  function advanceWorld(dt: number): void {
    time += dt;
    stateVersion++;
    updateOcean(fast, currentSwell, time, w, h, dt);
  }

  function bumpVersion(tileIds?: readonly number[]): void {
    if (tileIds === undefined) {
      for (let i = 0; i < tileCount; i++) {
        tiles[i].version++;
      }
    } else {
      for (let i = 0; i < tileIds.length; i++) {
        const id = tileIds[i];
        if (id >= 0 && id < tileCount) {
          tiles[id].version++;
        }
      }
    }
    stateVersion++;
  }

  function setSwell(newSwell: SwellParams): void {
    currentSwell = { ...newSwell };
    fast = buildOcean(bathy, currentSwell, tideM);
    teacher = makeTeacher(currentSwell);
    teacherStepCount = 0;
    timeMaxAbsEta.fill(0);
    arrivedAt.fill(-1);
    corrRatio.fill(0);
    corrValid.fill(0);
    for (let i = 0; i < tileCount; i++) {
      tiles[i].correctedVersion = -1;
      tiles[i].correctedAt = -Infinity;
    }
    bumpVersion();
  }

  function pumpAccurate(dt: number): number {
    const steps = Math.max(0, Math.round(dt / SOLVER_DT));
    for (let s = 0; s < steps; s++) {
      teacher.step(SOLVER_DT);
      teacherStepCount++;
      if (teacherStepCount % SAMPLE_EVERY_STEPS === 0) {
        foldTeacherSample();
      }
    }
    return steps;
  }

  function startCorrection(tileIds?: readonly number[]): CorrectionJob {
    const ids = tileIds === undefined ? tiles.map((t) => t.id) : Array.from(tileIds);
    const baseVersions = new Map<number, number>();
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (id >= 0 && id < tileCount) {
        baseVersions.set(id, tiles[id].version);
      }
    }
    return { baseVersions, tileIds: ids };
  }

  function finishCorrection(job: CorrectionJob): CorrectionResult {
    const applied: number[] = [];
    const dropped: number[] = [];
    const skipped: number[] = [];
    const depth = bathy.depth;

    for (let i = 0; i < job.tileIds.length; i++) {
      const id = job.tileIds[i];
      const tile = tiles[id];
      if (!tile) continue;
      const baseVer = job.baseVersions.get(id);
      if (baseVer === undefined || tile.version !== baseVer) {
        droppedStaleCorrections++;
        dropped.push(id);
        continue;
      }
      if (!tileEligible(id)) {
        skippedIneligibleTiles++;
        skipped.push(id);
        continue;
      }

      for (let y = tile.y0; y < tile.y1; y++) {
        const row = y * w;
        for (let x = tile.x0; x < tile.x1; x++) {
          const idx = row + x;
          if (isCellMatureIdx(idx)) {
            const d = depth[idx];
            corrRatio[idx] = d > 0 ? (2 * timeMaxAbsEta[idx]) / d : 0;
            corrValid[idx] = 1;
          }
        }
      }
      tile.correctedVersion = tile.version;
      tile.correctedAt = time;
      appliedCorrections++;
      applied.push(id);
    }

    return { applied, dropped, skipped };
  }

  function correctTiles(tileIds?: readonly number[]): CorrectionResult {
    return finishCorrection(startCorrection(tileIds));
  }

  function fastRatioAtIdx(idx: number): number {
    const dEff = fast.depthEff[idx];
    return dEff > 0 ? fast.amp[idx] / dEff : 0;
  }

  function tileAge(tile: HybridTile): number {
    return tile.correctedAt === -Infinity ? Infinity : time - tile.correctedAt;
  }

  function hybridRatioAt(x: number, y: number): RatioQueryResult {
    const tile = tiles[tileIdAt(x, y)];
    tile.queries++;
    const idx = y * w + x;
    if (corrValid[idx] === 1) {
      return {
        value: corrRatio[idx],
        provenance: 'corrected',
        correctionAgeS: tileAge(tile),
      };
    }
    return {
      value: fastRatioAtIdx(idx),
      provenance: 'fast',
      correctionAgeS: tileAge(tile),
    };
  }

  function hybridOnsetRow(x: number): OnsetQueryResult {
    const depth = bathy.depth;
    for (let y = 0; y < h; y++) {
      const idx = y * w + x;
      if (depth[idx] > 0) {
        const isCorr = corrValid[idx] === 1;
        const r = isCorr ? corrRatio[idx] : fastRatioAtIdx(idx);
        if (r >= BREAKING_RATIO) {
          const tile = tiles[tileIdAt(x, y)];
          tile.queries++;
          return {
            row: y,
            provenance: isCorr ? 'corrected' : 'fast',
            correctionAgeS: tileAge(tile),
          };
        }
      }
    }
    const bottomTile = tiles[tileIdAt(x, h - 1)];
    bottomTile.queries++;
    return {
      row: -1,
      provenance: 'fast',
      correctionAgeS: tileAge(bottomTile),
    };
  }

  function fastOnsetRow(x: number): number {
    const depth = bathy.depth;
    for (let y = 0; y < h; y++) {
      const idx = y * w + x;
      if (depth[idx] > 0 && fastRatioAtIdx(idx) >= BREAKING_RATIO) {
        return y;
      }
    }
    return -1;
  }

  function getTile(id: number): HybridTile {
    return { ...tiles[id] };
  }

  function tilesSnapshot(): TileSnapshot[] {
    return tiles.map((t) => ({
      ...t,
      eligible: tileEligible(t.id),
    }));
  }

  return {
    width: w,
    height: h,
    tilesX,
    tilesY,
    tileCount,
    get fast(): OceanField {
      return fast;
    },
    get teacher(): TeacherSolver {
      return teacher;
    },
    get time(): number {
      return time;
    },
    get stateVersion(): number {
      return stateVersion;
    },
    corrRatio,
    corrValid,
    timeMaxAbsEta,
    arrivedAt,
    get appliedCorrections(): number {
      return appliedCorrections;
    },
    get droppedStaleCorrections(): number {
      return droppedStaleCorrections;
    },
    get skippedIneligibleTiles(): number {
      return skippedIneligibleTiles;
    },
    get counters(): HybridCounters {
      return {
        appliedCorrections,
        droppedStaleCorrections,
        skippedIneligibleTiles,
      };
    },
    advanceWorld,
    bumpVersion,
    setSwell,
    pumpAccurate,
    startCorrection,
    finishCorrection,
    correctTiles,
    hybridRatioAt,
    hybridOnsetRow,
    fastOnsetRow,
    tileIdAt,
    tileEligible,
    cellMature,
    getTile,
    tilesSnapshot,
  };
}
