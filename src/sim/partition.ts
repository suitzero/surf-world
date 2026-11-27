import { CELL_M } from '../contracts/types.ts';
import type { Bathymetry } from '../contracts/types.ts';
import { GRAVITY } from './ocean.ts';
import type { TeacherOptions } from './teacher.ts';

/**
 * Partitioned variant of the 2D linear shallow-water "teacher" solver.
 *
 * WHAT THIS IS:
 * - Re-implements the exact numerical scheme of `src/sim/teacher.ts` (Arakawa C-grid,
 *   forward-backward time stepping, CFL substepping, offshore forcing/relaxation zone,
 *   and quadratic Rayleigh damping sponges) over the global `width x height` arrays,
 *   decomposed into P contiguous vertical column strips with explicit halo exchange
 *   between neighboring partitions.
 *
 * WHY THIS IS NOT A WRAPPER AROUND `createTeacher`:
 * - `teacher.ts` keeps its stepping inside a closure over globally precomputed boundary
 *   arrays (face depths `hx`/`hy`, chamfer `distToShore`, and sponge rates `sigmaEta`/
 *   `sigmaU`/`sigmaV`). Its API cannot step a sub-grid, and instantiating one teacher
 *   per strip would erroneously place side sponge layers along internal partition
 *   boundaries. Re-implementing the identical scheme on the global grid preserves exact
 *   boundary geometry and produces bit-exact results vs `createTeacher`.
 *
 * WHY HALO EXCHANGE HAPPENS EVERY SUBSTEP:
 * - Inside a single CFL substep, phase (1) updates `eta` at column `x1` using the left
 *   neighbor's face velocity `u` at column `x1 - 1`, and phase (4) updates `u` at column
 *   `x1 - 1` using the right neighbor's newly updated `eta` at column `x1`. Exchanging
 *   only once per `step(dt)` call would feed stale halos whenever `step(dt)` splits into
 *   multiple substeps. Therefore each substep performs two one-way column exchanges per
 *   internal boundary: `u` (left -> right) before phase (1), and `eta` (right -> left)
 *   after phase (3) / before phase (4).
 *
 * HONESTY CAVEAT ON TIMINGS:
 * - All partitions execute sequentially within a single JavaScript process/thread.
 *   The reported `stats` (`computeMs`, `exchangeMs`, `exchanges`, `bytesExchanged`)
 *   measure bookkeeping/serialization/halo-exchange overhead and per-partition cost
 *   structure, NOT parallel speedup.
 *
 * DETERMINISM:
 * - Strictly deterministic: no `Math.random`, no `Date.now` (`performance.now` is used
 *   exclusively to populate `stats` timings and never affects simulation state).
 */

export interface PartitionedTeacherOptions extends TeacherOptions {
  /** Number of contiguous column partitions along x (default 1, must satisfy 1 <= partitions <= width). */
  partitions?: number;
}

export interface PartitionStats {
  computeMs: number;
  exchangeMs: number;
  exchanges: number;
  bytesExchanged: number;
}

export interface PartitionedTeacher {
  readonly width: number;
  readonly height: number;
  readonly cellM: number;
  readonly eta: Float32Array;
  readonly u: Float32Array;
  readonly v: Float32Array;
  readonly time: number;
  readonly lastSubsteps: number;
  readonly stats: PartitionStats;
  step(dt: number): void;
  sampleEta(xM: number, yM: number): number;
}

interface PartitionSlice {
  readonly x0: number;
  readonly x1: number;
  /** Left-neighbor `u` column (column `x0 - 1`), null for partition 0. */
  readonly haloU: Float32Array | null;
  readonly haloUBytes: Uint8Array | null;
  /** Right-neighbor `eta` column (column `x1`), null for the last partition. */
  readonly haloEta: Float32Array | null;
  readonly haloEtaBytes: Uint8Array | null;
}

/** Number of rows from y = 0 in the offshore forcing/relaxation zone. */
const FORCING_ROWS = 6;
/** Peak relaxation rate (1/s) inside the offshore forcing zone (rows 1..FORCING_ROWS-1). */
const FORCING_RELAX_RATE = 2.0;
/** Peak Rayleigh damping rate (1/s) at the outer edge of the sponge layers. */
const SPONGE_MAX_RATE = 2.0;
/** Smooth startup ramp duration (s) to avoid an initial step discontinuity at t = 0. */
const RAMP_DURATION_S = 1.0;

export function createPartitionedTeacher(
  bathy: Bathymetry,
  opts: PartitionedTeacherOptions = {}
): PartitionedTeacher {
  const w = bathy.width;
  const h = bathy.height;
  const cellM = opts.cellM ?? CELL_M;
  const n = w * h;

  const partitions = opts.partitions ?? 1;
  if (!Number.isInteger(partitions) || partitions < 1 || partitions > w) {
    throw new RangeError(
      `partitions must be an integer in [1, ${w}], got ${partitions}`
    );
  }

  const amplitudeM = opts.amplitudeM ?? 0.9;
  const periodS = opts.periodS ?? 9;
  const directionDeg = opts.directionDeg ?? 0;
  const cflSafety = opts.cflSafety ?? 0.9;
  const spongeCells = Math.max(0, Math.floor(opts.spongeCells ?? 8));

  const depth = bathy.depth;
  const wet = new Uint8Array(n);
  let maxDepth = 0;

  for (let i = 0; i < n; i++) {
    const d = depth[i];
    if (d > 0) {
      wet[i] = 1;
      if (d > maxDepth) maxDepth = d;
    }
  }

  // Average wet depth along offshore row 0 for plane-wave reference celerity.
  let row0Sum = 0;
  let row0Count = 0;
  for (let x = 0; x < w; x++) {
    const d = depth[x];
    if (d > 0) {
      row0Sum += d;
      row0Count++;
    }
  }
  const refDepth = row0Count > 0 ? row0Sum / row0Count : maxDepth;
  const cRef = refDepth > 0 ? Math.sqrt(GRAVITY * refDepth) : 0;
  const omega = periodS > 0 ? (2 * Math.PI) / periodS : 0;
  const k = cRef > 0 ? omega / cRef : 0;
  const theta = (directionDeg * Math.PI) / 180;
  const sinT = Math.sin(theta);
  const cosT = Math.cos(theta);
  const kx = k * sinT;
  const ky = k * cosT;

  const cMax = maxDepth > 0 ? Math.sqrt(GRAVITY * maxDepth) : 0;
  const dtMax = cMax > 0 ? (cflSafety * cellM) / (Math.SQRT2 * cMax) : Infinity;

  // Precompute face depths on the staggered C-grid.
  // A face is inactive (depth = 0) if it lies on the outer edge or touches a dry cell.
  const hx = new Float32Array(n);
  const hy = new Float32Array(n);
  const invSqrtHx = new Float32Array(n);
  const invSqrtHy = new Float32Array(n);

  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const i = row + x;
      if (x < w - 1 && wet[i] && wet[i + 1]) {
        const dFace = 0.5 * (depth[i] + depth[i + 1]);
        hx[i] = dFace;
        invSqrtHx[i] = Math.sqrt(GRAVITY / dFace);
      }
      if (y < h - 1 && wet[i] && wet[i + w]) {
        const dFace = 0.5 * (depth[i] + depth[i + w]);
        hy[i] = dFace;
        invSqrtHy[i] = Math.sqrt(GRAVITY / dFace);
      }
    }
  }

  // Compute isotropic cell distance from each cell to the shoreward boundary
  // (either dry land depth <= 0 or the bottom edge y = h) via two-pass chamfer.
  const distToShore = new Float32Array(n);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const distBottom = h - y;
    for (let x = 0; x < w; x++) {
      const i = row + x;
      distToShore[i] = wet[i] ? distBottom : 0;
    }
  }
  const DIAG = Math.SQRT2;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const i = row + x;
      let d = distToShore[i];
      if (d === 0) continue;
      if (x > 0 && distToShore[i - 1] + 1 < d) d = distToShore[i - 1] + 1;
      if (y > 0) {
        const up = i - w;
        if (distToShore[up] + 1 < d) d = distToShore[up] + 1;
        if (x > 0 && distToShore[up - 1] + DIAG < d) d = distToShore[up - 1] + DIAG;
        if (x < w - 1 && distToShore[up + 1] + DIAG < d) d = distToShore[up + 1] + DIAG;
      }
      distToShore[i] = d;
    }
  }
  for (let y = h - 1; y >= 0; y--) {
    const row = y * w;
    for (let x = w - 1; x >= 0; x--) {
      const i = row + x;
      let d = distToShore[i];
      if (d === 0) continue;
      if (x < w - 1 && distToShore[i + 1] + 1 < d) d = distToShore[i + 1] + 1;
      if (y < h - 1) {
        const dn = i + w;
        if (distToShore[dn] + 1 < d) d = distToShore[dn] + 1;
        if (x < w - 1 && distToShore[dn + 1] + DIAG < d) d = distToShore[dn + 1] + DIAG;
        if (x > 0 && distToShore[dn - 1] + DIAG < d) d = distToShore[dn - 1] + DIAG;
      }
      distToShore[i] = d;
    }
  }

  // Precompute quadratic Rayleigh sponge rates (1/s) on left, right, and shoreward edges.
  const sigmaEta = new Float32Array(n);
  const sigmaU = new Float32Array(n);
  const sigmaV = new Float32Array(n);

  const shoreSpongeFrac = (distCells: number): number => {
    if (spongeCells <= 0) return 0;
    const r = (spongeCells - (distCells - 1)) / spongeCells;
    return r <= 0 ? 0 : r >= 1 ? 1 : r;
  };

  const sideSpongeFrac = (xPos: number): number => {
    if (spongeCells <= 0) return 0;
    const dLeft = xPos < spongeCells ? (spongeCells - xPos) / spongeCells : 0;
    const dRight = xPos > w - 1 - spongeCells ? (xPos - (w - 1 - spongeCells)) / spongeCells : 0;
    return dLeft > dRight ? dLeft : dRight;
  };

  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const i = row + x;
      const rSideEta = sideSpongeFrac(x);
      const rShoreEta = shoreSpongeFrac(distToShore[i]);
      const rEta = rSideEta > rShoreEta ? rSideEta : rShoreEta;
      sigmaEta[i] = rEta > 0 ? SPONGE_MAX_RATE * rEta * rEta : 0;

      const rSideU = sideSpongeFrac(x + 0.5);
      const dShoreU =
        x < w - 1 ? 0.5 * (distToShore[i] + distToShore[i + 1]) : distToShore[i];
      const rShoreU = shoreSpongeFrac(dShoreU);
      const rU = rSideU > rShoreU ? rSideU : rShoreU;
      sigmaU[i] = rU > 0 ? SPONGE_MAX_RATE * rU * rU : 0;

      const rSideV = sideSpongeFrac(x);
      const dShoreV =
        y < h - 1 ? 0.5 * (distToShore[i] + distToShore[i + w]) : distToShore[i];
      const rShoreV = shoreSpongeFrac(dShoreV);
      const rV = rSideV > rShoreV ? rSideV : rShoreV;
      sigmaV[i] = rV > 0 ? SPONGE_MAX_RATE * rV * rV : 0;
    }
  }

  // Split width into P contiguous column partitions as evenly as possible (sizes differ by <= 1).
  const baseWidth = Math.floor(w / partitions);
  const extra = w % partitions;
  const slices: PartitionSlice[] = [];
  let cursor = 0;
  for (let p = 0; p < partitions; p++) {
    const sliceW = baseWidth + (p < extra ? 1 : 0);
    const x0 = cursor;
    const x1 = cursor + sliceW;
    cursor = x1;

    const haloU = p > 0 ? new Float32Array(h) : null;
    const haloUBytes = haloU !== null ? new Uint8Array(haloU.buffer) : null;
    const haloEta = p < partitions - 1 ? new Float32Array(h) : null;
    const haloEtaBytes = haloEta !== null ? new Uint8Array(haloEta.buffer) : null;

    slices.push({ x0, x1, haloU, haloUBytes, haloEta, haloEtaBytes });
  }

  // Scratch pack buffer for strided column -> contiguous Float32Array packing.
  const columnBytes = h * 4;
  const packBuf = new Float32Array(h);
  const packBytes = new Uint8Array(packBuf.buffer);

  // Global state fields: surface elevation eta (m) and depth-averaged velocities u, v (m/s).
  const eta = new Float32Array(n);
  const u = new Float32Array(n);
  const v = new Float32Array(n);

  const stats: PartitionStats = {
    computeMs: 0,
    exchangeMs: 0,
    exchanges: 0,
    bytesExchanged: 0,
  };

  let simTime = 0;
  let lastSubsteps = 0;
  const forcingRows = Math.min(FORCING_ROWS, h);
  const invDx = 1 / cellM;

  function exchangeColumn(
    src: Float32Array,
    colX: number,
    destHaloBytes: Uint8Array
  ): void {
    // Pack strided column into contiguous Float32Array.
    for (let y = 0; y < h; y++) {
      packBuf[y] = src[y * w + colX];
    }
    // Serialize into a fresh Uint8Array wire buffer.
    const wire = new Uint8Array(columnBytes);
    wire.set(packBytes);
    // Deserialize back out into destination partition's halo Float32Array.
    destHaloBytes.set(wire);
    stats.exchanges += 1;
    stats.bytesExchanged += columnBytes;
  }

  function stepSub(dtSub: number, tNext: number): number {
    const dtOverDx = dtSub * invDx;
    const gDtOverDx = GRAVITY * dtOverDx;
    let substepExchangeMs = 0;

    // Halo exchange before phase (1): left partition's u at x1 - 1 -> right partition's haloU.
    if (partitions > 1) {
      const exStart = performance.now();
      for (let b = 0; b < partitions - 1; b++) {
        const left = slices[b];
        const right = slices[b + 1];
        exchangeColumn(u, left.x1 - 1, right.haloUBytes!);
      }
      substepExchangeMs += performance.now() - exStart;
    }

    // Smooth startup ramp to avoid a t = 0 discontinuity inside the forcing zone.
    const ramp = tNext < RAMP_DURATION_S ? tNext / RAMP_DURATION_S : 1;
    const ampNow = amplitudeM * ramp;
    const omegaT = omega * tNext;

    // Phases (1), (2), (3) per partition: eta update, offshore forcing on eta, sponge on eta.
    for (let p = 0; p < partitions; p++) {
      const slice = slices[p];
      const x0 = slice.x0;
      const x1 = slice.x1;
      const haloU = slice.haloU;

      // 1. Forward step for eta from current face velocities (u, v).
      for (let y = 0; y < h; y++) {
        const row = y * w;
        // First column of partition (x = x0): read left face u from haloU if internal boundary.
        {
          const i = row + x0;
          if (!wet[i]) {
            eta[i] = 0;
          } else {
            const fxR = hx[i] * u[i];
            const fxL = haloU !== null ? hx[i - 1] * haloU[y] : 0;
            const fyB = hy[i] * v[i];
            const fyT = y > 0 ? hy[i - w] * v[i - w] : 0;
            eta[i] -= dtOverDx * (fxR - fxL + (fyB - fyT));
          }
        }
        for (let x = x0 + 1; x < x1; x++) {
          const i = row + x;
          if (!wet[i]) {
            eta[i] = 0;
            continue;
          }
          const fxR = hx[i] * u[i];
          const fxL = hx[i - 1] * u[i - 1];
          const fyB = hy[i] * v[i];
          const fyT = y > 0 ? hy[i - w] * v[i - w] : 0;
          eta[i] -= dtOverDx * (fxR - fxL + (fyB - fyT));
        }
      }

      // 2. Offshore forcing/relaxation zone on eta (rows 0 .. forcingRows - 1).
      if (cRef > 0 && forcingRows > 0) {
        for (let y = 0; y < forcingRows; y++) {
          const row = y * w;
          const yM = (y + 0.5) * cellM;
          const s = (forcingRows - y) / forcingRows;
          const relax = y === 0 ? 1 : 1 - Math.exp(-FORCING_RELAX_RATE * s * s * dtSub);
          for (let x = x0; x < x1; x++) {
            const i = row + x;
            if (!wet[i]) continue;
            const xM = (x + 0.5) * cellM;
            const etaTarget = ampNow * Math.sin(omegaT - (kx * xM + ky * yM));
            eta[i] += relax * (etaTarget - eta[i]);
          }
        }
      }

      // 3. Rayleigh damping sponge on eta (left, right, shoreward bottom edges).
      if (spongeCells > 0) {
        for (let y = 0; y < h; y++) {
          const row = y * w;
          for (let x = x0; x < x1; x++) {
            const i = row + x;
            const sig = sigmaEta[i];
            if (sig > 0 && wet[i]) {
              eta[i] *= Math.exp(-sig * dtSub);
            }
          }
        }
      }
    }

    // Halo exchange after phase (3), before phase (4):
    // right partition's new eta at x1 -> left partition's haloEta.
    if (partitions > 1) {
      const exStart = performance.now();
      for (let b = 0; b < partitions - 1; b++) {
        const left = slices[b];
        const right = slices[b + 1];
        exchangeColumn(eta, right.x0, left.haloEtaBytes!);
      }
      substepExchangeMs += performance.now() - exStart;
    }

    // Phases (4), (5), (6) per partition: (u, v) update, offshore forcing on (u, v), sponge on (u, v).
    for (let p = 0; p < partitions; p++) {
      const slice = slices[p];
      const x0 = slice.x0;
      const x1 = slice.x1;
      const haloEta = slice.haloEta;
      const xInnerEnd = haloEta !== null ? x1 - 1 : x1;

      // 4. Backward step for (u, v) from the newly updated eta.
      for (let y = 0; y < h; y++) {
        const row = y * w;
        for (let x = x0; x < xInnerEnd; x++) {
          const i = row + x;
          if (hx[i] > 0) {
            u[i] -= gDtOverDx * (eta[i + 1] - eta[i]);
          } else {
            u[i] = 0;
          }
          if (hy[i] > 0) {
            v[i] -= gDtOverDx * (eta[i + w] - eta[i]);
          } else {
            v[i] = 0;
          }
        }
        if (haloEta !== null) {
          const i = row + (x1 - 1);
          if (hx[i] > 0) {
            u[i] -= gDtOverDx * (haloEta[y] - eta[i]);
          } else {
            u[i] = 0;
          }
          if (hy[i] > 0) {
            v[i] -= gDtOverDx * (eta[i + w] - eta[i]);
          } else {
            v[i] = 0;
          }
        }
      }

      // 5. Offshore forcing/relaxation zone on (u, v).
      if (cRef > 0 && forcingRows > 0) {
        for (let y = 0; y < forcingRows; y++) {
          const row = y * w;
          const yCenterM = (y + 0.5) * cellM;
          const yFaceM = (y + 1) * cellM;
          const s = (forcingRows - y) / forcingRows;
          const relax = y === 0 ? 1 : 1 - Math.exp(-FORCING_RELAX_RATE * s * s * dtSub);
          for (let x = x0; x < x1; x++) {
            const i = row + x;
            if (hx[i] > 0) {
              const xFaceM = (x + 1) * cellM;
              const etaFace = ampNow * Math.sin(omegaT - (kx * xFaceM + ky * yCenterM));
              const uTarget = etaFace * invSqrtHx[i] * sinT;
              u[i] += relax * (uTarget - u[i]);
            }
            if (hy[i] > 0) {
              const xCenterM = (x + 0.5) * cellM;
              const etaFace = ampNow * Math.sin(omegaT - (kx * xCenterM + ky * yFaceM));
              const vTarget = etaFace * invSqrtHy[i] * cosT;
              v[i] += relax * (vTarget - v[i]);
            }
          }
        }
      }

      // 6. Rayleigh damping sponge on (u, v).
      if (spongeCells > 0) {
        for (let y = 0; y < h; y++) {
          const row = y * w;
          for (let x = x0; x < x1; x++) {
            const i = row + x;
            const su = sigmaU[i];
            if (su > 0 && hx[i] > 0) {
              u[i] *= Math.exp(-su * dtSub);
            }
            const sv = sigmaV[i];
            if (sv > 0 && hy[i] > 0) {
              v[i] *= Math.exp(-sv * dtSub);
            }
          }
        }
      }
    }

    return substepExchangeMs;
  }

  function step(dt: number): void {
    const stepStart = performance.now();
    if (!(dt > 0)) {
      lastSubsteps = 0;
      stats.computeMs += performance.now() - stepStart;
      return;
    }
    const substeps = Number.isFinite(dtMax) ? Math.max(1, Math.ceil(dt / dtMax)) : 1;
    const dtSub = dt / substeps;
    lastSubsteps = substeps;

    if (maxDepth <= 0) {
      simTime += dt;
      stats.computeMs += performance.now() - stepStart;
      return;
    }

    let stepExchangeMs = 0;
    for (let s = 0; s < substeps; s++) {
      simTime += dtSub;
      stepExchangeMs += stepSub(dtSub, simTime);
    }

    const stepTotalMs = performance.now() - stepStart;
    stats.exchangeMs += stepExchangeMs;
    stats.computeMs += Math.max(0, stepTotalMs - stepExchangeMs);
  }

  function sampleEta(xM: number, yM: number): number {
    if (!Number.isFinite(xM) || !Number.isFinite(yM)) return 0;
    const worldW = w * cellM;
    const worldH = h * cellM;
    if (xM < 0 || xM >= worldW || yM < 0 || yM >= worldH) return 0;

    // If the containing cell is dry, return 0.
    const cx = Math.min(w - 1, Math.floor(xM * invDx));
    const cy = Math.min(h - 1, Math.floor(yM * invDx));
    if (!wet[cy * w + cx]) return 0;

    // Bilinear interpolation between cell centres ((x + 0.5)*cellM, (y + 0.5)*cellM).
    const gx = Math.max(0, Math.min(w - 1, xM * invDx - 0.5));
    const gy = Math.max(0, Math.min(h - 1, yM * invDx - 0.5));
    const x0 = Math.floor(gx);
    const y0 = Math.floor(gy);
    const x1 = Math.min(w - 1, x0 + 1);
    const y1 = Math.min(h - 1, y0 + 1);
    const tx = gx - x0;
    const ty = gy - y0;

    const e00 = eta[y0 * w + x0];
    const e10 = eta[y0 * w + x1];
    const e01 = eta[y1 * w + x0];
    const e11 = eta[y1 * w + x1];

    const top = e00 + (e10 - e00) * tx;
    const bot = e01 + (e11 - e01) * tx;
    return top + (bot - top) * ty;
  }

  return {
    width: w,
    height: h,
    cellM,
    eta,
    u,
    v,
    get time(): number {
      return simTime;
    },
    get lastSubsteps(): number {
      return lastSubsteps;
    },
    stats,
    step,
    sampleEta,
  };
}
