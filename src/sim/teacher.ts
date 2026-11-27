import { CELL_M } from '../contracts/types.ts';
import type { Bathymetry } from '../contracts/types.ts';
import { GRAVITY } from './ocean.ts';

/**
 * Standalone 2D linear shallow-water "teacher" solver (simplified teacher v1).
 *
 * WHAT THIS MODEL IS:
 * - Depth-averaged, 2D linear shallow-water equations (LSWE):
 *     d(eta)/dt + d(h*u)/dx + d(h*v)/dy = 0
 *     du/dt     + g * d(eta)/dx         = 0
 *     dv/dt     + g * d(eta)/dy         = 0
 * - Non-dispersive shallow-water phase speed c = sqrt(g * depth).
 * - Independent time-stepping reference solver; does not import or wrap the
 *   game's eikonal ocean approximation.
 * - Strictly deterministic (no Math.random, no Date.now).
 *
 * WHAT THIS MODEL IS NOT (simplified teacher v1, NOT high fidelity):
 * - NOT finite-depth dispersive (no omega^2 = g*k*tanh(k*h) dispersion).
 * - NO wave breaking model or surf-zone dissipation.
 * - NO nonlinear shoaling, advection (u*grad(u)), or finite-amplitude effects.
 * - NO wetting-and-drying shoreline front tracking: cells with depth <= 0 are
 *   permanently inactive dry land with eta = 0 and zero normal flux across any
 *   adjacent face.
 * - Ignores tide; uses bathy.depth directly.
 *
 * GRID & TIME-STEPPING SCHEME:
 * - Staggered Arakawa C-grid on width x height arrays (cell spacing cellM (TeacherOptions, default CELL_M)):
 *   - eta[y*w + x] sits at cell centre ((x + 0.5)*cellM, (y + 0.5)*cellM).
 *   - u[y*w + x]   sits on the +x face between (x, y) and (x + 1, y) at
 *                  ((x + 1)*cellM, (y + 0.5)*cellM), with u = 0 on x = w - 1
 *                  and on any face touching a dry cell.
 *   - v[y*w + x]   sits on the +y face between (x, y) and (x, y + 1) at
 *                  ((x + 0.5)*cellM, (y + 1)*cellM), with v = 0 on y = h - 1
 *                  and on any face touching a dry cell.
 * - Forward-backward (semi-implicit / symplectic) time stepping: each substep
 *   first updates eta from the current face velocities (u, v), then updates
 *   (u, v) from the newly updated eta.
 * - CFL substepping: step(dt) splits dt into n = ceil(dt / dtMax) equal
 *   substeps where dtMax = cflSafety * cellM / (sqrt(2) * cMax) and
 *   cMax = sqrt(g * maxDepth) over wet cells.
 *
 * BOUNDARY CONDITIONS:
 * - Offshore edge (y = 0): forcing/relaxation zone a few rows deep where eta
 *   and (u, v) are relaxed toward the incoming monochromatic plane wave
 *     eta_target = A * sin(omega * t - k * (sin(theta)*x_m + cos(theta)*y_m))
 *     (u, v)_target = eta_target * sqrt(g / depth_local) * (sin(theta), cos(theta))
 *   with omega = 2*pi/periodS, k = omega / c_ref, and c_ref = sqrt(g * h_row0)
 *   averaged over wet cells in row 0. Using a smooth relaxation zone both
 *   injects the incoming wave and absorbs outgoing reflected waves at y = 0.
 * - Left (x = 0), right (x = w - 1), and shoreward bottom (y = h - 1, plus any
 *   dry-land shoreline) edges: Rayleigh damping sponge layers on eta, u, and v
 *   whose damping rate ramps quadratically over spongeCells to suppress
 *   boundary and shoreline reflections over long (120 s) runs.
 * - Dry cells (depth <= 0): eta = 0, u = 0, v = 0; any face touching a dry
 *   cell is forced to zero velocity (zero-flux wall).
 */

export interface TeacherOptions {
  /** Offshore wave amplitude A, metres (default 0.9). */
  amplitudeM?: number;
  /** Wave period T, seconds (default 9). */
  periodS?: number;
  /**
   * Direction the swell travels, degrees from straight-onshore (+y).
   * Positive angles tilt travel toward +x (same convention as SwellParams).
   * Default 0.
   */
  directionDeg?: number;
  /** CFL safety factor applied to cellM / (sqrt(2) * cMax) (default 0.9). */
  cflSafety?: number;
  /** Width in cells of the Rayleigh damping sponge on left/right/bottom edges (default 8). */
  spongeCells?: number;
  /**
   * Cell size in metres (default CELL_M = 4). Bathymetry width/height are taken
   * from the Bathymetry object as before; this only overrides the physical cell
   * size used for CFL, distances and sampling.
   */
  cellM?: number;
}

export interface TeacherSolver {
  readonly width: number;
  readonly height: number;
  readonly cellM: number;
  readonly eta: Float32Array;
  readonly u: Float32Array;
  readonly v: Float32Array;
  readonly time: number;
  readonly lastSubsteps: number;
  step(dt: number): void;
  sampleEta(xM: number, yM: number): number;
}

/** Number of rows from y = 0 in the offshore forcing/relaxation zone. */
const FORCING_ROWS = 6;
/** Peak relaxation rate (1/s) inside the offshore forcing zone (rows 1..FORCING_ROWS-1). */
const FORCING_RELAX_RATE = 2.0;
/** Peak Rayleigh damping rate (1/s) at the outer edge of the sponge layers. */
const SPONGE_MAX_RATE = 2.0;
/** Smooth startup ramp duration (s) to avoid an initial step discontinuity at t = 0. */
const RAMP_DURATION_S = 1.0;

export function createTeacher(bathy: Bathymetry, opts: TeacherOptions = {}): TeacherSolver {
  const w = bathy.width;
  const h = bathy.height;
  const cellM = opts.cellM ?? CELL_M;
  const n = w * h;

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
  // On a flat wet tank this equals (h - y); on sloped bathymetries where dry
  // land begins before y = h - 1, it anchors the shoreward sponge right offshore
  // of the coastline instead of sitting inactive on dry sand.
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
    // distCells = 1 for the outermost wet cell bordering dry land or y = h -> frac = 1.
    // distCells > spongeCells -> frac = 0.
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

  // State fields: surface elevation eta (m) and depth-averaged velocities u, v (m/s).
  const eta = new Float32Array(n);
  const u = new Float32Array(n);
  const v = new Float32Array(n);

  let simTime = 0;
  let lastSubsteps = 0;
  const forcingRows = Math.min(FORCING_ROWS, h);
  const invDx = 1 / cellM;

  function stepSub(dtSub: number, tNext: number): void {
    const dtOverDx = dtSub * invDx;
    const gDtOverDx = GRAVITY * dtOverDx;

    // 1. Forward step for eta from current face velocities (u, v).
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const i = row + x;
        if (!wet[i]) {
          eta[i] = 0;
          continue;
        }
        const fxR = hx[i] * u[i];
        const fxL = x > 0 ? hx[i - 1] * u[i - 1] : 0;
        const fyB = hy[i] * v[i];
        const fyT = y > 0 ? hy[i - w] * v[i - w] : 0;
        eta[i] -= dtOverDx * (fxR - fxL + (fyB - fyT));
      }
    }

    // Smooth startup ramp to avoid a t = 0 discontinuity inside the forcing zone.
    const ramp = tNext < RAMP_DURATION_S ? tNext / RAMP_DURATION_S : 1;
    const ampNow = amplitudeM * ramp;
    const omegaT = omega * tNext;

    // 2. Offshore forcing/relaxation zone on eta (rows 0 .. forcingRows - 1).
    if (cRef > 0 && forcingRows > 0) {
      for (let y = 0; y < forcingRows; y++) {
        const row = y * w;
        const yM = (y + 0.5) * cellM;
        const s = (forcingRows - y) / forcingRows;
        const relax = y === 0 ? 1 : 1 - Math.exp(-FORCING_RELAX_RATE * s * s * dtSub);
        for (let x = 0; x < w; x++) {
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
      for (let i = 0; i < n; i++) {
        const sig = sigmaEta[i];
        if (sig > 0 && wet[i]) {
          eta[i] *= Math.exp(-sig * dtSub);
        }
      }
    }

    // 4. Backward step for (u, v) from the newly updated eta.
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
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
    }

    // 5. Offshore forcing/relaxation zone on (u, v).
    if (cRef > 0 && forcingRows > 0) {
      for (let y = 0; y < forcingRows; y++) {
        const row = y * w;
        const yCenterM = (y + 0.5) * cellM;
        const yFaceM = (y + 1) * cellM;
        const s = (forcingRows - y) / forcingRows;
        const relax = y === 0 ? 1 : 1 - Math.exp(-FORCING_RELAX_RATE * s * s * dtSub);
        for (let x = 0; x < w; x++) {
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
      for (let i = 0; i < n; i++) {
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

  function step(dt: number): void {
    if (!(dt > 0)) {
      lastSubsteps = 0;
      return;
    }
    const substeps = Number.isFinite(dtMax) ? Math.max(1, Math.ceil(dt / dtMax)) : 1;
    const dtSub = dt / substeps;
    lastSubsteps = substeps;

    if (maxDepth <= 0) {
      simTime += dt;
      return;
    }

    for (let s = 0; s < substeps; s++) {
      simTime += dtSub;
      stepSub(dtSub, simTime);
    }
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
    step,
    sampleEta,
  };
}
