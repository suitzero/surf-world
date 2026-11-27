import { CELL_M } from '../contracts/types.ts';
import type { Bathymetry, OceanField, SwellParams } from '../contracts/types.ts';

export const GRAVITY = 9.81;
/** Depth ratio H/d above which a crest is considered breaking (McCowan). */
export const BREAKING_RATIO = 0.78;
/** Upper edge of the fresh-breaking band; beyond it the wave is whitewater. */
export const BREAKING_RATIO_MAX = 1.15;
export const REF_DEPTH_M = 12;

const INF = 1e9;

/** Shallow-water wave speed c = sqrt(g * depth). Dry cells get ~0. */
export function waveSpeed(depthM: number): number {
  return Math.sqrt(GRAVITY * Math.max(depthM, 0.02));
}

/** Green's-law shoaling coefficient (d_ref / d)^(1/4), clamped. */
export function shoalingK(depthM: number): number {
  const d = Math.max(depthM, 0.3);
  const k = Math.pow(REF_DEPTH_M / d, 0.25);
  return k < 0.6 ? 0.6 : k > 2.2 ? 2.2 : k;
}

/**
 * Solve the eikonal equation |grad tau| = 1/c over the grid by relaxed
 * Gauss-Seidel sweeps (alternating direction), seeded on the boundary with
 * the incoming plane wave's delay tau_pw = (x sinθ + y cosθ) / c_deep.
 * The boundary tilt is what makes an angled swell sweep alongshore;
 * slower shallow cells bend and delay crests (refraction/shoaling proxy).
 * Deterministic: fixed sweep order, fixed iteration cap.
 */
function solveTravelTime(
  depthEff: Float32Array,
  speed: Float32Array,
  w: number,
  h: number,
  cellM: number,
  thetaRad: number
): Float32Array {
  const tau = new Float32Array(w * h).fill(INF);
  const sinT = Math.sin(thetaRad);
  const cosT = Math.cos(thetaRad);
  let maxDepth = 0.1;
  for (let i = 0; i < depthEff.length; i++) if (depthEff[i] > maxDepth) maxDepth = depthEff[i];
  const cDeep = waveSpeed(maxDepth);
  // Seed all four edges with the plane-wave delay.
  for (let x = 0; x < w; x++) {
    const xm = x * cellM;
    tau[x] = (xm * sinT) / cDeep; // top row (y = 0)
    tau[(h - 1) * w + x] = (xm * sinT + (h - 1) * cellM * cosT) / cDeep;
  }
  for (let y = 0; y < h; y++) {
    const ym = y * cellM;
    tau[y * w] = (ym * cosT) / cDeep;
    tau[y * w + (w - 1)] = ((w - 1) * cellM * sinT + ym * cosT) / cDeep;
  }
  const isBoundary = (x: number, y: number): boolean =>
    x === 0 || y === 0 || x === w - 1 || y === h - 1;

  for (let iter = 0; iter < 400; iter++) {
    let maxDelta = 0;
    const forward = iter % 2 === 0;
    const yStart = forward ? 0 : h - 1;
    const yEnd = forward ? h : -1;
    const yStep = forward ? 1 : -1;
    const xStart = forward ? 0 : w - 1;
    const xEnd = forward ? w : -1;
    const xStep = forward ? 1 : -1;
    for (let y = yStart; y !== yEnd; y += yStep) {
      for (let x = xStart; x !== xEnd; x += xStep) {
        if (isBoundary(x, y)) continue;
        const i = y * w + x;
        const stepCost = cellM / speed[i];
        let best = tau[i];
        const l = tau[i - 1] + stepCost;
        if (l < best) best = l;
        const r = tau[i + 1] + stepCost;
        if (r < best) best = r;
        const u = tau[i - w] + stepCost;
        if (u < best) best = u;
        const d = tau[i + w] + stepCost;
        if (d < best) best = d;
        if (best < tau[i]) {
          const delta = tau[i] - best;
          if (delta > maxDelta) maxDelta = delta;
          tau[i] = best;
        }
      }
    }
    if (maxDelta < 1e-5) break;
  }
  return tau;
}

/** Build the static ocean field for a bathymetry + swell + tide combo. */
export function buildOcean(bathy: Bathymetry, swell: SwellParams, tideM: number): OceanField {
  const w = bathy.width;
  const h = bathy.height;
  const n = w * h;
  const depthEff = new Float32Array(n);
  const speed = new Float32Array(n);
  const amp = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = bathy.depth[i] + tideM;
    depthEff[i] = d;
    speed[i] = waveSpeed(d);
    amp[i] = d > 0.3 ? swell.heightM * shoalingK(d) : 0;
  }
  const theta = (swell.directionDeg * Math.PI) / 180;
  const tau = solveTravelTime(depthEff, speed, w, h, CELL_M, theta);
  return {
    tau,
    speed,
    amp,
    depthEff,
    cosPhase: new Float32Array(n),
    ratio: new Float32Array(n),
    breaking: new Uint8Array(n),
    foam: new Float32Array(n),
  };
}

/**
 * Advance the dynamic field by one step: phase, set envelope, breaking mask
 * and foam. Waves arrive in sets: amplitude is modulated by a slow envelope
 * travelling with the wave (function of t - tau), group period = 7 * T.
 */
export function updateOcean(field: OceanField, swell: SwellParams, time: number, w: number, h: number): void {
  const omega = (2 * Math.PI) / swell.periodS;
  const groupT = 7 * swell.periodS;
  const groupOmega = (2 * Math.PI) / groupT;
  const theta = (swell.directionDeg * Math.PI) / 180;
  // Upwave look-back (cells): the fresh break line is where the ratio first
  // crosses the threshold along the propagation direction. Cells shoreward
  // of an already-supercritical upwave neighbour are whitewater, not a face.
  const backX = Math.round(Math.sin(theta) * 3);
  const backY = Math.round(Math.cos(theta) * 3);
  const { tau, amp, depthEff, cosPhase, ratio, breaking, foam } = field;
  const foamDecay = 0.996;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const i = row + x;
      const d = depthEff[i];
      if (d <= 0.25 || amp[i] <= 0) {
        cosPhase[i] = 0;
        ratio[i] = 0;
        breaking[i] = 0;
        foam[i] *= foamDecay;
        continue;
      }
      const s = time - tau[i];
      const cp = Math.cos(omega * s);
      cosPhase[i] = cp;
      const env = 0.6 + 0.4 * Math.cos(groupOmega * s);
      const r = (amp[i] * env) / d;
      ratio[i] = r;
      const ux = x - backX;
      const uy = y - backY;
      const upRatio =
        ux >= 0 && ux < w && uy >= 0 && uy < h ? ratio[uy * w + ux] : 0;
      const isBreaking =
        cp > 0.45 && r >= BREAKING_RATIO && upRatio < BREAKING_RATIO * 1.03 && d >= 0.7 ? 1 : 0;
      breaking[i] = isBreaking;
      // Foam: fresh at breaking cells, drifts shoreward (+y), decays slowly.
      let f = foam[i] * foamDecay;
      if (isBreaking) f = 1;
      if (y >= 2) {
        const up = foam[i - 2 * w] * 0.93;
        if (up > f) f = up;
      }
      foam[i] = f > 1 ? 1 : f;
    }
  }
}

/**
 * Peel direction sign at a cell: +1 if breaking sweeps toward +x, -1 toward
 * -x, 0 if the crest arrives (nearly) simultaneously alongshore.
 */
export function peelSignAt(field: OceanField, w: number, cellX: number, cellY: number): number {
  const cx = Math.max(1, Math.min(w - 2, cellX));
  const i = cellY * w + cx;
  const gx = (field.tau[i + 1] - field.tau[i - 1]) / (2 * CELL_M); // s per metre
  if (gx > 0.004) return 1;
  if (gx < -0.004) return -1;
  return 0;
}
