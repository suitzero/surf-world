import { CELL_M } from '../contracts/types.ts';
import type { Bathymetry, OceanField, SwellParams } from '../contracts/types.ts';

export const GRAVITY = 9.81;
/** Depth ratio H/d above which a crest is considered breaking (McCowan). */
export const BREAKING_RATIO = 0.78;
/** Upper edge of the fresh-breaking band; beyond it the wave is whitewater. */
export const BREAKING_RATIO_MAX = 1.15;
export const REF_DEPTH_M = 12;

const INF = 1e9;

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Linear ramp: 0 at lo, 1 at hi. */
function ramp(lo: number, hi: number, v: number): number {
  return clamp01((v - lo) / (hi - lo));
}

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
 * Finite-depth linear dispersion solver: omega^2 = g * k * tanh(k * depthM),
 * solved for wavenumber k (rad/m) via Newton iteration.
 */
export function dispersionK(omega: number, depthM: number): number {
  const omega2 = omega * omega;
  let k = omega2 / GRAVITY;
  if (depthM <= 0.05) return k;
  for (let i = 0; i < 10; i++) {
    const tanhKD = Math.tanh(Math.min(k * depthM, 20));
    const f = GRAVITY * k * tanhKD - omega2;
    const fp = GRAVITY * tanhKD + GRAVITY * k * depthM * (1 - tanhKD * tanhKD);
    k -= f / fp;
    if (k < 1e-4) k = 1e-4;
  }
  return k;
}

/** Finite-depth linear wave group speed c_g = 0.5 * c * (1 + 2kd / sinh(2kd)). */
export function groupSpeed(omega: number, depthM: number): number {
  const k = dispersionK(omega, depthM);
  const c = omega / k;
  if (depthM <= 0.05) return 0.5 * c;
  const kd = k * depthM;
  const twoKD = 2 * kd;
  if (twoKD > 30) return 0.5 * c;
  return 0.5 * c * (1 + twoKD / Math.sinh(twoKD));
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
  thetaRad: number,
  boundarySpeed?: number
): Float32Array {
  const tau = new Float32Array(w * h).fill(INF);
  const sinT = Math.sin(thetaRad);
  const cosT = Math.cos(thetaRad);
  let maxDepth = 0.1;
  for (let i = 0; i < depthEff.length; i++) if (depthEff[i] > maxDepth) maxDepth = depthEff[i];
  const cDeep = boundarySpeed ?? waveSpeed(maxDepth);
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
  const cg = new Float32Array(n);
  const amp = new Float32Array(n);
  const omega = (2 * Math.PI) / swell.periodS;
  let maxDepth = 0.1;
  for (let i = 0; i < n; i++) {
    const d = bathy.depth[i] + tideM;
    depthEff[i] = d;
    if (d > maxDepth) maxDepth = d;
    speed[i] = waveSpeed(d);
    cg[i] = d > 0.25 ? groupSpeed(omega, d) : speed[i];
    amp[i] = d > 0.3 ? swell.heightM * shoalingK(d) : 0;
  }
  const theta = (swell.directionDeg * Math.PI) / 180;
  const tau = solveTravelTime(depthEff, speed, w, h, CELL_M, theta);
  const groupTau = solveTravelTime(depthEff, cg, w, h, CELL_M, theta, groupSpeed(omega, maxDepth));
  return {
    tau,
    groupTau,
    speed,
    amp,
    depthEff,
    cosPhase: new Float32Array(n),
    ratio: new Float32Array(n),
    breaking: new Uint8Array(n),
    foam: new Float32Array(n),
    face: new Float32Array(n),
  };
}

/**
 * Advance the dynamic field by one step: phase, set envelope, breaking mask
 * and foam. Waves arrive in sets: amplitude is modulated by a slow envelope
 * travelling at group velocity (function of t - groupTau), group period = 7 * T.
 */
export function updateOcean(field: OceanField, swell: SwellParams, time: number, w: number, h: number, dt: number): void {
  const omega = (2 * Math.PI) / swell.periodS;
  const groupT = 7 * swell.periodS;
  const groupOmega = (2 * Math.PI) / groupT;
  const theta = (swell.directionDeg * Math.PI) / 180;
  // Upwave look-back (cells): the fresh break line is where the ratio first
  // crosses the threshold along the propagation direction. Cells shoreward
  // of an already-supercritical upwave neighbour are whitewater, not a face.
  const backX = Math.round(Math.sin(theta) * 3);
  const backY = Math.round(Math.cos(theta) * 3);
  const { tau, groupTau, amp, depthEff, cosPhase, ratio, breaking, foam, face } = field;
  // Foam lives ~5 s and is advected shoreward at bore speed; sample the
  // previous step's field so whitewater forms travelling bands that wash
  // in and fade, instead of saturating the whole inside section.
  const foamPrev = foam.slice();
  const foamDecay = Math.exp(-dt / 5);
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
      const sg = time - groupTau[i];
      const env = 0.6 + 0.4 * Math.cos(groupOmega * sg);
      const r = (amp[i] * env) / d;
      ratio[i] = r;
      const ux = x - backX;
      const uy = y - backY;
      const upRatio =
        ux >= 0 && ux < w && uy >= 0 && uy < h ? ratio[uy * w + ux] : 0;
      const isBreaking =
        cp > 0.45 && r >= BREAKING_RATIO && upRatio < BREAKING_RATIO * 1.03 && d >= 0.7 ? 1 : 0;
      breaking[i] = isBreaking;
      // Foam: fresh at breaking cells, then advected shoreward at bore
      // speed (~0.6 * local wave speed, clamped) with decay.
      let f = foamPrev[i] * foamDecay;
      const bore = Math.min(Math.max(field.speed[i] * 0.6, 1), 4.5);
      const driftCells = (bore * dt) / CELL_M;
      const srcY = y - driftCells;
      if (srcY >= 0) {
        const y0 = Math.floor(srcY);
        const frac = srcY - y0;
        const a = foamPrev[y0 * w + x];
        const b = y0 + 1 < h ? foamPrev[(y0 + 1) * w + x] : a;
        const adv = (a + (b - a) * frac) * 0.995;
        if (adv > f) f = adv;
      }
      if (isBreaking) f = 1;
      foam[i] = f > 1 ? 1 : f;
    }
  }

  // Wave face: the steep, rideable, still-unbroken slope just offshore of
  // the break line. Strength peaks where the breaking ratio sits in the
  // sub-breaking band rising toward the crest, weighted by how fast the
  // ratio grows toward shore (shoaling steepening) and suppressed where
  // whitewater has already taken over. Computed in a second pass so the
  // shoreward ratio gradient reads this step's completed ratio field.
  const fwdX = Math.round(Math.sin(theta) * 3);
  const fwdY = Math.round(Math.cos(theta) * 3);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const i = row + x;
      if (depthEff[i] <= 0.25 || amp[i] <= 0) {
        face[i] = 0;
        continue;
      }
      const r = ratio[i];
      const band = Math.min(ramp(0.42, 0.56, r), clamp01((BREAKING_RATIO + 0.12 - r) / 0.12));
      const phaseW = ramp(-0.15, 0.6, cosPhase[i]);
      const jx = x + fwdX < 0 ? 0 : x + fwdX >= w ? w - 1 : x + fwdX;
      const jy = y + fwdY < 0 ? 0 : y + fwdY >= h ? h - 1 : y + fwdY;
      const grad = ratio[jy * w + jx] - r;
      const gradW = clamp01(0.45 + grad * 2.0);
      const foamW = 1 - ramp(0.25, 0.7, foam[i]);
      face[i] = band * phaseW * gradW * foamW;
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
