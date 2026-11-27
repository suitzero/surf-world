import type { Bathymetry } from '../contracts/types.ts';
import { depthColor } from '../render/canvasRenderer.ts';
import type { Rgb } from '../render/canvasRenderer.ts';

export function renderTerrainPixels(bathy: Bathymetry, out: Uint8ClampedArray): void {
  const count = bathy.width * bathy.height;
  const col: Rgb = { r: 0, g: 0, b: 0 };
  for (let i = 0; i < count; i++) {
    depthColor(bathy.depth[i], col);
    const p = i * 4;
    out[p] = col.r;
    out[p + 1] = col.g;
    out[p + 2] = col.b;
    out[p + 3] = 255;
  }
}
