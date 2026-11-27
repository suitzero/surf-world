import { afterAll, describe, expect, it, vi } from 'vitest';
import { SurferState } from '../src/contracts/types.ts';
import { SurfRenderer } from '../src/render/canvasRenderer.ts';
import { createWorld } from '../src/sim/world.ts';

class FakeCtx {
  imageSmoothingEnabled = true;
  fillStyle = '';
  strokeStyle = '';
  lineWidth = 1;
  font = '';
  textAlign = '';

  createImageData(w: number, h: number): ImageData {
    return {
      data: new Uint8ClampedArray(w * h * 4),
      width: w,
      height: h,
    } as ImageData;
  }

  putImageData(..._args: unknown[]): void {}
  drawImage(..._args: unknown[]): void {}
  save(): void {}
  restore(): void {}
  translate(..._args: unknown[]): void {}
  rotate(..._args: unknown[]): void {}
  beginPath(): void {}
  moveTo(..._args: unknown[]): void {}
  lineTo(..._args: unknown[]): void {}
  ellipse(..._args: unknown[]): void {}
  arc(..._args: unknown[]): void {}
  fill(): void {}
  stroke(): void {}
  fillText(..._args: unknown[]): void {}
}

class FakeCanvas {
  width = 300;
  height = 150;
  private readonly ctx = new FakeCtx();

  getContext(_type: string): FakeCtx {
    return this.ctx;
  }

  getBoundingClientRect(): DOMRect {
    return {
      left: 0,
      top: 0,
      width: this.width,
      height: this.height,
    } as DOMRect;
  }
}

describe('SurfRenderer headless smoke', () => {
  afterAll(() => {
    vi.unstubAllGlobals();
  });

  it('renders riding surfers with particles in follow and full camera modes', () => {
    vi.stubGlobal('document', {
      createElement: () => new FakeCanvas(),
    });

    const world = createWorld({ aiCount: 2 }, 7);
    for (const i of [0, 1]) {
      world.surferState[i] = SurferState.RIDE;
      world.rideDirX[i] = 0.89;
      world.rideDirY[i] = 0.45;
      world.surferVX[i] = 5;
      world.surferVY[i] = 2.2;
      world.surferHeading[i] = Math.atan2(2.2, 5);
    }

    const renderer = new SurfRenderer(
      new FakeCanvas() as unknown as HTMLCanvasElement,
      world.config.gridW,
      world.config.gridH,
      'follow'
    );

    for (let f = 0; f < 60; f++) {
      renderer.render(world, 1 / 60);
    }
    expect(renderer.activeParticleCount).toBeGreaterThan(0);

    renderer.setCameraMode('full');
    for (let f = 0; f < 60; f++) {
      renderer.render(world, 1 / 60);
    }
  });
});
