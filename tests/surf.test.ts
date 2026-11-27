import { describe, expect, it } from 'vitest';
import { NO_INPUT } from '../src/contracts/types.ts';
import type { BathymetryKind, SwellParams, WorldState } from '../src/contracts/types.ts';
import { RuleAgent } from '../src/policies/ruleAgent.ts';
import { generateBathymetry } from '../src/sim/bathymetry.ts';
import { buildOcean, waveSpeed } from '../src/sim/ocean.ts';
import { createWorld, stepWorld } from '../src/sim/world.ts';

const SWELL: SwellParams = { heightM: 1.8, periodS: 9, directionDeg: 8 };

function runSteps(world: WorldState, steps: number): void {
  for (let s = 0; s < steps; s++) stepWorld(world, [NO_INPUT]);
}

describe('AI Surf World — M1 verification suite', () => {
  it('4. bathymetry determinism: same seed reproduces the same seafloor', () => {
    for (const kind of ['sandbar', 'beach', 'point', 'reef'] as BathymetryKind[]) {
      const a = generateBathymetry(kind, 42);
      const b = generateBathymetry(kind, 42);
      expect(Array.from(a.depth)).toEqual(Array.from(b.depth));
    }
    const s1 = generateBathymetry('sandbar', 42);
    const s2 = generateBathymetry('sandbar', 43);
    expect(Array.from(s1.depth)).not.toEqual(Array.from(s2.depth));
  });

  it('3. wave speed: shallow water is slower than deep water', () => {
    expect(waveSpeed(2)).toBeLessThan(waveSpeed(10));
    expect(waveSpeed(10) / waveSpeed(2)).toBeCloseTo(Math.sqrt(5), 5);
    // Same fact through the built field on a real bathymetry.
    const bathy = generateBathymetry('beach', 5);
    const ocean = buildOcean(bathy, SWELL, 0);
    let shallowSum = 0;
    let shallowN = 0;
    let deepSum = 0;
    let deepN = 0;
    for (let i = 0; i < ocean.depthEff.length; i++) {
      const d = ocean.depthEff[i];
      if (d >= 1.5 && d <= 2.5) {
        shallowSum += ocean.speed[i];
        shallowN++;
      } else if (d >= 10 && d <= 13) {
        deepSum += ocean.speed[i];
        deepN++;
      }
    }
    expect(shallowN).toBeGreaterThan(0);
    expect(deepN).toBeGreaterThan(0);
    expect(shallowSum / shallowN).toBeLessThan(deepSum / deepN);
  });

  it(
    '1. same swell breaks at different locations on sandbar vs straight beach',
    () => {
      const stats = (kind: BathymetryKind): { count: number; offshoreFrac: number } => {
        const world = createWorld({ bathymetry: kind, swell: SWELL, aiCount: 0 }, 7);
        const w = world.config.gridW;
        let count = 0;
        let offshore = 0;
        for (let s = 0; s < 1500; s++) {
          stepWorld(world, [NO_INPUT]);
          if (s % 5 !== 0) continue;
          for (let i = 0; i < world.ocean.breaking.length; i++) {
            if (world.ocean.breaking[i]) {
              count++;
              if (Math.floor(i / w) < 50) offshore++;
            }
          }
        }
        return { count, offshoreFrac: count > 0 ? offshore / count : 0 };
      };
      const bar = stats('sandbar');
      const beach = stats('beach');
      expect(bar.count).toBeGreaterThan(200);
      expect(beach.count).toBeGreaterThan(200);
      // A large share of the sandbar's breaking happens on the offshore bar
      // (rows < 50); the straight beach has essentially no breaking there —
      // its break line sits much closer to shore. Different seafloor,
      // different break location, same swell.
      expect(bar.offshoreFrac).toBeGreaterThan(0.1);
      expect(beach.offshoreFrac).toBeLessThan(0.02);
    },
    30000
  );

  it(
    '2. reversing the swell angle sign flips the peel (crest-arrival lag changes sign)',
    () => {
      const crestLag = (directionDeg: number): number => {
        const world = createWorld(
          {
            bathymetry: 'beach',
            swell: { heightM: 1.8, periodS: 9, directionDeg },
            aiCount: 0,
          },
          11
        );
        const w = world.config.gridW;
        // Find the row with the most breaking activity.
        runSteps(world, 500);
        const rowCount = new Float32Array(world.config.gridH);
        for (let s = 0; s < 300; s++) {
          stepWorld(world, [NO_INPUT]);
          for (let i = 0; i < world.ocean.breaking.length; i++) {
            if (world.ocean.breaking[i]) rowCount[Math.floor(i / w)]++;
          }
        }
        let yb = 0;
        for (let y = 0; y < rowCount.length; y++) if (rowCount[y] > rowCount[yb]) yb = y;
        const xA = 48;
        const xB = 80;
        const idxA = yb * w + xA;
        const idxB = yb * w + xB;
        const N = 2400;
        const seriesA = new Float32Array(N);
        const seriesB = new Float32Array(N);
        for (let s = 0; s < N; s++) {
          stepWorld(world, [NO_INPUT]);
          seriesA[s] = world.ocean.cosPhase[idxA];
          seriesB[s] = world.ocean.cosPhase[idxB];
        }
        // Cross-correlate: peak lag of B relative to A (steps, dt = 0.05 s).
        let bestLag = 0;
        let bestCorr = -Infinity;
        for (let lag = -89; lag <= 89; lag++) {
          let corr = 0;
          for (let t = 100; t < N - 100; t++) {
            corr += seriesA[t] * seriesB[t + lag];
          }
          if (corr > bestCorr) {
            bestCorr = corr;
            bestLag = lag;
          }
        }
        return bestLag;
      };
      const lagPos = crestLag(15);
      const lagNeg = crestLag(-15);
      // +15 deg: crests reach the +x cell later (positive lag). -15 deg: opposite.
      expect(lagPos).toBeGreaterThan(8);
      expect(lagNeg).toBeLessThan(-8);
    },
    30000
  );

  it(
    '5. a rule-based AI agent catches at least one wave on the sandbar',
    () => {
      const world = createWorld(
        { bathymetry: 'sandbar', swell: SWELL, aiCount: 1 },
        7
      );
      const agent = new RuleAgent(0, 1);
      const inputs = [NO_INPUT, NO_INPUT];
      let steps = 0;
      const MAX_STEPS = 7000; // 350 simulated seconds
      while (steps < MAX_STEPS && world.surferCatches[1] === 0) {
        inputs[1] = agent.decide(world, 1);
        stepWorld(world, inputs);
        steps++;
      }
      expect(world.surferCatches[1]).toBeGreaterThanOrEqual(1);
    },
    60000
  );

  it('6. world determinism: identical seed and inputs give identical trajectories', () => {
    const mk = (): WorldState => createWorld({ bathymetry: 'point', swell: SWELL, aiCount: 2 }, 99);
    const a = mk();
    const b = mk();
    const inputs = [NO_INPUT, NO_INPUT, NO_INPUT];
    for (let s = 0; s < 300; s++) {
      stepWorld(a, inputs);
      stepWorld(b, inputs);
    }
    expect(Array.from(a.surferX)).toEqual(Array.from(b.surferX));
    expect(Array.from(a.surferY)).toEqual(Array.from(b.surferY));
    expect(Array.from(a.ocean.foam)).toEqual(Array.from(b.ocean.foam));
  });
});
