import { existsSync, readFileSync, statSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  PILOT_EPISODES,
  TEACHER_CAVEATS,
  computeLabels,
  episodeFileName,
} from '../src/data/contract.ts';
import { readEpisodeFile, serializeEpisode } from '../src/data/episode.ts';
import { generateEpisode } from '../src/data/generate.ts';

const PILOT_DIR = 'data/pilot';

describe('S08a pilot dataset', () => {
  for (const spec of PILOT_EPISODES) {
    it(`episode ${spec.index} (${spec.kind}) loads and validates`, () => {
      const path = `${PILOT_DIR}/${episodeFileName(spec)}`;
      expect(
        existsSync(path),
        `missing pilot file ${path} - run: npx vite-node scripts/gen_pilot.ts`
      ).toBe(true);

      const ep = readEpisodeFile(path);

      expect(ep.header.episode.index).toBe(spec.index);
      expect(ep.header.episode.kind).toBe(spec.kind);
      expect(ep.header.episode.seed).toBe(spec.seed);
      expect(ep.header.episode.durationS).toBe(spec.durationS);
      expect(ep.header.episode.swell).toEqual(spec.swell);
      expect(ep.header.caveats).toEqual([...TEACHER_CAVEATS]);

      const { width: w, height: h } = ep.header.grid;
      const numFrames = ep.header.episode.numFrames;

      expect(ep.depth.length).toBe(w * h);
      expect(ep.eta.length).toBe(numFrames * w * h);
      expect(ep.u.length).toBe(numFrames * w * h);
      expect(ep.v.length).toBe(numFrames * w * h);
      expect(ep.timeMaxAbsEta.length).toBe(w * h);
      expect(ep.onsetMask.length).toBe(w * h);
      expect(ep.crestArrivalS.length).toBe(w * h);
      expect(ep.onsetRow.length).toBe(w);
      expect(ep.onsetConfidence.length).toBe(w);

      for (const [name, arr] of [
        ['eta', ep.eta],
        ['u', ep.u],
        ['v', ep.v],
      ] as const) {
        let badIdx = -1;
        for (let i = 0; i < arr.length; i++) {
          if (!Number.isFinite(arr[i])) {
            badIdx = i;
            break;
          }
        }
        expect(badIdx, `${name} contains non-finite value at index ${badIdx}`).toBe(-1);
      }

      const labels = computeLabels(ep.eta, ep.depth, w, h, numFrames);

      const pairs: readonly [string, ArrayLike<number>, ArrayLike<number>][] = [
        ['timeMaxAbsEta', ep.timeMaxAbsEta, labels.timeMaxAbsEta],
        ['onsetMask', ep.onsetMask, labels.onsetMask],
        ['onsetRow', ep.onsetRow, labels.onsetRow],
        ['onsetConfidence', ep.onsetConfidence, labels.onsetConfidence],
        ['crestArrivalS', ep.crestArrivalS, labels.crestArrivalS],
      ];

      for (const [name, storedArr, computedArr] of pairs) {
        let mismatches = 0;
        for (let i = 0; i < storedArr.length; i++) {
          if (storedArr[i] !== computedArr[i]) mismatches++;
        }
        expect(mismatches, `${name} had ${mismatches} mismatches vs computeLabels`).toBe(0);
      }
    });
  }

  it('reports total pilot size', () => {
    let total = 0;
    for (const spec of PILOT_EPISODES) {
      const path = `${PILOT_DIR}/${episodeFileName(spec)}`;
      total += statSync(path).size;
    }
    console.log(`[S08a] pilot total: ${(total / 1048576).toFixed(2)} MiB across 8 episodes`);
    expect(total).toBeGreaterThan(0);
  });

  it(
    'regenerating episode 0 reproduces the stored file byte-for-byte',
    { timeout: 180000 },
    () => {
      const spec = PILOT_EPISODES[0];
      const data = generateEpisode(spec);
      const bytes = serializeEpisode(data);
      const stored = readFileSync(`${PILOT_DIR}/${episodeFileName(spec)}`);
      expect(bytes.length).toBe(stored.length);
      let mismatches = 0;
      for (let i = 0; i < bytes.length; i++) {
        if (bytes[i] !== stored[i]) mismatches++;
      }
      expect(mismatches).toBe(0);
    }
  );
});
