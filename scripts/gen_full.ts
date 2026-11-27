/**
 * Full + heldout dataset generator for M2 AI Ocean training/evaluation.
 *
 * Seed rules (deterministic, derived strictly from kind + swell indices):
 *
 *   FULL_EPISODES (108 episodes = 4 bathymetry kinds x 27 swell combos):
 *     - kindIdx: 0..3 matching BATHYMETRY_KINDS (beach, sandbar, point, reef)
 *     - hIdx:    0..2 into heightM      = [1.2, 1.8, 2.4]
 *     - tIdx:    0..2 into periodS      = [7, 9, 11]
 *     - dIdx:    0..2 into directionDeg = [-12, 0, 12]
 *     - comboIdx = hIdx * 9 + tIdx * 3 + dIdx
 *     - index    = kindIdx * 27 + comboIdx
 *     - seed     = 100000 + kindIdx * 1000 + hIdx * 100 + tIdx * 10 + dIdx
 *     - durationS = 60
 *
 *   HELDOUT_EPISODES (8 episodes = 4 bathymetry kinds x 2 off-grid swell combos):
 *     - variantIdx 0: { heightM: 2.1, periodS: 10, directionDeg: 6 }
 *     - variantIdx 1: { heightM: 1.5, periodS: 8,  directionDeg: -6 }
 *     - index = kindIdx * 2 + variantIdx
 *     - seed  = 200000 + kindIdx * 10 + variantIdx
 *     - durationS = 60
 *
 * Run with: npx vite-node scripts/gen_full.ts
 */

import { mkdirSync, statSync, writeFileSync } from 'node:fs';
import { BATHYMETRY_KINDS } from '../src/contracts/types.ts';
import type { BathymetryKind, SwellParams } from '../src/contracts/types.ts';
import { episodeFileName } from '../src/data/contract.ts';
import type { EpisodeSpec } from '../src/data/contract.ts';
import { readEpisodeFile, serializeEpisode } from '../src/data/episode.ts';
import { generateEpisode } from '../src/data/generate.ts';

declare const process: { cwd(): string };

const DURATION_S = 60;
const FULL_HEIGHTS_M = [1.2, 1.8, 2.4] as const;
const FULL_PERIODS_S = [7, 9, 11] as const;
const FULL_DIRECTIONS_DEG = [-12, 0, 12] as const;

const HELDOUT_SWELLS: readonly SwellParams[] = [
  { heightM: 2.1, periodS: 10, directionDeg: 6 },
  { heightM: 1.5, periodS: 8, directionDeg: -6 },
];

const SEED_RULES: Record<string, string> = {
  full: '100000 + kindIdx*1000 + hIdx*100 + tIdx*10 + dIdx',
  heldout: '200000 + kindIdx*10 + variantIdx',
};

const SHA256_HEX_RE = /^[0-9a-f]{64}$/;

export const FULL_EPISODES: readonly EpisodeSpec[] = (() => {
  const specs: EpisodeSpec[] = [];
  for (let kindIdx = 0; kindIdx < BATHYMETRY_KINDS.length; kindIdx++) {
    const kind = BATHYMETRY_KINDS[kindIdx];
    for (let hIdx = 0; hIdx < FULL_HEIGHTS_M.length; hIdx++) {
      for (let tIdx = 0; tIdx < FULL_PERIODS_S.length; tIdx++) {
        for (let dIdx = 0; dIdx < FULL_DIRECTIONS_DEG.length; dIdx++) {
          const comboIdx = hIdx * 9 + tIdx * 3 + dIdx;
          specs.push({
            index: kindIdx * 27 + comboIdx,
            kind,
            seed: 100000 + kindIdx * 1000 + hIdx * 100 + tIdx * 10 + dIdx,
            swell: {
              heightM: FULL_HEIGHTS_M[hIdx],
              periodS: FULL_PERIODS_S[tIdx],
              directionDeg: FULL_DIRECTIONS_DEG[dIdx],
            },
            durationS: DURATION_S,
          });
        }
      }
    }
  }
  return specs;
})();

export const HELDOUT_EPISODES: readonly EpisodeSpec[] = (() => {
  const specs: EpisodeSpec[] = [];
  for (let kindIdx = 0; kindIdx < BATHYMETRY_KINDS.length; kindIdx++) {
    const kind = BATHYMETRY_KINDS[kindIdx];
    for (let variantIdx = 0; variantIdx < HELDOUT_SWELLS.length; variantIdx++) {
      specs.push({
        index: kindIdx * 2 + variantIdx,
        kind,
        seed: 200000 + kindIdx * 10 + variantIdx,
        swell: { ...HELDOUT_SWELLS[variantIdx] },
        durationS: DURATION_S,
      });
    }
  }
  return specs;
})();

function comboKey(spec: EpisodeSpec): string {
  return `${spec.kind}|${spec.swell.heightM}|${spec.swell.periodS}|${spec.swell.directionDeg}`;
}

function assertHeldoutDisjoint(
  fullSpecs: readonly EpisodeSpec[],
  heldoutSpecs: readonly EpisodeSpec[]
): void {
  const fullKeys = new Set<string>(fullSpecs.map(comboKey));
  for (const spec of heldoutSpecs) {
    const key = comboKey(spec);
    if (fullKeys.has(key)) {
      throw new Error(
        `Heldout episode ${spec.index} (${key}) overlaps with the full swell grid`
      );
    }
  }
}

interface ManifestEpisodeRecord {
  file: string;
  index: number;
  kind: BathymetryKind;
  seed: number;
  swell: SwellParams;
  contentHash: string;
  sizeBytes: number;
  wallS: number;
}

interface DatasetManifest {
  dataset: string;
  seedRule: string;
  episodeCount: number;
  totalBytes: number;
  totalWallS: number;
  episodes: ManifestEpisodeRecord[];
}

function validateWrittenEpisode(filePath: string, spec: EpisodeSpec): string {
  const parsed = readEpisodeFile(filePath);
  const { contentHash, episode } = parsed.header;

  if (typeof contentHash !== 'string' || !SHA256_HEX_RE.test(contentHash)) {
    throw new Error(
      `Invalid contentHash in ${filePath}: expected 64-char hex SHA-256, got '${String(contentHash)}'`
    );
  }

  if (
    episode.index !== spec.index ||
    episode.kind !== spec.kind ||
    episode.seed !== spec.seed ||
    episode.durationS !== spec.durationS ||
    episode.swell.heightM !== spec.swell.heightM ||
    episode.swell.periodS !== spec.swell.periodS ||
    episode.swell.directionDeg !== spec.swell.directionDeg
  ) {
    throw new Error(
      `Header episode metadata mismatch in ${filePath}: ` +
        `got ${JSON.stringify(episode)}, expected ${JSON.stringify(spec)}`
    );
  }

  return contentHash;
}

export function runDataset(
  specs: readonly EpisodeSpec[],
  outDirName: string,
  datasetName: string
): void {
  const outDir = `${process.cwd()}/data/${outDirName}`;
  mkdirSync(outDir, { recursive: true });

  const totalStart = performance.now();
  let totalBytes = 0;
  const episodes: ManifestEpisodeRecord[] = [];

  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i];
    const file = episodeFileName(spec);
    const filePath = `${outDir}/${file}`;

    let contentHash: string;
    let sizeBytes: number;
    let wallS: number;

    try {
      const t0 = performance.now();
      const data = generateEpisode(spec);
      const bytes = serializeEpisode(data);
      writeFileSync(filePath, bytes);
      contentHash = validateWrittenEpisode(filePath, spec);
      wallS = (performance.now() - t0) / 1000;
      sizeBytes = statSync(filePath).size;
    } catch (err) {
      console.error(
        `[${datasetName}][ep${String(spec.index).padStart(2, '0')}] FAILED (${file}):`,
        err
      );
      throw err;
    }

    totalBytes += sizeBytes;
    episodes.push({
      file,
      index: spec.index,
      kind: spec.kind,
      seed: spec.seed,
      swell: { ...spec.swell },
      contentHash,
      sizeBytes,
      wallS,
    });

    console.log(
      `[${datasetName}][ep${String(spec.index).padStart(2, '0')}] (${i + 1}/${specs.length}) ` +
        `kind=${spec.kind} seed=${spec.seed} ` +
        `H=${spec.swell.heightM}m T=${spec.swell.periodS}s dir=${spec.swell.directionDeg}deg ` +
        `wall=${wallS.toFixed(2)}s size=${(sizeBytes / 1048576).toFixed(2)} MiB ` +
        `sha256=${contentHash}`
    );
  }

  const totalWallS = (performance.now() - totalStart) / 1000;
  const manifest: DatasetManifest = {
    dataset: datasetName,
    seedRule: SEED_RULES[datasetName] ?? 'custom',
    episodeCount: episodes.length,
    totalBytes,
    totalWallS,
    episodes,
  };
  writeFileSync(
    `${outDir}/manifest.json`,
    new TextEncoder().encode(`${JSON.stringify(manifest, null, 2)}\n`)
  );

  console.log(
    `[${datasetName}] total: ${(totalBytes / 1048576).toFixed(2)} MiB across ` +
      `${episodes.length} episodes in ${totalWallS.toFixed(2)}s`
  );
}

assertHeldoutDisjoint(FULL_EPISODES, HELDOUT_EPISODES);
runDataset(FULL_EPISODES, 'full', 'full');
runDataset(HELDOUT_EPISODES, 'heldout', 'heldout');
