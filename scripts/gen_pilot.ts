import { mkdirSync, statSync, writeFileSync } from 'node:fs';
import { PILOT_EPISODES, episodeFileName } from '../src/data/contract.ts';
import { serializeEpisode } from '../src/data/episode.ts';
import { generateEpisode } from '../src/data/generate.ts';

declare const process: { cwd(): string };

const outDir = `${process.cwd()}/data/pilot`;
mkdirSync(outDir, { recursive: true });

const totalStart = performance.now();
let totalBytes = 0;

for (const spec of PILOT_EPISODES) {
  const t0 = performance.now();
  const data = generateEpisode(spec);
  const bytes = serializeEpisode(data);
  const filePath = `${outDir}/${episodeFileName(spec)}`;
  writeFileSync(filePath, bytes);
  const elapsedS = (performance.now() - t0) / 1000;

  const sizeBytes = statSync(filePath).size;
  totalBytes += sizeBytes;

  const headerLen = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(
    4,
    true
  );
  const header = JSON.parse(
    new TextDecoder().decode(bytes.subarray(8, 8 + headerLen))
  ) as { contentHash: string };

  console.log(
    `[ep${String(spec.index).padStart(2, '0')}] kind=${spec.kind} seed=${spec.seed} ` +
      `wall=${elapsedS.toFixed(2)}s size=${(sizeBytes / 1048576).toFixed(2)} MiB ` +
      `sha256=${header.contentHash}`
  );
}

const totalS = (performance.now() - totalStart) / 1000;
console.log(
  `[pilot] total: ${(totalBytes / 1048576).toFixed(2)} MiB across ${PILOT_EPISODES.length} episodes in ${totalS.toFixed(2)}s`
);
