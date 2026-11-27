/**
 * Binary container codec for S07e surf-world episodes.
 *
 * Layout:
 *   bytes 0..3                  ASCII magic 'SWE1'
 *   bytes 4..7                  uint32 LE header JSON byte length (headerLen)
 *   bytes 8 .. 8+headerLen-1    header JSON (UTF-8)
 *   bytes 8+headerLen .. end    payload = concatenated raw field bytes in
 *                               FIELD_ORDER, little-endian (native typed-array
 *                               byte order on x86/ARM LE platforms)
 *
 * contentHash is the lowercase SHA-256 hex digest of the payload bytes only.
 */

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import {
  DTYPE_SIZES,
  EPISODE_FORMAT,
  EPISODE_FORMAT_VERSION,
  FIELD_ORDER,
  fieldShape,
} from './contract.ts';
import type { EpisodeData, EpisodeHeader, FieldDType, FieldSpec } from './contract.ts';

const MAGIC = [0x53, 0x57, 0x45, 0x31] as const; // 'SWE1'

const FIELD_DTYPES: Record<string, FieldDType> = {
  depth: 'f32',
  eta: 'f32',
  u: 'f32',
  v: 'f32',
  timeMaxAbsEta: 'f32',
  onsetMask: 'u8',
  onsetRow: 'i16',
  onsetConfidence: 'f32',
  crestArrivalS: 'f32',
};

type FieldTypedArray = Float32Array | Uint8Array | Int16Array;

function getFieldArray(data: EpisodeData, name: string): FieldTypedArray {
  switch (name) {
    case 'depth':
      return data.depth;
    case 'eta':
      return data.eta;
    case 'u':
      return data.u;
    case 'v':
      return data.v;
    case 'timeMaxAbsEta':
      return data.timeMaxAbsEta;
    case 'onsetMask':
      return data.onsetMask;
    case 'onsetRow':
      return data.onsetRow;
    case 'onsetConfidence':
      return data.onsetConfidence;
    case 'crestArrivalS':
      return data.crestArrivalS;
    default:
      throw new Error(`Unknown episode field: ${name}`);
  }
}

export function buildFieldSpecs(
  width: number,
  height: number,
  numFrames: number
): FieldSpec[] {
  const specs: FieldSpec[] = [];
  let offset = 0;

  for (const name of FIELD_ORDER) {
    const dtype = FIELD_DTYPES[name];
    if (!dtype) {
      throw new Error(`Missing dtype mapping for field: ${name}`);
    }
    const shape = fieldShape(name, width, height, numFrames);
    let count = 1;
    for (const dim of shape) count *= dim;
    const byteLength = count * DTYPE_SIZES[dtype];
    specs.push({ name, dtype, shape, offset, byteLength });
    offset += byteLength;
  }

  return specs;
}

export function serializeEpisode(data: EpisodeData): Uint8Array {
  const { width, height } = data.header.grid;
  const { numFrames } = data.header.episode;
  const specs = buildFieldSpecs(width, height, numFrames);

  let totalPayloadBytes = 0;
  for (const spec of specs) totalPayloadBytes += spec.byteLength;

  const payload = new Uint8Array(totalPayloadBytes);
  for (const spec of specs) {
    const typed = getFieldArray(data, spec.name);
    const src = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
    if (src.byteLength !== spec.byteLength) {
      throw new Error(
        `Field ${spec.name} byteLength mismatch: got ${src.byteLength}, expected ${spec.byteLength}`
      );
    }
    payload.set(src, spec.offset);
  }

  const contentHash = createHash('sha256').update(payload).digest('hex');
  const headerObj: EpisodeHeader = {
    ...data.header,
    contentHash,
    fields: specs,
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(headerObj));

  const out = new Uint8Array(8 + headerBytes.byteLength + payload.byteLength);
  out[0] = MAGIC[0];
  out[1] = MAGIC[1];
  out[2] = MAGIC[2];
  out[3] = MAGIC[3];
  new DataView(out.buffer, out.byteOffset, 8).setUint32(4, headerBytes.byteLength, true);
  out.set(headerBytes, 8);
  out.set(payload, 8 + headerBytes.byteLength);

  return out;
}

function makeTypedView(
  payloadCopy: Uint8Array,
  spec: FieldSpec
): FieldTypedArray {
  const elemSize = DTYPE_SIZES[spec.dtype];
  const count = spec.byteLength / elemSize;
  // Copy the field bytes into a fresh, exactly-sized buffer so the view is
  // always aligned and detached. payloadCopy may be a view into a larger
  // Buffer (node:fs readFileSync) with an arbitrary byteOffset, where taking
  // .buffer of a sub-slice would silently point at the wrong bytes.
  const fieldBytes = new Uint8Array(spec.byteLength);
  fieldBytes.set(payloadCopy.subarray(spec.offset, spec.offset + spec.byteLength));

  switch (spec.dtype) {
    case 'f32':
      return new Float32Array(fieldBytes.buffer, 0, count);
    case 'u8':
      return fieldBytes;
    case 'i16':
      return new Int16Array(fieldBytes.buffer, 0, count);
  }
}

export function parseEpisode(bytes: Uint8Array): EpisodeData {
  if (bytes.byteLength < 8) {
    throw new Error(`Episode buffer too short: ${bytes.byteLength} bytes (need at least 8)`);
  }
  if (
    bytes[0] !== MAGIC[0] ||
    bytes[1] !== MAGIC[1] ||
    bytes[2] !== MAGIC[2] ||
    bytes[3] !== MAGIC[3]
  ) {
    throw new Error(
      `Invalid episode magic: expected 'SWE1', got [${bytes[0]}, ${bytes[1]}, ${bytes[2]}, ${bytes[3]}]`
    );
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = view.getUint32(4, true);
  if (headerLen <= 0 || headerLen > bytes.byteLength - 8) {
    throw new Error(
      `Invalid episode header length: ${headerLen} (buffer size ${bytes.byteLength})`
    );
  }

  let header: EpisodeHeader;
  try {
    const headerJson = new TextDecoder('utf-8', { fatal: true }).decode(
      bytes.subarray(8, 8 + headerLen)
    );
    header = JSON.parse(headerJson) as EpisodeHeader;
  } catch (err) {
    throw new Error(
      `Failed to parse episode header JSON: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  if (!header || typeof header !== 'object') {
    throw new Error('Episode header JSON must be an object');
  }
  if (header.format !== EPISODE_FORMAT) {
    throw new Error(
      `Unexpected episode format: got '${String(header.format)}', expected '${EPISODE_FORMAT}'`
    );
  }
  if (header.formatVersion !== EPISODE_FORMAT_VERSION) {
    throw new Error(
      `Unexpected episode formatVersion: got ${String(header.formatVersion)}, expected ${EPISODE_FORMAT_VERSION}`
    );
  }

  if (
    !header.grid ||
    !Number.isInteger(header.grid.width) ||
    header.grid.width <= 0 ||
    !Number.isInteger(header.grid.height) ||
    header.grid.height <= 0
  ) {
    throw new Error('Episode header has invalid grid dimensions');
  }
  if (
    !header.episode ||
    !Number.isInteger(header.episode.numFrames) ||
    header.episode.numFrames <= 0
  ) {
    throw new Error('Episode header has invalid numFrames');
  }

  const expectedSpecs = buildFieldSpecs(
    header.grid.width,
    header.grid.height,
    header.episode.numFrames
  );

  if (!Array.isArray(header.fields) || header.fields.length !== expectedSpecs.length) {
    throw new Error(
      `Episode header fields count mismatch: got ${Array.isArray(header.fields) ? header.fields.length : 'non-array'}, expected ${expectedSpecs.length}`
    );
  }

  let expectedPayloadBytes = 0;
  for (let i = 0; i < expectedSpecs.length; i++) {
    const got = header.fields[i];
    const exp = expectedSpecs[i];
    if (
      !got ||
      got.name !== exp.name ||
      got.dtype !== exp.dtype ||
      got.offset !== exp.offset ||
      got.byteLength !== exp.byteLength ||
      !Array.isArray(got.shape) ||
      got.shape.length !== exp.shape.length ||
      got.shape.some((d, j) => d !== exp.shape[j])
    ) {
      throw new Error(
        `Episode FieldSpec mismatch at index ${i} (${exp.name}): got ${JSON.stringify(got)}, expected ${JSON.stringify(exp)}`
      );
    }
    expectedPayloadBytes += exp.byteLength;
  }

  const payloadStart = 8 + headerLen;
  const actualPayloadBytes = bytes.byteLength - payloadStart;
  if (actualPayloadBytes !== expectedPayloadBytes) {
    throw new Error(
      `Episode payload length mismatch: got ${actualPayloadBytes} bytes, expected ${expectedPayloadBytes} bytes`
    );
  }

  // Copy payload into a fresh ArrayBuffer so typed-array views are aligned and detached.
  const payloadCopy = bytes.slice(payloadStart);
  const actualHash = createHash('sha256').update(payloadCopy).digest('hex');
  if (actualHash !== header.contentHash) {
    throw new Error(
      `Episode contentHash mismatch: computed ${actualHash}, header has ${header.contentHash}`
    );
  }

  const fieldMap: Record<string, FieldTypedArray> = {};
  for (const spec of expectedSpecs) {
    fieldMap[spec.name] = makeTypedView(payloadCopy, spec);
  }

  return {
    header,
    depth: fieldMap.depth as Float32Array,
    eta: fieldMap.eta as Float32Array,
    u: fieldMap.u as Float32Array,
    v: fieldMap.v as Float32Array,
    timeMaxAbsEta: fieldMap.timeMaxAbsEta as Float32Array,
    onsetMask: fieldMap.onsetMask as Uint8Array,
    onsetRow: fieldMap.onsetRow as Int16Array,
    onsetConfidence: fieldMap.onsetConfidence as Float32Array,
    crestArrivalS: fieldMap.crestArrivalS as Float32Array,
  };
}

export function writeEpisodeFile(path: string, data: EpisodeData): void {
  writeFileSync(path, serializeEpisode(data));
}

export function readEpisodeFile(path: string): EpisodeData {
  return parseEpisode(readFileSync(path));
}
