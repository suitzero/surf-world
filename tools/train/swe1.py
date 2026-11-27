"""SWE1 binary episode format reader for surf-world."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MAGIC = b"SWE1"

DTYPE_MAP: dict[str, np.dtype] = {
    "f32": np.dtype("<f4"),
    "u8": np.dtype("u1"),
    "i16": np.dtype("<i2"),
}


def load_manifest(data_dir: str | Path) -> dict[str, Any]:
    """Load and parse manifest.json from a dataset directory."""
    manifest_path = Path(data_dir) / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_header(path: str | Path) -> tuple[dict[str, Any], int]:
    """Read and parse only the SWE1 header JSON, returning (header, payload_offset)."""
    file_path = Path(path)
    with file_path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise ValueError(f"Episode file too short ({len(prefix)} bytes): {file_path}")
        if prefix[:4] != MAGIC:
            raise ValueError(
                f"Invalid SWE1 magic {prefix[:4]!r} in {file_path}; expected {MAGIC!r}"
            )
        (header_len,) = struct.unpack("<I", prefix[4:8])
        if header_len <= 0:
            raise ValueError(f"Invalid SWE1 header length {header_len} in {file_path}")
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(
                f"Truncated SWE1 header in {file_path}: expected {header_len} bytes, got {len(header_bytes)}"
            )
    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"SWE1 header JSON must be an object in {file_path}")
    return header, 8 + header_len


def _decode_field(payload: memoryview | bytes, spec: dict[str, Any]) -> np.ndarray:
    name = spec["name"]
    dtype_str = spec["dtype"]
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {dtype_str!r} for field {name!r}")
    dtype = DTYPE_MAP[dtype_str]
    shape = tuple(int(d) for d in spec["shape"])
    offset = int(spec["offset"])
    byte_length = int(spec["byteLength"])

    count = math.prod(shape)
    expected_bytes = count * dtype.itemsize
    if byte_length != expected_bytes:
        raise ValueError(
            f"Field {name!r} byteLength mismatch: header has {byte_length}, "
            f"shape {shape} with dtype {dtype_str} requires {expected_bytes}"
        )
    if offset < 0 or offset + byte_length > len(payload):
        raise ValueError(
            f"Field {name!r} range [{offset}, {offset + byte_length}) exceeds "
            f"payload size {len(payload)}"
        )

    return np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape).copy()


def load_episode(
    path: str | Path,
    fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Load an SWE1 episode file, verify payload SHA-256, and return header + numpy arrays."""
    file_path = Path(path)
    raw = file_path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"Episode buffer too short ({len(raw)} bytes): {file_path}")
    if raw[:4] != MAGIC:
        raise ValueError(
            f"Invalid SWE1 magic {raw[:4]!r} in {file_path}; expected {MAGIC!r}"
        )

    (header_len,) = struct.unpack_from("<I", raw, 4)
    if header_len <= 0 or header_len > len(raw) - 8:
        raise ValueError(
            f"Invalid SWE1 header length {header_len} for file size {len(raw)}: {file_path}"
        )

    header_end = 8 + header_len
    header = json.loads(raw[8:header_end].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"SWE1 header JSON must be an object in {file_path}")

    payload = memoryview(raw)[header_end:]
    actual_hash = hashlib.sha256(payload).hexdigest()
    expected_hash = header.get("contentHash")
    if actual_hash != expected_hash:
        raise ValueError(
            f"SWE1 contentHash mismatch in {file_path}: "
            f"computed {actual_hash}, header has {expected_hash}"
        )

    field_specs = header.get("fields")
    if not isinstance(field_specs, list):
        raise ValueError(f"SWE1 header missing valid 'fields' list in {file_path}")

    wanted = set(fields) if fields is not None else None
    result: dict[str, Any] = {"header": header}
    for spec in field_specs:
        name = spec["name"]
        if wanted is not None and name not in wanted:
            continue
        result[name] = _decode_field(payload, spec)

    if wanted is not None:
        missing = wanted.difference(result.keys())
        if missing:
            raise KeyError(f"Requested fields not found in {file_path}: {sorted(missing)}")

    return result


def load_episode_fields_fast(
    path: str | Path,
    header: dict[str, Any],
    payload_offset: int,
    fields: Iterable[str],
) -> dict[str, np.ndarray]:
    """Read selected fields directly via seek for an episode whose hash was already verified."""
    file_path = Path(path)
    wanted = set(fields)
    specs_by_name = {spec["name"]: spec for spec in header["fields"]}
    missing = wanted.difference(specs_by_name.keys())
    if missing:
        raise KeyError(f"Requested fields not found in {file_path}: {sorted(missing)}")

    arrays: dict[str, np.ndarray] = {}
    with file_path.open("rb") as f:
        for name in wanted:
            spec = specs_by_name[name]
            dtype_str = spec["dtype"]
            if dtype_str not in DTYPE_MAP:
                raise ValueError(f"Unsupported dtype {dtype_str!r} for field {name!r}")
            dtype = DTYPE_MAP[dtype_str]
            shape = tuple(int(d) for d in spec["shape"])
            offset = int(spec["offset"])
            byte_length = int(spec["byteLength"])
            count = math.prod(shape)
            if byte_length != count * dtype.itemsize:
                raise ValueError(f"Field {name!r} byteLength mismatch in {file_path}")

            f.seek(payload_offset + offset)
            buf = f.read(byte_length)
            if len(buf) != byte_length:
                raise ValueError(
                    f"Truncated read for field {name!r} in {file_path}: "
                    f"expected {byte_length} bytes, got {len(buf)}"
                )
            arrays[name] = np.frombuffer(buf, dtype=dtype, count=count).reshape(shape).copy()

    return arrays
