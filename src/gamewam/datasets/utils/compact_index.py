"""Exact mmap-backed lookup for large string-to-record JSON mappings.

The precompute index format remains the source of truth.  A content-validated
sidecar is built lazily next to it and reused by all local ranks.  Hashes only
locate candidates: every successful lookup verifies the original UTF-8 key,
so hash collisions cannot change cache semantics.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


_FORMAT_ID = "blake2b128-record-key-offsets"
_HASH_DTYPE = np.dtype([("hi", "<u8"), ("lo", "<u8")])
_MISSING = object()


def _source_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "inode": int(stat.st_ino),
    }


def _digest_parts(key_bytes: bytes) -> tuple[int, int]:
    digest = hashlib.blake2b(key_bytes, digest_size=16).digest()
    return (
        int.from_bytes(digest[:8], "little", signed=False),
        int.from_bytes(digest[8:], "little", signed=False),
    )


def _sidecar_paths(index_path: Path) -> dict[str, Path]:
    prefix = Path(f"{index_path}.compact-{_FORMAT_ID}")
    return {
        "manifest": Path(f"{prefix}.json"),
        "hashes": Path(f"{prefix}.hashes.npy"),
        "records": Path(f"{prefix}.records.npy"),
        "key_indices": Path(f"{prefix}.key_indices.npy"),
        "offsets": Path(f"{prefix}.offsets.npy"),
        "keys": Path(f"{prefix}.keys.bin"),
        "lock": Path(f"{prefix}.lock"),
    }


@contextlib.contextmanager
def _exclusive_file_lock(path: Path):
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class CompactStringIntMapping(Mapping[str, int]):
    """Read-only exact mapping backed by shared mmap files."""

    def __init__(
        self,
        *,
        hashes_path: Path,
        records_path: Path,
        key_indices_path: Path,
        offsets_path: Path,
        keys_path: Path,
        count: int,
    ) -> None:
        self._hashes = np.load(hashes_path, mmap_mode="r")
        self._records = np.load(records_path, mmap_mode="r")
        self._key_indices = np.load(key_indices_path, mmap_mode="r")
        self._offsets = np.load(offsets_path, mmap_mode="r")
        self._key_bytes = (
            np.memmap(keys_path, dtype=np.uint8, mode="r")
            if keys_path.stat().st_size > 0
            else np.empty(0, dtype=np.uint8)
        )
        self._count = int(count)
        if (
            self._hashes.dtype != _HASH_DTYPE
            or self._hashes.shape != (self._count,)
            or self._records.dtype != np.dtype(np.uint64)
            or self._records.shape != (self._count,)
            or self._key_indices.dtype != np.dtype(np.uint64)
            or self._key_indices.shape != (self._count,)
            or self._offsets.dtype != np.dtype(np.uint64)
            or self._offsets.shape != (self._count + 1,)
        ):
            raise ValueError("Compact cache-index sidecar has inconsistent arrays.")
        if self._count < 0:
            raise ValueError("Compact cache-index count cannot be negative.")
        if int(self._offsets[0]) != 0 or int(self._offsets[-1]) != int(self._key_bytes.size):
            raise ValueError("Compact cache-index key offsets are inconsistent.")

    def __len__(self) -> int:
        return self._count

    def _key_at_original_index(self, key_index: int) -> bytes:
        start = int(self._offsets[key_index])
        end = int(self._offsets[key_index + 1])
        return self._key_bytes[start:end].tobytes()

    def get(self, key: str, default=None):
        if not isinstance(key, str):
            return default
        key_bytes = key.encode("utf-8")
        hi, lo = _digest_parts(key_bytes)
        needle = np.array((hi, lo), dtype=_HASH_DTYPE)[()]
        left = int(np.searchsorted(self._hashes, needle, side="left"))
        right = int(np.searchsorted(self._hashes, needle, side="right"))
        for position in range(left, right):
            key_index = int(self._key_indices[position])
            if not 0 <= key_index < self._count:
                raise ValueError("Compact cache-index key index is out of bounds.")
            if self._key_at_original_index(key_index) == key_bytes:
                return int(self._records[position])
        return default

    def get_many(
        self,
        keys: Sequence[object],
        default=None,
    ) -> list[Any]:
        """Look up a bounded batch while retaining exact-key semantics.

        Hashing and binary-search bounds are vectorized, but each candidate
        range is still verified against the original UTF-8 key. Callers
        should pass bounded chunks so temporary encoded keys stay small.
        """
        key_count = len(keys)
        if key_count == 0:
            return []

        encoded: list[bytes | None] = [None] * key_count
        needles = np.zeros(key_count, dtype=_HASH_DTYPE)
        valid = np.zeros(key_count, dtype=np.bool_)
        for index, key in enumerate(keys):
            if not isinstance(key, str):
                continue
            key_bytes = key.encode("utf-8")
            encoded[index] = key_bytes
            needles[index] = _digest_parts(key_bytes)
            valid[index] = True

        results: list[Any] = [default] * key_count
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            return results
        valid_needles = needles[valid_indices]
        left_bounds = np.searchsorted(
            self._hashes, valid_needles, side="left"
        )
        right_bounds = np.searchsorted(
            self._hashes, valid_needles, side="right"
        )
        for result_index, left, right in zip(
            valid_indices, left_bounds, right_bounds
        ):
            key_bytes = encoded[int(result_index)]
            for position in range(int(left), int(right)):
                key_index = int(self._key_indices[position])
                if not 0 <= key_index < self._count:
                    raise ValueError(
                        "Compact cache-index key index is out of bounds."
                    )
                if self._key_at_original_index(key_index) == key_bytes:
                    results[int(result_index)] = int(self._records[position])
                    break
        return results

    def __getitem__(self, key: str) -> int:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return int(value)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get(key, _MISSING) is not _MISSING

    def __iter__(self) -> Iterator[str]:
        for key_index in range(self._count):
            yield self._key_at_original_index(key_index).decode("utf-8")


def _load_manifest(
    index_path: Path,
    paths: dict[str, Path],
) -> tuple[dict[str, Any], CompactStringIntMapping] | None:
    manifest_path = paths["manifest"]
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if (
            manifest.get("format_id") != _FORMAT_ID
            or manifest.get("source_signature") != _source_signature(index_path)
        ):
            return None
        array_names = ("hashes", "records", "key_indices", "offsets", "keys")
        file_sizes = manifest.get("file_sizes")
        if not isinstance(file_sizes, dict):
            return None
        for name in array_names:
            if (
                not paths[name].is_file()
                or int(file_sizes.get(name, -1)) != int(paths[name].stat().st_size)
            ):
                return None
        count = int(manifest["count"])
        mapping = CompactStringIntMapping(
            hashes_path=paths["hashes"],
            records_path=paths["records"],
            key_indices_path=paths["key_indices"],
            offsets_path=paths["offsets"],
            keys_path=paths["keys"],
            count=count,
        )
        metadata = manifest["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Compact cache-index metadata must be an object.")
        return metadata, mapping
    except (
        OSError, ValueError, TypeError, OverflowError, EOFError, KeyError,
        json.JSONDecodeError,
    ):
        return None


def _open_array(path: Path, dtype, shape):
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def _build_sidecar(
    index_path: Path,
    paths: dict[str, Path],
) -> tuple[dict[str, Any], CompactStringIntMapping]:
    source_before = _source_signature(index_path)
    with open(index_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    mapping = payload.pop("mapping", None)
    if not isinstance(mapping, dict):
        raise ValueError(f"Cache index {index_path} has no mapping object.")

    count = len(mapping)
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{index_path.name}.compact-{_FORMAT_ID}.",
            dir=index_path.parent,
        )
    )
    try:
        unsorted_hashes = _open_array(
            temp_dir / "unsorted_hashes.npy", _HASH_DTYPE, (count,)
        )
        unsorted_records = _open_array(
            temp_dir / "unsorted_records.npy", np.uint64, (count,)
        )
        offsets = _open_array(
            temp_dir / "offsets.npy", np.uint64, (count + 1,)
        )
        keys_path = temp_dir / "keys.bin"
        offsets[0] = 0
        with open(keys_path, "wb") as key_stream:
            for key_index, (key, record_index) in enumerate(mapping.items()):
                if not isinstance(key, str):
                    raise ValueError("Cache mapping keys must be strings.")
                record_index = int(record_index)
                if record_index < 0:
                    raise ValueError("Cache mapping record indices must be non-negative.")
                key_bytes = key.encode("utf-8")
                unsorted_hashes[key_index] = _digest_parts(key_bytes)
                unsorted_records[key_index] = record_index
                key_stream.write(key_bytes)
                offsets[key_index + 1] = offsets[key_index] + len(key_bytes)
        # The expanded JSON dictionary is not needed during the large
        # argsort/gather phase; release it before allocating the sort order.
        del mapping
        unsorted_hashes.flush()
        unsorted_records.flush()
        offsets.flush()

        order = np.argsort(
            unsorted_hashes,
            order=("hi", "lo"),
            kind="stable",
        )
        hashes = _open_array(temp_dir / "hashes.npy", _HASH_DTYPE, (count,))
        records = _open_array(temp_dir / "records.npy", np.uint64, (count,))
        key_indices = _open_array(
            temp_dir / "key_indices.npy", np.uint64, (count,)
        )
        hashes[:] = unsorted_hashes[order]
        records[:] = unsorted_records[order]
        key_indices[:] = order
        hashes.flush()
        records.flush()
        key_indices.flush()
        del order
        del hashes, records, key_indices
        del unsorted_hashes, unsorted_records, offsets

        if _source_signature(index_path) != source_before:
            raise RuntimeError(f"Cache index changed while compacting: {index_path}")

        replacements = {
            "hashes": temp_dir / "hashes.npy",
            "records": temp_dir / "records.npy",
            "key_indices": temp_dir / "key_indices.npy",
            "offsets": temp_dir / "offsets.npy",
            "keys": keys_path,
        }
        file_sizes = {
            name: int(temp_path.stat().st_size)
            for name, temp_path in replacements.items()
        }
        # Unlink the manifest first so unlocked readers never observe a
        # valid-looking manifest that refers to a mixture of generations.
        paths["manifest"].unlink(missing_ok=True)
        for name, temp_path in replacements.items():
            os.replace(temp_path, paths[name])

        manifest = {
            "format_id": _FORMAT_ID,
            "source_signature": source_before,
            "count": count,
            "file_sizes": file_sizes,
            "metadata": payload,
        }
        temp_manifest = temp_dir / "manifest.json"
        with open(temp_manifest, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False)
        os.replace(temp_manifest, paths["manifest"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    loaded = _load_manifest(index_path, paths)
    if loaded is None:
        raise RuntimeError(f"Failed to validate compact cache index for {index_path}")
    return loaded


def load_compact_json_index(
    index_path: str | Path,
) -> tuple[dict[str, Any], CompactStringIntMapping]:
    """Load or build an exact compact view of a JSON cache index.

    The manifest is published last, making readers insensitive to an
    interrupted builder.  A file lock ensures only one local rank expands the
    original large JSON mapping.
    """

    index_path = Path(index_path)
    paths = _sidecar_paths(index_path)
    loaded = _load_manifest(index_path, paths)
    if loaded is not None:
        return loaded
    with _exclusive_file_lock(paths["lock"]):
        loaded = _load_manifest(index_path, paths)
        if loaded is not None:
            return loaded
        return _build_sidecar(index_path, paths)
