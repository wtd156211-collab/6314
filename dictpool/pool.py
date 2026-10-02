"""Train a shared dictionary pool and load it for (un)packing."""

import os
from collections import Counter

from . import common
from .common import (
    BATCH_SAMPLE_CAP,
    BLOCK_SAMPLE_BYTES,
    DICT_MAX_BYTES,
    FRAGMENT_LEN,
    MIN_FRAGMENT_COUNT,
    POOL_CAPACITY,
    DataValidationError,
    UsageError,
    VersionMismatchError,
    atomic_write,
    block_id,
    dump_json,
    list_block_files,
    sha256_hex,
)

POOL_JSON = "pool.json"


def sample_batch(blocks_dir):
    """Sample one training batch (README section 2, step 1).

    Heads of blocks in file-name byte order; accumulation stops at the
    per-batch cap and the final segment may be cut.
    """
    segments = []
    remaining = BATCH_SAMPLE_CAP
    for name in list_block_files(blocks_dir):
        if remaining <= 0:
            break
        path = os.path.join(blocks_dir, name)
        with open(path, "rb") as f:
            head = f.read(min(BLOCK_SAMPLE_BYTES, remaining))
        if head:
            segments.append(head)
            remaining -= len(head)
    return segments


def train_dictionary(segments):
    """Count windows per segment, select fragments, build dict bytes."""
    counts = Counter()
    for seg in segments:
        for i in range(len(seg) - FRAGMENT_LEN + 1):
            counts[seg[i : i + FRAGMENT_LEN]] += 1
    candidates = [
        (count, frag) for frag, count in counts.items() if count >= MIN_FRAGMENT_COUNT
    ]
    # Ascending (count, bytes); take the tail, keep that order.
    candidates.sort(key=lambda item: (item[0], item[1]))
    limit = DICT_MAX_BYTES // FRAGMENT_LEN
    selected = candidates[-limit:]
    return b"".join(frag for _, frag in selected)


def train(batch_dirs, out_dir):
    """Train one dictionary per batch (index follows CLI batch order)."""
    if not batch_dirs:
        raise UsageError("train requires at least one --blocks directory")
    if len(batch_dirs) > POOL_CAPACITY:
        raise UsageError(
            "pool capacity is %d dictionaries, got %d batches"
            % (POOL_CAPACITY, len(batch_dirs))
        )

    dicts = []
    for index, blocks_dir in enumerate(batch_dirs):
        batch = common.batch_name(blocks_dir)
        segments = sample_batch(blocks_dir)
        dict_bytes = train_dictionary(segments)
        if len(dict_bytes) > DICT_MAX_BYTES:
            raise DataValidationError("dictionary exceeds %d bytes" % DICT_MAX_BYTES)
        dicts.append(
            {
                "index": index,
                "batch": batch,
                "sample_bytes": sum(len(s) for s in segments),
                "dict_bytes": len(dict_bytes),
                "sha256": sha256_hex(dict_bytes),
                "_bytes": dict_bytes,
            }
        )

    pool_id = sha256_hex("".join(d["sha256"] for d in dicts).encode("ascii"))
    write_pool(out_dir, dicts, pool_id)
    return pool_id, dicts


def write_pool(out_dir, dicts, pool_id):
    os.makedirs(out_dir, exist_ok=True)
    # Remove stale dictionary files so a retrained pool cannot mix versions.
    for name in list(list_dir_names(out_dir)):
        if name.startswith("d") and name.endswith(".bin"):
            try:
                os.unlink(os.path.join(out_dir, name))
            except OSError:
                pass
    payload = [
        {
            "index": d["index"],
            "batch": d["batch"],
            "sample_bytes": d["sample_bytes"],
            "dict_bytes": d["dict_bytes"],
            "sha256": d["sha256"],
        }
        for d in dicts
    ]
    atomic_write(
        os.path.join(out_dir, POOL_JSON),
        dump_json({"pool_id": pool_id, "dictionaries": payload}),
    )
    for d in dicts:
        atomic_write(
            os.path.join(out_dir, "d%d.bin" % d["index"]), d["_bytes"]
        )


def list_dir_names(directory):
    try:
        return os.listdir(directory)
    except FileNotFoundError:
        return []


def load_pool(pool_dir, expected_pool_id=None):
    """Load and fully verify a pool.

    pool.json is checked for self-consistency and each dN.bin against its
    sha256. If expected_pool_id is given it must match; a mismatch is a
    version error (exit 2), never a silent dictionary swap.
    """
    import json

    path = os.path.join(pool_dir, POOL_JSON)
    if not os.path.isfile(path):
        raise UsageError("pool not found: %s" % path)
    with open(path, "rb") as f:
        raw = f.read()
    try:
        meta = json.loads(raw.decode("utf-8"))
        entries = meta["dictionaries"]
        pool_id = meta["pool_id"]
    except (ValueError, KeyError, UnicodeDecodeError):
        raise DataValidationError("malformed pool index: %s" % path)

    if not isinstance(pool_id, str) or not isinstance(entries, list):
        raise DataValidationError("malformed pool index: %s" % path)
    if not entries:
        raise DataValidationError("pool has no dictionaries: %s" % path)
    if len(entries) > POOL_CAPACITY:
        raise DataValidationError("pool exceeds capacity of %d" % POOL_CAPACITY)

    expected_indexes = list(range(len(entries)))
    entries.sort(key=lambda d: d.get("index", -1))
    if [d.get("index") for d in entries] != expected_indexes:
        raise DataValidationError("pool dictionary indexes must be 0..n-1")

    hexes = []
    dictionaries = []
    for entry in entries:
        try:
            index = entry["index"]
            sha = entry["sha256"]
            size = entry["dict_bytes"]
        except KeyError:
            raise DataValidationError("malformed pool entry in %s" % path)
        with open(os.path.join(pool_dir, "d%d.bin" % index), "rb") as f:
            data = f.read()
        actual = sha256_hex(data)
        if actual != sha:
            raise VersionMismatchError(
                "dictionary d%d sha256 mismatch: pool says %s, file is %s"
                % (index, sha, actual)
            )
        if len(data) != size:
            raise DataValidationError(
                "dictionary d%d size mismatch: %s vs %s"
                % (index, size, len(data))
            )
        if len(data) > DICT_MAX_BYTES:
            raise DataValidationError("dictionary d%d exceeds size cap" % index)
        hexes.append(sha)
        dictionaries.append(
            {"index": index, "batch": entry.get("batch"), "bytes": data}
        )

    computed = sha256_hex("".join(hexes).encode("ascii"))
    if computed != pool_id:
        raise VersionMismatchError(
            "pool_id mismatch: index says %s, dictionaries hash to %s"
            % (pool_id, computed)
        )
    if expected_pool_id is not None and expected_pool_id != pool_id:
        raise VersionMismatchError(
            "archive needs pool %s, provided pool is %s"
            % (expected_pool_id, pool_id)
        )

    windows = [common.dict_windows(d["bytes"]) for d in dictionaries]
    return {"pool_id": pool_id, "dictionaries": dictionaries, "windows": windows}
