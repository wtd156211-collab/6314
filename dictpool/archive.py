"""Pack blocks with the pool and unpack them back, byte for byte."""

import os
import struct
import zlib

from . import common
from .common import (
    COVERAGE_THRESHOLD,
    DPK_HEADER_LEN,
    DPK_MAGIC,
    DPK_NO_DICT,
    MAX_BLOCK_BYTES,
    DataValidationError,
    UsageError,
    atomic_write,
    block_id,
    coverage,
    dump_json,
    list_block_files,
    sha256_hex,
)
from .pool import load_pool

INDEX_JSON = "index.json"
HEADER_STRUCT = struct.Struct("<4sHQ")  # magic, dict_index, raw_len

LEVEL = 9
WBITS = -15


def compress_payload(block, dict_bytes):
    """Raw deflate (level=9, wbits=-15, default memLevel/strategy)."""
    kwargs = {}
    if dict_bytes is not None:
        kwargs["zdict"] = dict_bytes
    co = zlib.compressobj(
        LEVEL, zlib.DEFLATED, WBITS, zlib.DEF_MEM_LEVEL, zlib.Z_DEFAULT_STRATEGY,
        **kwargs
    )
    return co.compress(block) + co.flush()


def decompress_payload(payload, dict_bytes):
    """Inverse of compress_payload; rejects truncated/trailing streams."""
    if dict_bytes is not None:
        do = zlib.decompressobj(WBITS, zdict=dict_bytes)
    else:
        do = zlib.decompressobj(WBITS)
    try:
        out = do.decompress(payload)
        out += do.flush()
    except zlib.error as exc:
        raise DataValidationError("deflate stream corrupt: %s" % exc)
    if not do.eof or do.unused_data:
        raise DataValidationError("deflate stream truncated or has trailing data")
    return out


def choose_dictionary(block, pool):
    """Best coverage over the pool; ties go to the smaller index."""
    best_index = -1
    best_rho = -1.0
    for index, (windows, entry) in enumerate(zip(pool["windows"], pool["dictionaries"])):
        rho = coverage(block, windows, entry["bytes"])
        if rho > best_rho:
            best_rho = rho
            best_index = index
    return best_index, best_rho


def _read_block(path):
    size = os.path.getsize(path)
    if size > MAX_BLOCK_BYTES:
        raise DataValidationError(
            "block %s is %d bytes, above the %d-byte cap"
            % (path, size, MAX_BLOCK_BYTES)
        )
    with open(path, "rb") as f:
        return f.read()


def pack(pool_dir, batch_dirs, out_dir):
    """Pack every block of every batch with the shared pool."""
    if not batch_dirs:
        raise UsageError("pack requires at least one --blocks directory")
    pool = load_pool(pool_dir)

    os.makedirs(out_dir, exist_ok=True)
    # Replace, never merge: stale .dpk files from another pool must not linger.
    _clean_output_dir(out_dir)

    entries = []
    seen_ids = set()
    for blocks_dir in batch_dirs:
        batch = common.batch_name(blocks_dir)
        for name in list_block_files(blocks_dir):
            bid = block_id(name)
            full_id = batch + "/" + bid
            if full_id in seen_ids:
                raise UsageError("duplicate block id in archive: %s" % full_id)
            seen_ids.add(full_id)

            block = _read_block(os.path.join(blocks_dir, name))
            raw_len = len(block)
            raw_sha = sha256_hex(block)

            index, rho = choose_dictionary(block, pool)
            hit = rho >= COVERAGE_THRESHOLD
            if hit:
                dict_index = index
                dict_bytes = pool["dictionaries"][index]["bytes"]
            else:
                dict_index = None
                dict_bytes = None

            payload = compress_payload(block, dict_bytes)
            header_index = DPK_NO_DICT if dict_index is None else dict_index
            packed = (
                HEADER_STRUCT.pack(DPK_MAGIC, header_index, raw_len) + payload
            )
            packed_len = len(packed)
            atomic_write(os.path.join(out_dir, batch, bid + ".dpk"), packed)
            entries.append(
                {
                    "block_id": full_id,
                    "dict_index": dict_index,
                    "raw_len": raw_len,
                    "raw_sha256": raw_sha,
                    "packed_len": packed_len,
                    "coverage": round(rho, 6),
                    "hit": hit,
                }
            )

    atomic_write(
        os.path.join(out_dir, INDEX_JSON),
        dump_json({"pool_id": pool["pool_id"], "blocks": entries}),
    )
    return pool["pool_id"], entries


def _clean_output_dir(out_dir):
    """Remove batch dirs and stale index from an existing archive dir."""
    if not os.path.isdir(out_dir):
        return
    for name in os.listdir(out_dir):
        if name == INDEX_JSON or (
            os.path.isdir(os.path.join(out_dir, name)) and not name.startswith(".")
        ):
            path = os.path.join(out_dir, name)
            if os.path.isdir(path):
                _rm_tree(path)
            else:
                os.unlink(path)


def _rm_tree(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            os.rmdir(os.path.join(root, d))
    os.rmdir(path)


def _load_archive_index(archive_dir):
    import json

    path = os.path.join(archive_dir, INDEX_JSON)
    if not os.path.isfile(path):
        raise UsageError("archive index not found: %s" % path)
    with open(path, "rb") as f:
        raw = f.read()
    try:
        meta = json.loads(raw.decode("utf-8"))
        pool_id = meta["pool_id"]
        entries = meta["blocks"]
    except (ValueError, KeyError, UnicodeDecodeError):
        raise DataValidationError("malformed archive index: %s" % path)
    if not isinstance(pool_id, str) or not isinstance(entries, list):
        raise DataValidationError("malformed archive index: %s" % path)
    return pool_id, entries


def _unpack_one(archive_dir, entry, pool, dest_dir, write):
    full_id = entry["block_id"]
    if "/" not in full_id or full_id.startswith("/") or ".." in full_id.split("/"):
        raise DataValidationError("unsafe block id in index: %r" % full_id)
    dpk_path = os.path.join(archive_dir, *(full_id.split("/"))) + ".dpk"
    if not os.path.isfile(dpk_path):
        raise DataValidationError("missing packed block: %s" % full_id)
    with open(dpk_path, "rb") as f:
        packed = f.read()
    if len(packed) < DPK_HEADER_LEN:
        raise DataValidationError("short .dpk block: %s" % full_id)
    magic, header_index, raw_len = HEADER_STRUCT.unpack(packed[:DPK_HEADER_LEN])
    if magic != DPK_MAGIC:
        raise DataValidationError("bad magic in block: %s" % full_id)

    index_index = entry.get("dict_index")
    if index_index is None:
        if header_index != DPK_NO_DICT:
            raise DataValidationError("dict_index disagrees with header: %s" % full_id)
        dict_bytes = None
    else:
        if not isinstance(index_index, int) or not (
            0 <= index_index < len(pool["dictionaries"])
        ):
            raise VersionMismatchError(
                "block %s references unknown dictionary %r" % (full_id, index_index)
            )
        if header_index != index_index:
            raise DataValidationError("dict_index disagrees with header: %s" % full_id)
        dict_bytes = pool["dictionaries"][index_index]["bytes"]

    block = decompress_payload(packed[DPK_HEADER_LEN:], dict_bytes)
    if len(block) != raw_len or len(block) != entry.get("raw_len"):
        raise DataValidationError("length mismatch for block: %s" % full_id)
    if sha256_hex(block) != entry.get("raw_sha256"):
        raise DataValidationError("sha256 mismatch for block: %s" % full_id)

    dest_path = os.path.join(dest_dir, *(full_id.split("/"))) + ".blk"
    if write:
        atomic_write(dest_path, block)


def unpack(pool_dir, archive_dir, out_dir):
    """Verify every block first; only then write anything (no partial files)."""
    archive_pool_id, entries = _load_archive_index(archive_dir)
    pool = load_pool(pool_dir, expected_pool_id=archive_pool_id)

    for entry in entries:
        _unpack_one(archive_dir, entry, pool, out_dir, write=False)
    os.makedirs(out_dir, exist_ok=True)
    for entry in entries:
        _unpack_one(archive_dir, entry, pool, out_dir, write=True)
    return len(entries)
