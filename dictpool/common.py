"""Constants, errors and shared helpers for the dictionary pool.

All fixed parameters come from README section 2 and must not be tuned.
"""

import hashlib
import os
import tempfile

# Fixed parameters (README section 2).
BLOCK_SAMPLE_BYTES = 8192        # per-block sample: head of block
BATCH_SAMPLE_CAP = 131072        # per-batch sample cap
FRAGMENT_LEN = 16                # fragment / window length
MIN_FRAGMENT_COUNT = 2           # fragments seen less are dropped
DICT_MAX_BYTES = 8192            # dictionary cap: at most 512 fragments
POOL_CAPACITY = 4                # dictionary indexes 0..3
COVERAGE_THRESHOLD = 0.25        # hit gate
MAX_BLOCK_BYTES = 8 * 1024 * 1024  # single-block cap

# .dpk block header: magic(4) + dict_index uint16 LE + raw_len uint64 LE.
DPK_MAGIC = b"DPK1"
DPK_NO_DICT = 0xFFFF
DPK_HEADER_LEN = 14

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_VERSION_MISMATCH = 2
EXIT_DATA_INVALID = 3


class DictPoolError(Exception):
    """Base error carrying the process exit code."""

    exit_code = EXIT_USAGE


class UsageError(DictPoolError):
    exit_code = EXIT_USAGE


class VersionMismatchError(DictPoolError):
    exit_code = EXIT_VERSION_MISMATCH


class DataValidationError(DictPoolError):
    exit_code = EXIT_DATA_INVALID


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def list_block_files(blocks_dir):
    """Return .blk file names sorted by byte order (README section 2)."""
    try:
        names = os.listdir(blocks_dir)
    except FileNotFoundError:
        raise UsageError("blocks directory not found: %s" % blocks_dir)
    except NotADirectoryError:
        raise UsageError("not a blocks directory: %s" % blocks_dir)
    names = [n for n in names if n.endswith(".blk")]
    names.sort(key=lambda n: n.encode("utf-8", "surrogateescape"))
    return names


def batch_name(blocks_dir):
    """Batch name is the directory name of the blocks path."""
    norm = os.path.normpath(blocks_dir)
    name = os.path.basename(norm)
    if not name:
        raise UsageError("cannot derive batch name from: %s" % blocks_dir)
    return name


def block_id(file_name):
    """Block ID is the file name without the .blk extension."""
    return file_name[: -len(".blk")]


def dict_windows(dict_bytes):
    """W(d): the set of all 16-byte windows of a dictionary."""
    return {
        dict_bytes[i : i + FRAGMENT_LEN]
        for i in range(len(dict_bytes) - FRAGMENT_LEN + 1)
    }


def coverage(block, windows, dict_bytes):
    """Coverage rho of a block against one dictionary (README section 2)."""
    n = len(block)
    if n < FRAGMENT_LEN:
        if n == 0:
            return 0.0
        return 1.0 if block in dict_bytes else 0.0
    hits = 0
    for i in range(n - FRAGMENT_LEN + 1):
        if block[i : i + FRAGMENT_LEN] in windows:
            hits += 1
    return hits / (n - FRAGMENT_LEN + 1)


def atomic_write(path, data):
    """Write bytes to path atomically (temp file + rename)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def dump_json(obj):
    """Deterministic UTF-8 JSON serialization."""
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False).encode(
        "utf-8"
    ) + b"\n"
