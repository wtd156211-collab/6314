"""训练、压缩、解压的核心逻辑与 var/ 存储格式。"""

import hashlib
import json
import os
import shutil
import struct
import zlib

from .constants import (
    BLOCK_EXT,
    COVERAGE_THRESHOLD,
    DICT_MAX_FRAGMENTS,
    EXIT_DATA_MISMATCH,
    EXIT_POOL_MISMATCH,
    EXIT_USAGE,
    FRAGMENT_LEN,
    FRAGMENT_MIN_COUNT,
    HEADER_LEN,
    MAGIC,
    MAX_BLOCK_BYTES,
    NO_DICT,
    PACKED_EXT,
    POOL_CAPACITY,
    RESTORED_EXT,
    SAMPLE_CAP_PER_BATCH,
    SAMPLE_PER_BLOCK,
    ZLIB_LEVEL,
    ZLIB_WBITS,
)


class DictPoolError(Exception):
    """带退出码的业务错误。"""

    exit_code = EXIT_USAGE


class UsageError(DictPoolError):
    exit_code = EXIT_USAGE


class PoolMismatchError(DictPoolError):
    """池缺失、损坏或版本不匹配。"""

    exit_code = EXIT_POOL_MISMATCH


class DataMismatchError(DictPoolError):
    """归档数据校验失败。"""

    exit_code = EXIT_DATA_MISMATCH


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def _fs_sorted(names):
    return sorted(names, key=os.fsencode)


def list_block_files(blocks_dir):
    """按文件名字节序升序列出 (block_id, path)。"""
    if not os.path.isdir(blocks_dir):
        raise UsageError("块目录不存在: %s" % blocks_dir)
    names = [
        name
        for name in os.listdir(blocks_dir)
        if name.endswith(BLOCK_EXT)
        and os.path.isfile(os.path.join(blocks_dir, name))
    ]
    return [
        (name[: -len(BLOCK_EXT)], os.path.join(blocks_dir, name))
        for name in _fs_sorted(names)
    ]


def read_block(path):
    with open(path, "rb") as fh:
        data = fh.read(MAX_BLOCK_BYTES + 1)
    if len(data) > MAX_BLOCK_BYTES:
        raise UsageError("块超过 8 MiB 上限: %s" % path)
    return data


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def sample_segments(blocks_dir):
    """按口径采样：块名升序取块首，累加到单批上限截断。返回 (segments, total)。"""
    segments = []
    total = 0
    for _block_id, path in list_block_files(blocks_dir):
        if total >= SAMPLE_CAP_PER_BATCH:
            break
        with open(path, "rb") as fh:
            head = fh.read(SAMPLE_PER_BLOCK)
        take = min(len(head), SAMPLE_CAP_PER_BATCH - total)
        if take > 0:
            segments.append(head[:take])
            total += take
    return segments, total


def train_dict_from_segments(segments):
    """统计 16 B 窗口计数，按（计数升序，字节升序）取末尾 512 个拼成字典。"""
    counts = {}
    for seg in segments:
        for i in range(len(seg) - FRAGMENT_LEN + 1):
            window = seg[i : i + FRAGMENT_LEN]
            counts[window] = counts.get(window, 0) + 1
    eligible = [
        (count, window)
        for window, count in counts.items()
        if count >= FRAGMENT_MIN_COUNT
    ]
    eligible.sort(key=lambda item: (item[0], item[1]))
    chosen = eligible[-DICT_MAX_FRAGMENTS:]
    return b"".join(window for _count, window in chosen)


def train_batch(batch, blocks_dir, index):
    segments, sample_bytes = sample_segments(blocks_dir)
    dict_bytes = train_dict_from_segments(segments)
    return {
        "index": index,
        "batch": batch,
        "sample_bytes": sample_bytes,
        "dict_bytes": len(dict_bytes),
        "sha256": sha256_hex(dict_bytes),
        "_data": dict_bytes,
    }


def train_pool(batch_dirs):
    """batch_dirs: [(batch, dir), ...]，索引按给定顺序从 0 开始。"""
    if not 1 <= len(batch_dirs) <= POOL_CAPACITY:
        raise UsageError("训练批数量必须在 1..%d 之间" % POOL_CAPACITY)
    names = [batch for batch, _dir in batch_dirs]
    if len(set(names)) != len(names):
        raise UsageError("批名重复: %s" % ", ".join(names))
    dicts = [
        train_batch(batch, path, index)
        for index, (batch, path) in enumerate(batch_dirs)
    ]
    pool_id = sha256_hex(
        "".join(entry["sha256"] for entry in dicts).encode("ascii")
    )
    return pool_id, dicts


# ---------------------------------------------------------------------------
# 覆盖度与字典选择
# ---------------------------------------------------------------------------


def dict_windows(dict_bytes):
    """字典的全部 16 字节窗口集合。"""
    return {
        dict_bytes[i : i + FRAGMENT_LEN]
        for i in range(len(dict_bytes) - FRAGMENT_LEN + 1)
    }


def coverage(block, dict_bytes, windows):
    """按 README 口径计算块对字典的覆盖度。"""
    size = len(block)
    if size < FRAGMENT_LEN:
        return 1.0 if block in dict_bytes else 0.0
    total = size - FRAGMENT_LEN + 1
    hits = 0
    for i in range(total):
        if block[i : i + FRAGMENT_LEN] in windows:
            hits += 1
    return hits / total


def select_dict(block, dict_entries):
    """dict_entries: [(dict_bytes, windows), ...]。返回 (dict_index|None, coverage, hit)。"""
    best_index = None
    best_cov = -1.0
    for index, (dict_bytes, windows) in enumerate(dict_entries):
        cov = coverage(block, dict_bytes, windows)
        if cov > best_cov:  # 严格大于：并列取索引小者
            best_cov = cov
            best_index = index
    if best_index is not None and best_cov >= COVERAGE_THRESHOLD:
        return best_index, best_cov, True
    return None, best_cov, False


# ---------------------------------------------------------------------------
# deflate / inflate
# ---------------------------------------------------------------------------


def deflate(data, zdict=None):
    if zdict is None:
        comp = zlib.compressobj(ZLIB_LEVEL, zlib.DEFLATED, ZLIB_WBITS)
    else:
        comp = zlib.compressobj(
            ZLIB_LEVEL,
            zlib.DEFLATED,
            ZLIB_WBITS,
            zlib.DEF_MEM_LEVEL,
            zlib.Z_DEFAULT_STRATEGY,
            zdict,
        )
    return comp.compress(data) + comp.flush()


def inflate(payload, zdict=None):
    if zdict is None:
        decomp = zlib.decompressobj(ZLIB_WBITS)
    else:
        decomp = zlib.decompressobj(ZLIB_WBITS, zdict=zdict)
    try:
        out = bytearray()
        chunk = decomp.decompress(payload, MAX_BLOCK_BYTES + 1)
        out += chunk
        while decomp.unconsumed_tail:
            if len(out) > MAX_BLOCK_BYTES:
                raise DataMismatchError("解压结果超过单块上限")
            out += decomp.decompress(decomp.unconsumed_tail, MAX_BLOCK_BYTES + 1)
        out += decomp.flush()
    except zlib.error as exc:
        raise DataMismatchError("deflate 载荷解压失败: %s" % exc)
    if not decomp.eof:
        raise DataMismatchError("deflate 载荷不完整")
    if len(out) > MAX_BLOCK_BYTES:
        raise DataMismatchError("解压结果超过单块上限")
    return bytes(out)


# ---------------------------------------------------------------------------
# var/ 存储
# ---------------------------------------------------------------------------


def _write_json(path, obj):
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")


def _read_json(path, exc_type, what):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise exc_type("%s不存在: %s" % (what, path))
    except (OSError, ValueError) as exc:
        raise exc_type("%s无法解析: %s (%s)" % (what, path, exc))


def _fresh_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)
    os.makedirs(path)


def _promote(staging, final):
    """把暂存目录原子地放到最终位置，失败时不留残缺输出。"""
    if os.path.isdir(final):
        shutil.rmtree(final)
    elif os.path.lexists(final):
        os.remove(final)
    os.rename(staging, final)


def _public_dict(entry):
    return {
        "index": entry["index"],
        "batch": entry["batch"],
        "sample_bytes": entry["sample_bytes"],
        "dict_bytes": entry["dict_bytes"],
        "sha256": entry["sha256"],
    }


def write_pool(pool_dir, pool_id, dicts):
    staging = pool_dir.rstrip(os.sep) + ".staging"
    _fresh_dir(staging)
    try:
        for entry in dicts:
            name = "d%d.bin" % entry["index"]
            with open(os.path.join(staging, name), "wb") as fh:
                fh.write(entry["_data"])
        meta = {
            "pool_id": pool_id,
            "dicts": [_public_dict(entry) for entry in dicts],
        }
        _write_json(os.path.join(staging, "pool.json"), meta)
        _promote(staging, pool_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_pool(pool_dir):
    meta = _read_json(
        os.path.join(pool_dir, "pool.json"), PoolMismatchError, "池描述文件"
    )
    try:
        pool_id = meta["pool_id"]
        dicts_meta = meta["dicts"]
    except (KeyError, TypeError):
        raise PoolMismatchError("池描述文件缺少字段: %s" % pool_dir)
    dicts = []
    for entry in dicts_meta:
        index = entry["index"]
        name = "d%d.bin" % index
        path = os.path.join(pool_dir, name)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            raise PoolMismatchError("字典文件缺失: %s" % path)
        if sha256_hex(data) != entry["sha256"]:
            raise PoolMismatchError("字典校验失败（sha256 不符）: %s" % path)
        merged = dict(entry)
        merged["_data"] = data
        dicts.append(merged)
    dicts.sort(key=lambda entry: entry["index"])
    if [entry["index"] for entry in dicts] != list(range(len(dicts))):
        raise PoolMismatchError("字典索引不连续: %s" % pool_dir)
    actual = sha256_hex(
        "".join(entry["sha256"] for entry in dicts).encode("ascii")
    )
    if actual != pool_id:
        raise PoolMismatchError("pool_id 与字典内容不符: %s" % pool_dir)
    return pool_id, dicts


def write_archive(archive_dir, pool_id, dicts, blocks, packed_files):
    staging = archive_dir.rstrip(os.sep) + ".staging"
    _fresh_dir(staging)
    try:
        for rel_path, payload in packed_files:
            dest = os.path.join(staging, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(payload)
        index = {
            "pool_id": pool_id,
            "dicts": [_public_dict(entry) for entry in dicts],
            "blocks": blocks,
        }
        _write_json(os.path.join(staging, "index.json"), index)
        _promote(staging, archive_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_archive(archive_dir):
    index = _read_json(
        os.path.join(archive_dir, "index.json"), DataMismatchError, "归档索引"
    )
    for key in ("pool_id", "dicts", "blocks"):
        if key not in index:
            raise DataMismatchError("归档索引缺少字段 %s" % key)
    return index


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------


def _check_block_id(block_id):
    parts = block_id.split("/")
    if (
        len(parts) != 2
        or any(part in ("", ".", "..") for part in parts)
        or "\\" in block_id
    ):
        raise DataMismatchError("非法块 ID: %r" % block_id)


def pack_batches(pool_id, dicts, batch_dirs, archive_dir):
    """逐块选字典压缩，写入归档。返回归档索引中的块数组。"""
    dict_entries = [(entry["_data"], dict_windows(entry["_data"])) for entry in dicts]
    names = [batch for batch, _dir in batch_dirs]
    if len(set(names)) != len(names):
        raise UsageError("批名重复: %s" % ", ".join(names))
    blocks = []
    packed_files = []
    for batch, blocks_dir in batch_dirs:
        for block_id, path in list_block_files(blocks_dir):
            data = read_block(path)
            dict_index, cov, hit = select_dict(data, dict_entries)
            zdict = dicts[dict_index]["_data"] if hit else None
            payload = deflate(data, zdict)
            header = struct.pack(
                "<4sHQ", MAGIC, dict_index if hit else NO_DICT, len(data)
            )
            full_id = "%s/%s" % (batch, block_id)
            packed_files.append((full_id + PACKED_EXT, header + payload))
            blocks.append(
                {
                    "block_id": full_id,
                    "dict_index": dict_index if hit else None,
                    "raw_len": len(data),
                    "raw_sha256": sha256_hex(data),
                    "packed_len": HEADER_LEN + len(payload),
                    "coverage": cov,
                    "hit": hit,
                }
            )
    write_archive(archive_dir, pool_id, dicts, blocks, packed_files)
    return blocks


def unpack_archive(pool_id, dicts, archive_dir, out_dir):
    """校验并还原全部块；任何不符即抛错且不写残缺文件。"""
    index = read_archive(archive_dir)
    if index["pool_id"] != pool_id:
        raise PoolMismatchError(
            "归档 pool_id 与当前池不一致: %s != %s" % (index["pool_id"], pool_id)
        )
    archived = {entry["index"]: entry["sha256"] for entry in index["dicts"]}
    for entry in dicts:
        expect = archived.get(entry["index"])
        if expect is None or expect != entry["sha256"]:
            raise PoolMismatchError(
                "字典版本不匹配: 索引 %d" % entry["index"]
            )

    staging = out_dir.rstrip(os.sep) + ".staging"
    _fresh_dir(staging)
    restored = 0
    try:
        for entry in index["blocks"]:
            block_id = entry["block_id"]
            _check_block_id(block_id)
            dpk_path = os.path.join(archive_dir, block_id + PACKED_EXT)
            try:
                with open(dpk_path, "rb") as fh:
                    packed = fh.read()
            except OSError:
                raise DataMismatchError("归档块缺失: %s" % dpk_path)
            if len(packed) < HEADER_LEN or packed[:4] != MAGIC:
                raise DataMismatchError("块头非法: %s" % block_id)
            _magic, dict_index, raw_len = struct.unpack(
                "<4sHQ", packed[:HEADER_LEN]
            )
            expect_index = entry["dict_index"]
            if expect_index is None:
                expect_index = NO_DICT
            if dict_index != expect_index or raw_len != entry["raw_len"]:
                raise DataMismatchError("块头与索引不一致: %s" % block_id)
            if dict_index == NO_DICT:
                zdict = None
            else:
                if not 0 <= dict_index < len(dicts):
                    raise PoolMismatchError(
                        "块引用了池外字典索引 %d: %s" % (dict_index, block_id)
                    )
                zdict = dicts[dict_index]["_data"]
            data = inflate(packed[HEADER_LEN:], zdict)
            if len(data) != entry["raw_len"]:
                raise DataMismatchError("长度校验失败: %s" % block_id)
            if sha256_hex(data) != entry["raw_sha256"]:
                raise DataMismatchError("sha256 校验失败: %s" % block_id)
            dest = os.path.join(staging, block_id + RESTORED_EXT)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(data)
            restored += 1
        _promote(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return restored
