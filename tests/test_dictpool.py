"""dictpool 的 unittest 验收套件（仅用标准库）。"""

import json
import os
import random
import shutil
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dictpool import cli  # noqa: E402
from dictpool.engine import (  # noqa: E402
    coverage,
    dict_windows,
    select_dict,
)
from dictpool.constants import COVERAGE_THRESHOLD  # noqa: E402

SAMPLES = REPO / "samples"
TRAIN_BATCHES = ["logs", "src", "json", "mixed"]
ALL_BATCHES = TRAIN_BATCHES + ["drift", "size"]


def parse_pool_tsv():
    pool_id = None
    rows = []
    for line in (SAMPLES / "expected" / "pool.tsv").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# pool_id"):
            pool_id = line.split("\t")[1]
        elif not line.startswith("#"):
            index, batch, sample, size, sha = line.split("\t")
            rows.append((int(index), batch, int(sample), int(size), sha))
    return pool_id, rows


def parse_eval_tsv():
    bounds = {}
    for line in (SAMPLES / "expected" / "eval.tsv").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        batch, blocks, raw, hit_min, ratio_max = line.split("\t")
        bounds[batch] = (float(hit_min), float(ratio_max))
    return bounds


def blocks_args(names):
    args = []
    for name in names:
        args += ["--blocks", str(SAMPLES / "blocks" / name)]
    return args


def tree_bytes(root):
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = Path(dirpath) / name
            out[str(path.relative_to(root))] = path.read_bytes()
    return out


class DictPoolTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dictpool-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.pool = self.tmp / "var" / "pool"
        self.archive = self.tmp / "var" / "archive"
        self.restored = self.tmp / "var" / "restored"

    def train(self, names=TRAIN_BATCHES, out=None):
        out = out or self.pool
        rc = cli.main(["train"] + blocks_args(names) + ["--out", str(out)])
        self.assertEqual(rc, 0)
        return out

    def pack(self, names=ALL_BATCHES, out=None, pool=None):
        out = out or self.archive
        rc = cli.main(
            ["pack", "--pool", str(pool or self.pool)]
            + blocks_args(names)
            + ["--out", str(out)]
        )
        self.assertEqual(rc, 0)
        return out

    def unpack(self, pool=None, archive=None, out=None):
        return cli.main(
            [
                "unpack",
                "--pool",
                str(pool or self.pool),
                "--archive",
                str(archive or self.archive),
                "--out",
                str(out or self.restored),
            ]
        )

    def read_index(self):
        return json.loads((self.archive / "index.json").read_text())


class TestTrain(DictPoolTestCase):
    def test_pool_matches_expected_baseline(self):
        self.train()
        meta = json.loads((self.pool / "pool.json").read_text())
        pool_id, rows = parse_pool_tsv()
        self.assertEqual(meta["pool_id"], pool_id)
        self.assertEqual(len(meta["dicts"]), len(rows))
        for entry, (index, batch, sample, size, sha) in zip(meta["dicts"], rows):
            self.assertEqual(entry["index"], index)
            self.assertEqual(entry["batch"], batch)
            self.assertEqual(entry["sample_bytes"], sample)
            self.assertEqual(entry["dict_bytes"], size)
            self.assertEqual(entry["sha256"], sha)
            self.assertEqual(
                (self.pool / ("d%d.bin" % index)).read_bytes().__len__(), size
            )

    def test_train_is_deterministic(self):
        first = self.tmp / "p1"
        second = self.tmp / "p2"
        self.train(out=first)
        self.train(out=second)
        self.assertEqual(tree_bytes(first), tree_bytes(second))

    def test_train_rejects_over_capacity(self):
        rc = cli.main(
            ["train"] + blocks_args(ALL_BATCHES) + ["--out", str(self.tmp / "p")]
        )
        self.assertEqual(rc, 1)


class TestPackEval(DictPoolTestCase):
    def test_eval_bounds(self):
        self.train()
        self.pack()
        index = self.read_index()
        per_batch = {}
        for blk in index["blocks"]:
            name = blk["block_id"].split("/", 1)[0]
            agg = per_batch.setdefault(name, [0, 0, 0])
            agg[0] += 1
            agg[1] += blk["raw_len"]
            agg[2] += blk["packed_len"]
            agg.append(blk["hit"])
        for batch, (hit_min, ratio_max) in parse_eval_tsv().items():
            entries = per_batch[batch]
            blocks, raw, packed = entries[0], entries[1], entries[2]
            hits = sum(1 for flag in entries[3:] if flag)
            self.assertGreaterEqual(hits / blocks, hit_min, batch)
            self.assertLessEqual(packed / raw, ratio_max, batch)

    def test_pack_is_deterministic(self):
        self.train()
        first = self.tmp / "a1"
        second = self.tmp / "a2"
        self.pack(out=first)
        self.pack(out=second)
        self.assertEqual(tree_bytes(first), tree_bytes(second))

    def test_index_records_dict_version(self):
        self.train()
        self.pack()
        index = self.read_index()
        meta = json.loads((self.pool / "pool.json").read_text())
        self.assertEqual(index["pool_id"], meta["pool_id"])
        for blk in index["blocks"]:
            if blk["hit"]:
                self.assertIsInstance(blk["dict_index"], int)
            else:
                self.assertIsNone(blk["dict_index"])


class TestRoundTrip(DictPoolTestCase):
    def test_restore_byte_identical(self):
        self.train()
        self.pack()
        self.assertEqual(self.unpack(), 0)
        self.assertEqual(tree_bytes(SAMPLES / "blocks"), tree_bytes(self.restored))


class TestVersioning(DictPoolTestCase):
    def test_tampered_dict_rejected_with_code_2(self):
        self.train()
        self.pack()
        path = self.pool / "d0.bin"
        data = bytearray(path.read_bytes())
        data[0] ^= 0xFF
        path.write_bytes(bytes(data))
        self.assertEqual(self.unpack(), 2)

    def test_wrong_pool_rejected_with_code_2(self):
        self.train()
        self.pack()
        other = self.tmp / "other-pool"
        self.train(names=["src", "json", "logs"], out=other)
        self.assertEqual(self.unpack(pool=other), 2)

    def test_tampered_payload_rejected_with_code_3(self):
        self.train()
        self.pack()
        target = next((self.archive / "logs").glob("*.dpk"))
        data = bytearray(target.read_bytes())
        data[-1] ^= 0xFF
        target.write_bytes(bytes(data))
        self.assertEqual(self.unpack(), 3)

    def test_tampered_index_rejected_with_code_3(self):
        self.train()
        self.pack()
        index_path = self.archive / "index.json"
        index = json.loads(index_path.read_text())
        sha = index["blocks"][0]["raw_sha256"]
        index["blocks"][0]["raw_sha256"] = ("0" if sha[0] != "0" else "1") + sha[1:]
        index_path.write_text(json.dumps(index))
        self.assertEqual(self.unpack(), 3)

    def test_failure_leaves_no_partial_output(self):
        self.train()
        self.pack()
        target = next((self.archive / "logs").glob("*.dpk"))
        data = bytearray(target.read_bytes())
        data[-1] ^= 0xFF
        target.write_bytes(bytes(data))
        self.assertEqual(self.unpack(), 3)
        leftovers = [
            p for p in self.tmp.rglob("*.blk") if "samples" not in str(p)
        ]
        self.assertEqual(leftovers, [])
        self.assertFalse((self.tmp / "var" / "restored.staging").exists())


class TestCoverage(unittest.TestCase):
    def setUp(self):
        self.dict_bytes = bytes(range(256)) * 32  # 8192 B
        self.windows = dict_windows(self.dict_bytes)

    def test_short_block_substring(self):
        self.assertEqual(coverage(b"\x10\x11\x12", self.dict_bytes, self.windows), 1.0)

    def test_short_block_not_substring(self):
        self.assertEqual(
            coverage(b"\xff\x00\xff", self.dict_bytes, self.windows), 0.0
        )

    def test_empty_block(self):
        self.assertEqual(coverage(b"", self.dict_bytes, self.windows), 1.0)

    def test_full_match(self):
        block = self.dict_bytes[:100]
        self.assertEqual(coverage(block, self.dict_bytes, self.windows), 1.0)

    def test_select_dict_threshold(self):
        entries = [(self.dict_bytes, self.windows)]
        index, cov, hit = select_dict(self.dict_bytes[:64], entries)
        self.assertTrue(hit)
        self.assertEqual(index, 0)
        self.assertGreaterEqual(cov, COVERAGE_THRESHOLD)
        index, cov, hit = select_dict(os.urandom(64), entries)
        self.assertFalse(hit)
        self.assertIsNone(index)


class TestReport(DictPoolTestCase):
    def test_report_single_file_with_engine_numbers(self):
        self.train()
        self.pack()
        out = self.tmp / "var" / "report" / "index.html"
        rc = cli.main(
            ["report", "--archive", str(self.archive), "--out", str(out)]
        )
        self.assertEqual(rc, 0)
        html = out.read_text(encoding="utf-8")
        self.assertIn('type="module"', html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        index = self.read_index()
        self.assertIn(index["pool_id"], html)
        for name in ALL_BATCHES:
            self.assertIn(name, html)
        # 内联 JSON 与引擎输出逐块一致
        marker = '<script type="application/json" id="dictpool-data">'
        payload = html.split(marker, 1)[1].split("</script>", 1)[0]
        data = json.loads(payload)
        self.assertEqual(len(data["blocks"]), len(index["blocks"]))
        for got, want in zip(data["blocks"], index["blocks"]):
            self.assertEqual(got["block_id"], want["block_id"])
            self.assertEqual(got["packed_len"], want["packed_len"])
            self.assertEqual(got["hit"], want["hit"])
        drift = next(b for b in data["batches"] if b["name"] == "drift")
        self.assertFalse(drift["ok"])  # 命中率 0，应标红
        logs = next(b for b in data["batches"] if b["name"] == "logs")
        self.assertTrue(logs["ok"])


class TestMemory(DictPoolTestCase):
    """输入翻倍，tracemalloc 峰值增幅不得超过 32 MiB。"""

    def _make_blocks(self, root, count, size):
        rng = random.Random(20260311)
        words = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
            "status=200", "method=GET", "path=/v1/orders", "host=10.3.7.42",
        ]
        line = " ".join(words) + "\n"
        root.mkdir(parents=True)
        for i in range(count):
            body = (line * (size // len(line) + 1))[: size - 64]
            tail = str(rng.random()) + "x" * size
            (root / ("%04d.blk" % i)).write_text((body + tail)[:size])

    def _pack_peak(self, blocks_dir, n_blocks):
        pool_dir = self.tmp / ("pool-%d" % n_blocks)
        archive_dir = self.tmp / ("arch-%d" % n_blocks)
        self.assertEqual(
            cli.main(
                ["train", "--blocks", str(blocks_dir), "--out", str(pool_dir)]
            ),
            0,
        )
        tracemalloc.start()
        try:
            rc = cli.main(
                [
                    "pack",
                    "--pool",
                    str(pool_dir),
                    "--blocks",
                    str(blocks_dir),
                    "--out",
                    str(archive_dir),
                ]
            )
            self.assertEqual(rc, 0)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak

    def test_peak_does_not_scale_linearly(self):
        small = self.tmp / "blocks-small"
        large = self.tmp / "blocks-large"
        self._make_blocks(small, 8, 65536)
        self._make_blocks(large, 16, 65536)
        peak_small = self._pack_peak(small, 8)
        peak_large = self._pack_peak(large, 16)
        self.assertLessEqual(peak_small, 128 * 1024 * 1024)
        self.assertLessEqual(peak_large, 128 * 1024 * 1024)
        self.assertLessEqual(peak_large - peak_small, 32 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
