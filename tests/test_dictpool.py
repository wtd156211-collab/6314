"""Acceptance tests for dictpool (stdlib unittest only)."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dictpool import archive, common, pool, report  # noqa: E402
from dictpool.__main__ import main as cli_main  # noqa: E402

SAMPLES = os.path.join(REPO, "samples")
BLOCKS = os.path.join(SAMPLES, "blocks")
EXPECTED = os.path.join(SAMPLES, "expected")
TRAIN_BATCHES = ["logs", "src", "json", "mixed"]
ALL_BATCHES = TRAIN_BATCHES + ["drift", "size"]


def read_expected_pool():
    rows = []
    pool_id = None
    with open(os.path.join(EXPECTED, "pool.tsv"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# pool_id"):
                pool_id = line.split()[-1]
                continue
            if line.startswith("#"):
                continue
            index, batch, sample_bytes, dict_bytes, sha = line.split("\t")
            rows.append(
                {
                    "index": int(index),
                    "batch": batch,
                    "sample_bytes": int(sample_bytes),
                    "dict_bytes": int(dict_bytes),
                    "sha256": sha,
                }
            )
    return pool_id, rows


def read_expected_eval():
    rows = {}
    with open(os.path.join(EXPECTED, "eval.tsv"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            batch, blocks, raw, hit_min, ratio_max = line.split("\t")
            rows[batch] = {
                "blocks": int(blocks),
                "raw_bytes": int(raw),
                "hit_rate_min": float(hit_min),
                "ratio_max": float(ratio_max),
            }
    return rows


class DictPoolAcceptance(unittest.TestCase):
    """Train + pack once per class; every test reuses that engine output."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dictpool-test-")
        cls.pool_dir = os.path.join(cls.tmp, "pool")
        cls.archive_dir = os.path.join(cls.tmp, "archive")
        cls.restore_dir = os.path.join(cls.tmp, "restored")
        cls.pool_id, cls.dicts = pool.train(
            [os.path.join(BLOCKS, b) for b in TRAIN_BATCHES], cls.pool_dir
        )
        cls.pack_pool_id, cls.entries = archive.pack(
            cls.pool_dir, [os.path.join(BLOCKS, b) for b in ALL_BATCHES],
            cls.archive_dir,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # 1. pool.tsv baseline -------------------------------------------------
    def test_pool_matches_expected_baseline(self):
        expected_pool_id, rows = read_expected_pool()
        self.assertEqual(self.pool_id, expected_pool_id)
        self.assertEqual(len(self.dicts), len(rows))
        for got, want in zip(self.dicts, rows):
            self.assertEqual(got["index"], want["index"])
            self.assertEqual(got["batch"], want["batch"])
            self.assertEqual(got["sample_bytes"], want["sample_bytes"])
            self.assertEqual(got["dict_bytes"], want["dict_bytes"])
            self.assertEqual(got["sha256"], want["sha256"])
            self.assertLessEqual(got["dict_bytes"], common.DICT_MAX_BYTES)

    # 2. eval.tsv thresholds ------------------------------------------------
    def test_eval_thresholds(self):
        expected = read_expected_eval()
        stats = {}
        for e in self.entries:
            batch = e["block_id"].split("/", 1)[0]
            s = stats.setdefault(batch, {"blocks": 0, "hits": 0, "raw": 0, "packed": 0})
            s["blocks"] += 1
            s["hits"] += 1 if e["hit"] else 0
            s["raw"] += e["raw_len"]
            s["packed"] += e["packed_len"]
        self.assertEqual(set(stats), set(expected))
        for batch, want in expected.items():
            s = stats[batch]
            self.assertEqual(s["blocks"], want["blocks"], batch)
            self.assertEqual(s["raw"], want["raw_bytes"], batch)
            hit_rate = s["hits"] / s["blocks"]
            ratio = s["packed"] / s["raw"]
            self.assertGreaterEqual(hit_rate, want["hit_rate_min"], batch)
            self.assertLessEqual(ratio, want["ratio_max"], batch)

    # 3. byte-identical roundtrip ------------------------------------------
    def test_roundtrip_byte_identical(self):
        n = archive.unpack(self.pool_dir, self.archive_dir, self.restore_dir)
        self.assertEqual(n, len(self.entries))
        for batch in ALL_BATCHES:
            src = os.path.join(BLOCKS, batch)
            dst = os.path.join(self.restore_dir, batch)
            for name in common.list_block_files(src):
                with open(os.path.join(src, name), "rb") as f:
                    original = f.read()
                with open(os.path.join(dst, name), "rb") as f:
                    restored = f.read()
                self.assertEqual(original, restored, "%s/%s" % (batch, name))

    # 4. determinism ---------------------------------------------------------
    def test_deterministic_train_and_pack(self):
        pool2 = os.path.join(self.tmp, "pool2")
        archive2 = os.path.join(self.tmp, "archive2")
        pool_id2, _ = pool.train(
            [os.path.join(BLOCKS, b) for b in TRAIN_BATCHES], pool2
        )
        self.assertEqual(pool_id2, self.pool_id)
        for i in range(len(self.dicts)):
            with open(os.path.join(self.pool_dir, "d%d.bin" % i), "rb") as f:
                a = f.read()
            with open(os.path.join(pool2, "d%d.bin" % i), "rb") as f:
                b = f.read()
            self.assertEqual(a, b)
        archive.pack(pool2, [os.path.join(BLOCKS, b) for b in ALL_BATCHES], archive2)
        for batch in ALL_BATCHES:
            for name in common.list_block_files(os.path.join(BLOCKS, batch)):
                bid = common.block_id(name)
                with open(os.path.join(self.archive_dir, batch, bid + ".dpk"), "rb") as f:
                    a = f.read()
                with open(os.path.join(archive2, batch, bid + ".dpk"), "rb") as f:
                    b = f.read()
                self.assertEqual(a, b, "packed bytes differ for %s/%s" % (batch, bid))

    # 5. tamper detection ------------------------------------------------------
    def test_tampered_dictionary_rejected(self):
        evil_pool = os.path.join(self.tmp, "evil-pool")
        shutil.copytree(self.pool_dir, evil_pool)
        with open(os.path.join(evil_pool, "d0.bin"), "r+b") as f:
            f.write(b"X")
        with self.assertRaises(common.VersionMismatchError):
            archive.unpack(evil_pool, self.archive_dir,
                           os.path.join(self.tmp, "out-evil"))

    def test_tampered_archive_rejected(self):
        evil_archive = os.path.join(self.tmp, "evil-archive")
        shutil.copytree(self.archive_dir, evil_archive)
        victim = os.path.join(evil_archive, "logs", "0001.dpk")
        with open(victim, "r+b") as f:
            f.seek(common.DPK_HEADER_LEN)
            f.write(b"\xff")
        out = os.path.join(self.tmp, "out-evil-archive")
        with self.assertRaises(common.DataValidationError):
            archive.unpack(self.pool_dir, evil_archive, out)
        # no partial files may be written on failure
        self.assertFalse(os.path.exists(out) and os.listdir(out))

    def test_wrong_pool_rejected(self):
        other_pool = os.path.join(self.tmp, "other-pool")
        pool.train([os.path.join(BLOCKS, b) for b in reversed(TRAIN_BATCHES)],
                   other_pool)
        with self.assertRaises(common.VersionMismatchError):
            archive.unpack(other_pool, self.archive_dir,
                           os.path.join(self.tmp, "out-wrong-pool"))

    # 6. coverage semantics on fragment boundary ------------------------------
    def test_short_block_coverage(self):
        d = self.dicts[0]["_bytes"]
        windows = common.dict_windows(d)
        inside = d[100:110]
        self.assertEqual(common.coverage(inside, windows, d), 1.0)
        self.assertEqual(common.coverage(b"\x00" * 10, windows, d), 0.0)
        self.assertEqual(common.coverage(b"", windows, d), 0.0)
        w16 = d[200:216]
        self.assertEqual(common.coverage(w16, windows, d), 1.0)

    # 7. report ----------------------------------------------------------------
    def test_report_single_file_no_external_refs(self):
        out = os.path.join(self.tmp, "report", "index.html")
        batches = report.build_report(self.archive_dir, out)
        with open(out, encoding="utf-8") as f:
            doc = f.read()
        self.assertIn('<script type="module">', doc)
        self.assertNotIn("http://", doc)
        self.assertNotIn("https://", doc)
        self.assertNotIn("src=", doc)
        self.assertNotIn("href=", doc)
        self.assertIn(self.pool_id, doc)
        self.assertEqual({b["batch"] for b in batches}, set(ALL_BATCHES))
        drift = next(b for b in batches if b["batch"] == "drift")
        self.assertEqual(drift["hit_rate"], 0.0)
        self.assertLess(drift["ratio"], 1.0)  # fallback must not degrade

    # 8. CLI exit codes ----------------------------------------------------------
    def test_cli_exit_codes(self):
        self.assertEqual(cli_main(["train", "--blocks", os.path.join(BLOCKS, "logs"),
                                   "--out", os.path.join(self.tmp, "p3")]), 0)
        self.assertEqual(cli_main(["pack", "--pool", self.pool_dir,
                                   "--blocks", os.path.join(BLOCKS, "logs"),
                                   "--out", os.path.join(self.tmp, "a3")]), 0)
        self.assertEqual(cli_main(["unpack", "--pool", self.pool_dir,
                                   "--archive", self.archive_dir,
                                   "--out", os.path.join(self.tmp, "r3")]), 0)
        # usage error -> argparse exits with code 1
        with self.assertRaises(SystemExit) as cm:
            cli_main(["pack", "--pool", self.pool_dir])
        self.assertEqual(cm.exception.code, 1)
        # version mismatch -> 2
        other_pool = os.path.join(self.tmp, "p4")
        pool.train([os.path.join(BLOCKS, b) for b in reversed(TRAIN_BATCHES)],
                   other_pool)
        self.assertEqual(cli_main(["unpack", "--pool", other_pool,
                                   "--archive", self.archive_dir,
                                   "--out", os.path.join(self.tmp, "r4")]), 2)
        # data validation -> 3
        evil = os.path.join(self.tmp, "a5")
        shutil.copytree(self.archive_dir, evil)
        with open(os.path.join(evil, "index.json"), "r+b") as f:
            raw = f.read()
            f.seek(0)
            f.write(raw.replace(b'"hit": true', b'"hit": truf', 1))
        self.assertEqual(cli_main(["unpack", "--pool", self.pool_dir,
                                   "--archive", evil,
                                   "--out", os.path.join(self.tmp, "r5")]), 3)

    # 9. stdlib only ---------------------------------------------------------------
    def test_stdlib_imports_only(self):
        pkg = os.path.join(REPO, "dictpool")
        allowed = set(sys.stdlib_module_names)
        import ast

        for name in os.listdir(pkg):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(pkg, name), encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    roots = [node.module.split(".")[0]] if node.level == 0 else []
                else:
                    continue
                for root in roots:
                    self.assertIn(root, allowed, "%s imports %s" % (name, root))


class CliSmoke(unittest.TestCase):
    """The exact README command lines must work from the repo root."""

    def test_readme_pipeline(self):
        tmp = tempfile.mkdtemp(prefix="dictpool-cli-")
        self.addCleanup(shutil.rmtree, tmp, True)
        env = dict(os.environ)
        cmds = [
            ["train"] + sum(
                (["--blocks", os.path.join(BLOCKS, b)] for b in TRAIN_BATCHES), []
            ) + ["--out", os.path.join(tmp, "pool")],
            ["pack", "--pool", os.path.join(tmp, "pool")]
            + sum((["--blocks", os.path.join(BLOCKS, b)] for b in ALL_BATCHES), [])
            + ["--out", os.path.join(tmp, "archive")],
            ["unpack", "--pool", os.path.join(tmp, "pool"),
             "--archive", os.path.join(tmp, "archive"),
             "--out", os.path.join(tmp, "restored")],
            ["report", "--archive", os.path.join(tmp, "archive"),
             "--out", os.path.join(tmp, "report", "index.html")],
        ]
        for args in cmds:
            proc = subprocess.run(
                [sys.executable, "-m", "dictpool"] + args,
                cwd=REPO, env=env, capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, "%s: %s" % (args, proc.stderr))
        self.assertTrue(
            os.path.isfile(os.path.join(tmp, "report", "index.html"))
        )


if __name__ == "__main__":
    unittest.main()
