"""python -m dictpool — train / pack / unpack / report."""

import argparse
import sys

from . import archive, pool, report
from .common import (
    COVERAGE_THRESHOLD,
    DICT_MAX_BYTES,
    EXIT_OK,
    EXIT_USAGE,
    DictPoolError,
)


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(EXIT_USAGE, "%s: error: %s\n" % (self.prog, message))


def build_parser():
    p = Parser(
        prog="dictpool",
        description="Shared compression-dictionary pool (stdlib only).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train one dictionary per batch into a pool")
    t.add_argument("--blocks", action="append", required=True, metavar="DIR",
                   help="blocks directory of a training batch (repeatable, "
                        "pool index follows this order)")
    t.add_argument("--out", required=True, metavar="DIR", help="pool output dir")

    pk = sub.add_parser("pack", help="pack blocks with the pool")
    pk.add_argument("--pool", required=True, metavar="DIR")
    pk.add_argument("--blocks", action="append", required=True, metavar="DIR")
    pk.add_argument("--out", required=True, metavar="DIR", help="archive dir")

    u = sub.add_parser("unpack", help="verify and restore an archive")
    u.add_argument("--pool", required=True, metavar="DIR")
    u.add_argument("--archive", required=True, metavar="DIR")
    u.add_argument("--out", required=True, metavar="DIR", help="restore dir")

    r = sub.add_parser("report", help="render the single-file HTML report")
    r.add_argument("--archive", required=True, metavar="DIR")
    r.add_argument("--out", required=True, metavar="FILE", help="HTML file")
    r.add_argument("--pool", metavar="DIR", default=None,
                   help="pool dir for dictionary sizes "
                        "(default: <archive>/../pool if present)")
    r.add_argument("--hit-rate-min", type=float, default=COVERAGE_THRESHOLD,
                   help="per-batch hit-rate bar (default: %(default)s)")
    r.add_argument("--ratio-max", type=float, default=1.0,
                   help="per-batch compression-ratio bar (default: %(default)s)")
    return p


def cmd_train(args):
    pool_id, dicts = pool.train(args.blocks, args.out)
    for d in dicts:
        print("d%d  %-8s sample=%dB  dict=%dB/%dB  sha256=%s"
              % (d["index"], d["batch"], d["sample_bytes"], d["dict_bytes"],
                 DICT_MAX_BYTES, d["sha256"]))
    print("pool_id %s  (%d dictionaries, cap %d)"
          % (pool_id, len(dicts), pool.POOL_CAPACITY))
    return EXIT_OK


def cmd_pack(args):
    pool_id, entries = archive.pack(args.pool, args.blocks, args.out)
    hits = sum(1 for e in entries if e["hit"])
    raw = sum(e["raw_len"] for e in entries)
    packed = sum(e["packed_len"] for e in entries)
    ratio = (packed / raw) if raw else 0.0
    print("packed %d blocks (%d hit, %.1f%%)  ratio %.3f  pool %s"
          % (len(entries), hits, 100.0 * hits / len(entries) if entries else 0.0,
             ratio, pool_id))
    return EXIT_OK


def cmd_unpack(args):
    n = archive.unpack(args.pool, args.archive, args.out)
    print("restored %d blocks to %s" % (n, args.out))
    return EXIT_OK


def cmd_report(args):
    batches = report.build_report(
        args.archive, args.out, pool_dir=args.pool,
        hit_rate_min=args.hit_rate_min, ratio_max=args.ratio_max,
    )
    for b in batches:
        ok = b["hit_rate"] >= args.hit_rate_min and b["ratio"] <= args.ratio_max
        print("%-8s hit %.1f%%  ratio %.3f  %s"
              % (b["batch"], 100.0 * b["hit_rate"], b["ratio"],
                 "PASS" if ok else "FAIL"))
    print("report written to %s" % args.out)
    return EXIT_OK


def main(argv=None):
    args = build_parser().parse_args(argv)
    handler = {
        "train": cmd_train,
        "pack": cmd_pack,
        "unpack": cmd_unpack,
        "report": cmd_report,
    }[args.command]
    try:
        return handler(args)
    except DictPoolError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
