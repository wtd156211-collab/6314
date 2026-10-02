"""命令行入口：python -m dictpool {train|pack|unpack|report}。"""

import argparse
import os
import sys

from . import __version__
from .engine import (
    DictPoolError,
    pack_batches,
    read_pool,
    train_pool,
    unpack_archive,
)
from .constants import POOL_CAPACITY
from .report import write_report, aggregate_batches


class _Parser(argparse.ArgumentParser):
    """用法错误统一用退出码 1（argparse 默认是 2）。"""

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, "%s: error: %s\n" % (self.prog, message))


def _batch_dirs(paths):
    batch_dirs = []
    for path in paths:
        if not os.path.isdir(path):
            raise DictPoolError("块目录不存在: %s" % path)
        batch = os.path.basename(os.path.normpath(path))
        batch_dirs.append((batch, path))
    return batch_dirs


def _add_blocks(parser, required=True):
    parser.add_argument(
        "--blocks",
        action="append",
        required=required,
        metavar="DIR",
        help="块目录，可重复；批名取目录名",
    )
    parser.add_argument("--out", required=True, metavar="DIR", help="输出目录")


def _cmd_train(args):
    batch_dirs = _batch_dirs(args.blocks)
    if len(batch_dirs) > POOL_CAPACITY:
        raise DictPoolError("训练批数量超过池容量 %d" % POOL_CAPACITY)
    pool_id, dicts = train_pool(batch_dirs)
    from .engine import write_pool

    write_pool(args.out, pool_id, dicts)
    print("pool_id: %s" % pool_id)
    for entry in dicts:
        print(
            "  d%d %s sample=%d dict=%d sha256=%s"
            % (
                entry["index"],
                entry["batch"],
                entry["sample_bytes"],
                entry["dict_bytes"],
                entry["sha256"],
            )
        )
    return 0


def _cmd_pack(args):
    pool_id, dicts = read_pool(args.pool)
    batch_dirs = _batch_dirs(args.blocks)
    blocks = pack_batches(pool_id, dicts, batch_dirs, args.out)
    print("packed %d blocks into %s (pool_id=%s)" % (len(blocks), args.out, pool_id))
    batches = aggregate_batches({"pool_id": pool_id, "dicts": dicts, "blocks": blocks})
    for agg in batches:
        print(
            "  %s blocks=%d hit_rate=%.3f ratio=%.3f"
            % (agg["name"], agg["blocks"], agg["hit_rate"], agg["ratio"])
        )
    return 0


def _cmd_unpack(args):
    pool_id, dicts = read_pool(args.pool)
    count = unpack_archive(pool_id, dicts, args.archive, args.out)
    print("restored %d blocks into %s" % (count, args.out))
    return 0


def _cmd_report(args):
    path = write_report(args.archive, args.out)
    print("report: %s" % path)
    return 0


def build_parser():
    parser = _Parser(prog="python -m dictpool", description="压缩字典训练与复用")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="训练字典池")
    _add_blocks(p_train)
    p_train.set_defaults(func=_cmd_train)

    p_pack = sub.add_parser("pack", help="逐块选字典压缩")
    p_pack.add_argument("--pool", required=True, metavar="DIR")
    _add_blocks(p_pack)
    p_pack.set_defaults(func=_cmd_pack)

    p_unpack = sub.add_parser("unpack", help="校验并还原归档")
    p_unpack.add_argument("--pool", required=True, metavar="DIR")
    p_unpack.add_argument("--archive", required=True, metavar="DIR")
    p_unpack.add_argument("--out", required=True, metavar="DIR")
    p_unpack.set_defaults(func=_cmd_unpack)

    p_report = sub.add_parser("report", help="生成单文件 HTML 报告")
    p_report.add_argument("--archive", required=True, metavar="DIR")
    p_report.add_argument("--out", required=True, metavar="HTML")
    p_report.set_defaults(func=_cmd_report)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DictPoolError as exc:
        print("dictpool: error: %s" % exc, file=sys.stderr)
        return exc.exit_code
