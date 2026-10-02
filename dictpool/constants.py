"""固定参数（见 README 第 2 节，不得调参）。"""

BLOCK_EXT = ".blk"
PACKED_EXT = ".dpk"
RESTORED_EXT = ".blk"

MAGIC = b"DPK1"
NO_DICT = 0xFFFF
HEADER_LEN = 14  # magic 4 + dict_index 2 + raw_len 8

SAMPLE_PER_BLOCK = 8192        # 每块采样字节（取块首）
SAMPLE_CAP_PER_BATCH = 131072  # 单批采样上限
FRAGMENT_LEN = 16              # 片段长度
FRAGMENT_MIN_COUNT = 2         # 片段最小计数
DICT_MAX_BYTES = 8192          # 字典上限
DICT_MAX_FRAGMENTS = 512       # 字典上限对应的片段数
POOL_CAPACITY = 4              # 池容量（字典索引 0..3）
COVERAGE_THRESHOLD = 0.25      # 覆盖门限（命中判定阈值）

MAX_BLOCK_BYTES = 8 * 1024 * 1024  # 单块上限 8 MiB

ZLIB_LEVEL = 9
ZLIB_WBITS = -15  # 原始 deflate

# 报告门槛（与 README 固定参数一致）
REPORT_HIT_RATE_MIN = COVERAGE_THRESHOLD
REPORT_RATIO_MAX = 1.0

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_POOL_MISMATCH = 2
EXIT_DATA_MISMATCH = 3
