"""单文件静态报告：原生 HTML + ES module，数据内联，无外部资源。"""

import json
import os

from .constants import (
    COVERAGE_THRESHOLD,
    DICT_MAX_BYTES,
    REPORT_HIT_RATE_MIN,
    REPORT_RATIO_MAX,
)
from .engine import read_archive

_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dictpool 报告</title>
<style>
  :root { color-scheme: light; }
  body { font-family: system-ui, "PingFang SC", "Microsoft YaHei", sans-serif;
         margin: 2rem auto; max-width: 1080px; padding: 0 1rem; color: #1c1c1e; }
  h1 { font-size: 1.4rem; }
  h2 { font-size: 1.1rem; margin-top: 2rem; }
  .meta { color: #555; font-size: 0.85rem; word-break: break-all; }
  table { border-collapse: collapse; width: 100%; font-size: 0.9rem;
          font-variant-numeric: tabular-nums; }
  th, td { border: 1px solid #d8d8dc; padding: 0.35rem 0.6rem; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { background: #f2f2f5; }
  tr.fail td { background: #ffe3e3; color: #a40000; }
  tr.fail td:first-child::after { content: " \\26A0"; }
  .ok { color: #0a7a2f; font-weight: 600; }
  .bad { color: #c00; font-weight: 700; }
  select { font-size: 0.9rem; margin-bottom: 0.5rem; }
</style>
</head>
<body>
<h1>dictpool 压缩报告</h1>
<p class="meta" id="meta"></p>
<p class="meta" id="gates"></p>

<h2>逐批汇总</h2>
<table id="summary">
  <thead><tr>
    <th>批次</th><th>块数</th><th>原始字节</th><th>压缩后字节</th>
    <th>命中率</th><th>压缩率</th><th>字典字节</th><th>判定</th>
  </tr></thead>
  <tbody></tbody>
</table>

<h2>逐块明细</h2>
<select id="filter"></select>
<table id="detail">
  <thead><tr>
    <th>块 ID</th><th>原始字节</th><th>字典索引</th><th>覆盖度</th>
    <th>压缩后字节</th><th>命中</th>
  </tr></thead>
  <tbody></tbody>
</table>

<script type="application/json" id="dictpool-data">
__DATA__
</script>
<script type="module">
const data = JSON.parse(document.getElementById("dictpool-data").textContent);
const t = data.thresholds;
const pct = (x) => (x * 100).toFixed(1) + "%";

document.getElementById("meta").textContent =
  `pool_id: ${data.pool_id} ｜ 批次数: ${data.batches.length} ｜ 块数: ${data.blocks.length}`;
document.getElementById("gates").textContent =
  `门槛：命中率 ≥ ${pct(t.hit_rate_min)} ｜ 压缩率 ≤ ${t.ratio_max.toFixed(2)} ` +
  `｜ 字典 ≤ ${t.dict_max_bytes} B（覆盖门限 ${t.coverage_threshold}）`;

const summaryBody = document.querySelector("#summary tbody");
for (const b of data.batches) {
  const tr = document.createElement("tr");
  if (!b.ok) tr.className = "fail";
  const cells = [
    b.name, b.blocks, b.raw_bytes, b.packed_bytes,
    pct(b.hit_rate), b.ratio.toFixed(3), b.dict_bytes,
  ];
  for (const c of cells) {
    const td = document.createElement("td");
    td.textContent = c;
    tr.appendChild(td);
  }
  const verdict = document.createElement("td");
  verdict.textContent = b.ok ? "达标" : "不达标";
  verdict.className = b.ok ? "ok" : "bad";
  tr.appendChild(verdict);
  summaryBody.appendChild(tr);
}

const filter = document.getElementById("filter");
const all = document.createElement("option");
all.value = ""; all.textContent = "全部批次";
filter.appendChild(all);
for (const b of data.batches) {
  const opt = document.createElement("option");
  opt.value = b.name; opt.textContent = b.name;
  filter.appendChild(opt);
}

const detailBody = document.querySelector("#detail tbody");
function renderDetail(name) {
  detailBody.textContent = "";
  for (const blk of data.blocks) {
    if (name && !blk.block_id.startsWith(name + "/")) continue;
    const tr = document.createElement("tr");
    const cells = [
      blk.block_id, blk.raw_len,
      blk.dict_index === null ? "—" : blk.dict_index,
      blk.coverage.toFixed(4), blk.packed_len,
      blk.hit ? "是" : "否",
    ];
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.appendChild(td);
    }
    detailBody.appendChild(tr);
  }
}
filter.addEventListener("change", () => renderDetail(filter.value));
renderDetail("");
</script>
</body>
</html>
"""


def aggregate_batches(index):
    """从归档索引（引擎输出）汇总逐批指标。"""
    dict_sizes = {d["index"]: d["dict_bytes"] for d in index["dicts"]}
    batches = {}
    order = []
    for blk in index["blocks"]:
        name = blk["block_id"].split("/", 1)[0]
        if name not in batches:
            batches[name] = {
                "name": name,
                "blocks": 0,
                "raw_bytes": 0,
                "packed_bytes": 0,
                "hits": 0,
                "dict_bytes": 0,
            }
            order.append(name)
        agg = batches[name]
        agg["blocks"] += 1
        agg["raw_bytes"] += blk["raw_len"]
        agg["packed_bytes"] += blk["packed_len"]
        if blk["hit"]:
            agg["hits"] += 1
            agg["dict_bytes"] = max(
                agg["dict_bytes"], dict_sizes.get(blk["dict_index"], 0)
            )
    result = []
    for name in order:
        agg = batches[name]
        total = agg["blocks"]
        raw = agg["raw_bytes"]
        hit_rate = agg["hits"] / total if total else 0.0
        ratio = agg["packed_bytes"] / raw if raw else 0.0
        ok = (
            hit_rate >= REPORT_HIT_RATE_MIN
            and ratio <= REPORT_RATIO_MAX
            and agg["dict_bytes"] <= DICT_MAX_BYTES
        )
        result.append(
            {
                "name": name,
                "blocks": total,
                "raw_bytes": raw,
                "packed_bytes": agg["packed_bytes"],
                "hits": agg["hits"],
                "hit_rate": hit_rate,
                "ratio": ratio,
                "dict_bytes": agg["dict_bytes"],
                "ok": ok,
            }
        )
    return result


def render_report(index):
    data = {
        "pool_id": index["pool_id"],
        "thresholds": {
            "hit_rate_min": REPORT_HIT_RATE_MIN,
            "ratio_max": REPORT_RATIO_MAX,
            "dict_max_bytes": DICT_MAX_BYTES,
            "coverage_threshold": COVERAGE_THRESHOLD,
        },
        "batches": aggregate_batches(index),
        "blocks": [
            {
                "block_id": blk["block_id"],
                "dict_index": blk["dict_index"],
                "raw_len": blk["raw_len"],
                "packed_len": blk["packed_len"],
                "coverage": blk["coverage"],
                "hit": blk["hit"],
            }
            for blk in index["blocks"]
        ],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return _PAGE.replace("__DATA__", payload)


def write_report(archive_dir, out_path):
    index = read_archive(archive_dir)
    html = render_report(index)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    return out_path
