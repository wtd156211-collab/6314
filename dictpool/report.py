"""Render the single-file static report from engine output (archive index)."""

import hashlib
import json
import os

from .common import (
    COVERAGE_THRESHOLD,
    DICT_MAX_BYTES,
    DataValidationError,
    UsageError,
    atomic_write,
)
from .pool import POOL_JSON, load_pool

INDEX_JSON = "index.json"


def _load_index(archive_dir):
    path = os.path.join(archive_dir, INDEX_JSON)
    if not os.path.isfile(path):
        raise UsageError("archive index not found: %s" % path)
    with open(path, "rb") as f:
        raw = f.read()
    try:
        meta = json.loads(raw.decode("utf-8"))
        return meta["pool_id"], meta["blocks"]
    except (ValueError, KeyError, UnicodeDecodeError):
        raise DataValidationError("malformed archive index: %s" % path)


def _batch_stats(entries):
    """Per-batch hit rate and compression ratio from block records."""
    order = []
    stats = {}
    for e in entries:
        batch = e["block_id"].split("/", 1)[0]
        if batch not in stats:
            stats[batch] = {"blocks": 0, "hits": 0, "raw": 0, "packed": 0}
            order.append(batch)
        s = stats[batch]
        s["blocks"] += 1
        s["hits"] += 1 if e["hit"] else 0
        s["raw"] += e["raw_len"]
        s["packed"] += e["packed_len"]
    rows = []
    for batch in order:
        s = stats[batch]
        rows.append(
            {
                "batch": batch,
                "blocks": s["blocks"],
                "raw_bytes": s["raw"],
                "packed_bytes": s["packed"],
                "hit_rate": s["hits"] / s["blocks"] if s["blocks"] else 0.0,
                "ratio": s["packed"] / s["raw"] if s["raw"] else 0.0,
            }
        )
    return rows


def _try_load_pool(archive_dir, pool_dir):
    if pool_dir is None:
        guess = os.path.join(os.path.dirname(os.path.normpath(archive_dir)), "pool")
        if os.path.isfile(os.path.join(guess, POOL_JSON)):
            pool_dir = guess
        else:
            return None
    try:
        pool = load_pool(pool_dir)
    except Exception:
        return None
    return [
        {
            "index": d["index"],
            "batch": d["batch"],
            "dict_bytes": len(d["bytes"]),
            "sha256": hashlib.sha256(d["bytes"]).hexdigest(),
        }
        for d in pool["dictionaries"]
    ]


def build_report(archive_dir, out_path, pool_dir=None,
                 hit_rate_min=COVERAGE_THRESHOLD, ratio_max=1.0):
    pool_id, entries = _load_index(archive_dir)
    batches = _batch_stats(entries)
    dictionaries = _try_load_pool(archive_dir, pool_dir)

    data = {
        "pool_id": pool_id,
        "thresholds": {
            "hit_rate_min": hit_rate_min,
            "ratio_max": ratio_max,
            "coverage_gate": COVERAGE_THRESHOLD,
            "dict_max_bytes": DICT_MAX_BYTES,
        },
        "batches": batches,
        "dictionaries": dictionaries,
        "blocks": entries,
    }
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    doc = _HTML_TEMPLATE.replace("__TITLE__", "dictpool report").replace(
        "__DATA__", payload
    )
    atomic_write(out_path, doc.encode("utf-8"))
    return batches


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light; }
  body { font: 14px/1.5 system-ui, sans-serif; margin: 24px; color: #1c1c1e; }
  h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 28px; }
  table { border-collapse: collapse; margin: 8px 0 16px; min-width: 60%; }
  th, td { border: 1px solid #d0d0d4; padding: 4px 10px; text-align: right; }
  th { background: #f2f2f4; } td:first-child, th:first-child { text-align: left; }
  tr.fail td { background: #ffe3e3; color: #a40000; font-weight: 600; }
  tr.ok td { background: #eaf7ea; }
  .meta { color: #555; font-size: 13px; }
  code { background: #f2f2f4; padding: 1px 4px; border-radius: 3px; }
  .mono { font-family: ui-monospace, monospace; font-size: 12px; }
</style>
</head>
<body>
<h1>Dictionary pool report</h1>
<p class="meta">pool_id <code id="pool"></code> · generated from
<code>var/archive/index.json</code> (engine output, no recomputation).</p>
<h2>Per-batch summary</h2>
<table id="batches">
  <thead><tr>
    <th>batch</th><th>blocks</th><th>raw bytes</th><th>packed bytes</th>
    <th>hit rate</th><th>hit bar</th><th>ratio</th><th>ratio bar</th><th>status</th>
  </tr></thead>
  <tbody></tbody>
</table>
<h2>Pool dictionaries vs size cap</h2>
<table id="dicts">
  <thead><tr>
    <th>index</th><th>trained from</th><th>dict bytes</th>
    <th>cap</th><th>headroom</th><th>sha256</th>
  </tr></thead>
  <tbody></tbody>
</table>
<p class="meta" id="nodict" hidden>pool metadata not found next to the archive;
pass <code>--pool</code> to include dictionary sizes.</p>
<h2>Per-block detail</h2>
<table id="blocks">
  <thead><tr>
    <th>block</th><th>raw bytes</th><th>dict</th><th>coverage</th>
    <th>packed bytes</th><th>hit</th>
  </tr></thead>
  <tbody></tbody>
</table>
<script type="module">
const data = JSON.parse(document.getElementById('data').textContent);
const t = data.thresholds;
document.getElementById('pool').textContent = data.pool_id;
const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct = (x) => (x * 100).toFixed(1) + '%';
const f3 = (x) => x.toFixed(3);

const bt = document.querySelector('#batches tbody');
for (const b of data.batches) {
  const ok = b.hit_rate >= t.hit_rate_min && b.ratio <= t.ratio_max;
  const tr = document.createElement('tr');
  tr.className = ok ? 'ok' : 'fail';
  tr.innerHTML = `<td>${esc(b.batch)}</td><td>${b.blocks}</td>` +
    `<td>${b.raw_bytes}</td><td>${b.packed_bytes}</td>` +
    `<td>${pct(b.hit_rate)}</td><td>${pct(t.hit_rate_min)}</td>` +
    `<td>${f3(b.ratio)}</td><td>${f3(t.ratio_max)}</td>` +
    `<td>${ok ? 'PASS' : 'FAIL'}</td>`;
  bt.appendChild(tr);
}

const dt = document.querySelector('#dicts tbody');
if (data.dictionaries) {
  for (const d of data.dictionaries) {
    const tr = document.createElement('tr');
    tr.className = d.dict_bytes <= t.dict_max_bytes ? 'ok' : 'fail';
    tr.innerHTML = `<td>${d.index}</td><td>${esc(d.batch ?? '')}</td>` +
      `<td>${d.dict_bytes}</td><td>${t.dict_max_bytes}</td>` +
      `<td>${t.dict_max_bytes - d.dict_bytes}</td>` +
      `<td class="mono">${esc(d.sha256.slice(0, 16))}…</td>`;
    dt.appendChild(tr);
  }
} else {
  document.getElementById('nodict').hidden = false;
  document.getElementById('dicts').hidden = true;
}

const kt = document.querySelector('#blocks tbody');
for (const e of data.blocks) {
  const tr = document.createElement('tr');
  if (!e.hit) tr.className = 'fail';
  tr.innerHTML = `<td class="mono">${esc(e.block_id)}</td>` +
    `<td>${e.raw_len}</td>` +
    `<td>${e.dict_index === null ? '—' : 'd' + e.dict_index}</td>` +
    `<td>${f3(e.coverage)}</td><td>${e.packed_len}</td>` +
    `<td>${e.hit ? 'hit' : 'miss'}</td>`;
  kt.appendChild(tr);
}
</script>
<script id="data" type="application/json">__DATA__</script>
</body>
</html>
"""
