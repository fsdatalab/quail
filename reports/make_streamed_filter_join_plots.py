r"""Plot materialized versus streamed filter-to-join execution.

Pull the comparison directory off the volume (the stamp is in the
report), then run this script on it:

    W=/tmp/quail-streamed-filter-join; mkdir -p "$W"
    uv run modal volume get quail-results \
      ablations/streamed-filter-join-<stamp> "$W" --force
    uv run --with matplotlib python \
      reports/make_streamed_filter_join_plots.py \
      "$W/streamed-filter-join-<stamp>"

Writes plots/streamed_filter_join.png: query seconds per query and
configuration on the left, recomputed KV tokens in the middle, and
fresh input tokens on the right. Percentages are derived here from the
saved summaries.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, GRAY  # noqa: E402

CONFIGS = (("Materialized survivors", "materialized", GRAY),
           ("Streamed survivors", "streamed", BLUE))
FIELDS = (("seconds", "seconds", "Query time, excluding startup", "{:.1f}"),
          ("regret", "tokens", "Recomputed KV tokens", "{:,.0f}"),
          ("fresh_tokens", "tokens", "Fresh input tokens", "{:,.0f}"))


def load(workdir):
    setup = json.loads((workdir / "setup.json").read_text())
    rows = {}
    for _label, key, _color in CONFIGS:
        for qid in setup["queries"]:
            summary = json.loads(
                (workdir / key / qid / "summary.json").read_text())
            report = summary["report"]
            rows[(key, qid)] = dict(
                seconds=summary["query_seconds"],
                regret=report["regret_tokens"],
                fresh_tokens=report["fresh_tokens"],
                pairs=summary["evaluated_document_pairs"],
                hits=report["kv_manager"]["join_anchor_hits"],
                misses=report["kv_manager"]["join_anchor_misses"],
                rows=summary["rows"])
    return setup["queries"], rows


def bars(ax, queries, rows, field, unit, title, fmt):
    width = 0.38
    x = np.arange(len(queries))
    top = max(rows[(k, q)][field] for _l, k, _c in CONFIGS for q in queries)
    for i, (label, key, color) in enumerate(CONFIGS):
        values = [rows[(key, q)][field] for q in queries]
        pos = x + (i - 0.5) * width
        ax.bar(pos, values, width, color=color, label=label)
        for p, v in zip(pos, values):
            ax.text(p, v, fmt.format(v) if v else "0", ha="center",
                    va="bottom", fontsize=7)
    for xi, q in zip(x, queries):
        before = rows[("materialized", q)][field]
        after = rows[("streamed", q)][field]
        if before:
            change = (after - before) / before * 100
            ax.text(xi, max(before, after) + top * 0.09,
                    f"{change:+.1f}%", ha="center", va="bottom",
                    fontsize=8, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.set_ylabel(unit)
    ax.set_title(title)
    ax.set_ylim(0, top * 1.28)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} WORKDIR")
    workdir = Path(sys.argv[1])
    queries, rows = load(workdir)
    print("| Query | Configuration | Query seconds | Recomputed KV tokens | "
          "Fresh tokens | Anchor KV hits | Anchor KV misses | Pairs | Rows |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for q in queries:
        for label, key, _color in CONFIGS:
            r = rows[(key, q)]
            print(f"| {q} | {label} | {r['seconds']:.2f} | {r['regret']:,} | "
                  f"{r['fresh_tokens']:,} | {r['hits']:,} | {r['misses']:,} | "
                  f"{r['pairs']:,} | {r['rows']:,} |")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (field, unit, title, fmt) in zip(axes, FIELDS):
        bars(ax, queries, rows, field, unit, title, fmt)
    axes[0].legend(loc="upper left")
    fig.suptitle("Filter survivors streamed into the join on one H100, "
                 "Qwen3 4B fp8, sf=0.1", fontsize=11, y=1.02)
    fig.subplots_adjust(wspace=0.3)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "streamed_filter_join.png", dpi=300)
    print(f"wrote {OUT / 'streamed_filter_join.png'}")


if __name__ == "__main__":
    main()
