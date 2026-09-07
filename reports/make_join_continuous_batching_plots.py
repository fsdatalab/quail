r"""Plot pre-planned versus continuous join batching on IMDB-8, FEV-7, FEV-9.

Pull the comparison directory off the volume (the stamp is in the
report), then run this script on it:

    W=/tmp/quail-join-continuous-batching; mkdir -p "$W"
    uv run modal volume get quail-results \
      ablations/join-continuous-batching-<stamp> "$W" --force
    uv run --with matplotlib python \
      reports/make_join_continuous_batching_plots.py \
      "$W/join-continuous-batching-<stamp>"

Writes plots/join_continuous_batching.png: query seconds per query and
configuration on the left, and the seconds spent inside the join nodes
alone on the right. Percentages are derived here from the saved
summaries.
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

CONFIGS = (("Pre-planned groups", "preplanned", GRAY),
           ("Continuous batching", "continuous", BLUE))


def load(workdir):
    setup = json.loads((workdir / "setup.json").read_text())
    rows = {}
    for _label, key, _color in CONFIGS:
        for qid in setup["queries"]:
            summary = json.loads(
                (workdir / key / qid / "summary.json").read_text())
            join_seconds = sum(
                node["metrics"]["wall_s"]
                for node in summary["nodes"].values()
                if node["type"].endswith("anchored_join") and node["metrics"])
            rows[(key, qid)] = dict(
                seconds=summary["query_seconds"], join_seconds=join_seconds,
                pairs=summary["evaluated_document_pairs"],
                fresh_tokens=summary["report"]["fresh_tokens"],
                rows=summary["rows"])
    return setup["queries"], rows


def bars(ax, queries, rows, field, title):
    width = 0.38
    x = np.arange(len(queries))
    for i, (label, key, color) in enumerate(CONFIGS):
        values = [rows[(key, q)][field] for q in queries]
        pos = x + (i - 0.5) * width
        ax.bar(pos, values, width, color=color, label=label)
        for p, v in zip(pos, values):
            ax.text(p, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, q in zip(x, queries):
        before = rows[("preplanned", q)][field]
        after = rows[("continuous", q)][field]
        change = (after - before) / before * 100
        ax.text(xi, max(before, after) * 1.09, f"{change:+.1f}%",
                ha="center", va="bottom", fontsize=9, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.set_ylabel("seconds")
    ax.set_title(title)
    ax.set_ylim(0, max(rows[(k, q)][field]
                       for _l, k, _c in CONFIGS for q in queries) * 1.25)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} WORKDIR")
    workdir = Path(sys.argv[1])
    queries, rows = load(workdir)
    print("| Query | Configuration | Query seconds | Join node seconds | "
          "Pairs | Fresh tokens | Rows |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for q in queries:
        for label, key, _color in CONFIGS:
            r = rows[(key, q)]
            print(f"| {q} | {label} | {r['seconds']:.2f} | "
                  f"{r['join_seconds']:.2f} | {r['pairs']:,} | "
                  f"{r['fresh_tokens']:,} | {r['rows']:,} |")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    bars(left, queries, rows, "seconds", "Query time, excluding startup")
    bars(right, queries, rows, "join_seconds", "Time inside the join nodes")
    left.legend(loc="upper left")
    fig.suptitle("Join batching on one H100, Qwen3 4B fp8, sf=0.1",
                 fontsize=11, y=1.02)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "join_continuous_batching.png", dpi=300)
    print(f"wrote {OUT / 'join_continuous_batching.png'}")


if __name__ == "__main__":
    main()
