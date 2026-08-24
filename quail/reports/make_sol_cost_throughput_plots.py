"""Issue #26: the two columns from the report table
(quail.bench.sol_report.build_rows) that didn't get a chart yet -
dollar cost and throughput (docs/s, tokens/s). Reads the same
committed, validated summary the rest of this report's numbers come
from - no new arithmetic, just plotting columns build_rows() already
computes.

Run from the quail/ directory:

    uv run --with matplotlib python reports/make_sol_cost_throughput_plots.py

Writes reports/plots/sol_cost_by_query.png,
reports/plots/sol_docs_per_s.png, reports/plots/sol_tokens_per_s.png.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
RESULTS = ROOT / "results"
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, DARK, GREEN, ORANGE, TEAL

sys.path.insert(0, str(ROOT / "quail"))
from bench.sol_report import build_rows

SUITE_PATH = RESULTS / "sol_check_sf0.1_4b_corrected.json"


def load_rows():
    suite = json.loads(SUITE_PATH.read_text())
    return [r for r in build_rows(suite) if not r["error"]]


def fig_cost(rows):
    labels = [r["query"] for r in rows]
    cold = [r["cost_cold"] for r in rows]
    warm = [r["cost_warm"] for r in rows]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.3), 4.5))
    ax.bar(x - width / 2, cold, width, color=ORANGE, label="cold (incl. boot)")
    ax.bar(x + width / 2, warm, width, color=TEAL, label="warm")
    for xi, c, w in zip(x, cold, warm):
        ax.text(xi - width / 2, c, f"${c:.3f}", ha="center", va="bottom",
                fontsize=8)
        ax.text(xi + width / 2, w, f"${w:.3f}", ha="center", va="bottom",
                fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("cost ($)")
    ax.set_title("Dollar cost per query, cold vs. warm")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "sol_cost_by_query.png")
    plt.close(fig)


def fig_rate(rows, key, ylabel, title, color, out_name, fmt):
    labels = [r["query"] for r in rows]
    vals = [r[key] for r in rows]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4.2))
    ax.bar(x, vals, color=color, width=0.55)
    for xi, v in zip(x, vals):
        ax.text(xi, v, fmt(v), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / out_name)
    plt.close(fig)


if __name__ == "__main__":
    rows = load_rows()
    fig_cost(rows)
    fig_rate(rows, "docs_per_s_warm", "documents/second",
             "Documents processed per second (warm)", BLUE,
             "sol_docs_per_s.png", lambda v: f"{v:,.0f}")
    fig_rate(rows, "tokens_per_s_warm", "tokens/second",
             "Fresh tokens processed per second (warm)", GREEN,
             "sol_tokens_per_s.png", lambda v: f"{v/1000:.0f}k")
    print(f"[make_sol_cost_throughput_plots] wrote 3 PNGs to {OUT}, "
         f"from {SUITE_PATH.relative_to(ROOT)}")
