"""Issue #26: cost and throughput, charted against both dimensions
sol_report.build_rows() computes for each - the cold/warm split in
what actually happened, and the SOL-derived best case for that exact
workload. Reads the same committed, validated summary the rest of
this report's numbers come from - no new arithmetic, just plotting
columns build_rows() already computes.

Time and cost share a direction: sol_s is a FLOOR (nothing runs
faster), so sol-derived cost is a floor too (nothing costs less).
Throughput is a rate - the reciprocal of time - so sol-derived
docs/s and tokens/s are a CEILING (nothing goes faster than the
floor-time rate). See sol_report.build_rows()'s docstring for the
full reasoning.

Run from the quail/ directory:

    uv run --with matplotlib python reports/make_sol_cost_throughput_plots.py

Writes reports/plots/sol_cost_by_query.png,
reports/plots/sol_cost_vs_floor.png, reports/plots/sol_docs_per_s.png,
reports/plots/sol_tokens_per_s.png.
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
from plot_colors import BLUE, DARK, GRAY, GREEN, ORANGE, TEAL

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


def fig_pair(rows, measured_key, sol_key, ylabel, title, sol_color,
            sol_label, out_name, fmt):
    """Grouped bars: what actually happened (warm, gray) against what
    sol_s implies for that exact workload (sol_color) - the same
    measured-vs-floor pattern plots/sol_report_sf0.1_4b.png uses for
    wall time, extended to cost and throughput."""
    labels = [r["query"] for r in rows]
    measured = [r[measured_key] for r in rows]
    sol = [r[sol_key] for r in rows]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.3), 4.5))
    ax.bar(x - width / 2, measured, width, color=GRAY, label="measured (warm)")
    ax.bar(x + width / 2, sol, width, color=sol_color, label=sol_label)
    for xi, m, s in zip(x, measured, sol):
        ax.text(xi - width / 2, m, fmt(m), ha="center", va="bottom",
                fontsize=8)
        ax.text(xi + width / 2, s, fmt(s), ha="center", va="bottom",
                fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / out_name)
    plt.close(fig)


if __name__ == "__main__":
    rows = load_rows()
    fig_cost(rows)
    fig_pair(rows, "cost_warm", "cost_sol_floor", "cost ($)",
             "Cost: measured vs. the SOL floor (lower is the limit)",
             BLUE, "SOL floor", "sol_cost_vs_floor.png",
             lambda v: f"${v:.3f}")
    fig_pair(rows, "docs_per_s_warm", "docs_per_s_sol_ceiling",
             "documents/second",
             "Docs/s: measured vs. the SOL ceiling (higher is the limit)",
             BLUE, "SOL ceiling", "sol_docs_per_s.png",
             lambda v: f"{v:,.0f}")
    fig_pair(rows, "tokens_per_s_warm", "tokens_per_s_sol_ceiling",
             "tokens/second",
             "Tok/s: measured vs. the SOL ceiling (higher is the limit)",
             GREEN, "SOL ceiling", "sol_tokens_per_s.png",
             lambda v: f"{v/1000:.0f}k")
    print(f"[make_sol_cost_throughput_plots] wrote 4 PNGs to {OUT}, "
         f"from {SUITE_PATH.relative_to(ROOT)}")
