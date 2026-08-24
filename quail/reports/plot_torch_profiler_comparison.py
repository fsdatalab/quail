"""Issue #26 verification, phase 3 (the payoff): plots the real
torch.profiler kernel-class split (tests/gpu/torch_profiler_compare.py)
against what sol_seconds_breakdown() predicts, per query. Where the
formula's internal structure diverges from what the real kernels
actually spend time on - not just whether the aggregate sol_s stays
under wall_s, which section 12's invariant already guarantees.

Run from the quail/ directory, after torch_profiler_compare.py has
written results/torch_profiler_compare.json:

    uv run --with matplotlib python reports/plot_torch_profiler_comparison.py

Writes reports/plots/torch_profiler_vs_formula.png.
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
from plot_colors import BLUE, GRAY, GREEN, DARK, ORANGE, RED, TEAL

BUCKETS = ["projection", "elementwise", "attention"]
BUCKET_COLOR = {"projection": BLUE, "elementwise": ORANGE, "attention": GREEN}


def load(name="torch_profiler_compare.json"):
    with open(RESULTS / name) as f:
        return json.load(f)


def renormalized_measured(row):
    """measured_pct includes an "unmodeled" and "other" share (I/O,
    RoPE, KV-scatter, sampling kernels - see torch_profiler_compare.py's
    KERNEL_CLASS_RULES) that no SOL term claims to price. Comparing
    that directly against formula_pct (which sums to 100% over just
    the three modeled buckets) isn't apples to apples - renormalize
    measured to the same three-bucket basis so the comparison is
    about STRUCTURE (where does the modeled time go), not diluted by
    how big the unmodeled slice happened to be for this query."""
    m = row["measured_pct"]
    modeled_total = sum(m.get(b, 0.0) for b in BUCKETS)
    if modeled_total == 0:
        return {b: 0.0 for b in BUCKETS}
    return {b: m.get(b, 0.0) / modeled_total * 100 for b in BUCKETS}


def fig_comparison(rows):
    queries = [r["query"] for r in rows]
    n = len(queries)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    width = 0.32
    x = np.arange(len(BUCKETS))
    for ax, row in zip(axes, rows):
        measured = renormalized_measured(row)
        formula = row["formula_pct"]
        m_vals = [measured[b] for b in BUCKETS]
        f_vals = [formula[b] for b in BUCKETS]
        ax.bar(x - width / 2, m_vals, width, color=GRAY,
              label="measured (profiler)")
        ax.bar(x + width / 2, f_vals, width, color=BLUE,
              label="formula (sol_breakdown)")
        unmodeled = row["measured_pct"].get("unmodeled", 0.0)
        ax.set_title(f"{row['query']}\n({unmodeled:.0f}% unmodeled, "
                     f"excluded above)", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["proj", "elem", "attn"], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("% of modeled time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
              bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Real kernel time vs. SOL formula's predicted split",
                 y=1.14, fontsize=12, fontweight="bold")
    fig.savefig(OUT / "torch_profiler_vs_formula.png", bbox_inches="tight")
    plt.close(fig)


def fig_unmodeled_share(rows):
    """How much of each query's total measured kernel time falls into
    the "unmodeled" bucket at all - I/O, RoPE, KV-scatter bookkeeping,
    sampling kernels. Not a formula error (SOL only ever claimed to
    model the forward pass's four terms), but worth seeing per query
    since it's the ceiling on how tight the profiler comparison above
    can ever be."""
    queries = [r["query"] for r in rows]
    unmodeled = [r["measured_pct"].get("unmodeled", 0.0) for r in rows]
    other = [r["measured_pct"].get("other", 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(queries, unmodeled, color=RED, label="unmodeled (I/O, RoPE, "
          "KV-scatter, sampling)")
    ax.bar(queries, other, bottom=unmodeled, color=GRAY, label="other "
          "(uncategorized)")
    ax.set_ylabel("% of total measured kernel time")
    ax.set_title("What SOL's formula doesn't claim to price at all")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "torch_profiler_unmodeled_share.png")
    plt.close(fig)


def fig_totals(rows):
    """Total query time, three ways, per query: wall_s (this system's
    own end-to-end measurement - Python, kernel launches, everything),
    the torch.profiler kernel sum (total_kernel_us - only time the GPU
    was actually executing a kernel, so it's <= wall_s by construction),
    and sol_s (the formula's floor). Expected ordering is sol_s <=
    kernel time <= wall_s: the gap from sol_s up to kernel time is real
    kernels running below their peak rate (the attention finding
    above); the gap from kernel time up to wall_s is time spent outside
    any GPU kernel at all (Python, scheduling, kernel-launch gaps).

    wall_s here is measured under torch.profiler instrumentation, which
    adds its own overhead (torch_profiler_compare.py's own docstring
    flags this) - read it as "roughly wall clock while profiled," not
    as directly comparable to the un-instrumented cold/warm wall_s
    elsewhere in this report."""
    queries = [r["query"] for r in rows]
    n = len(queries)
    wall = [r["wall_s"] for r in rows]
    kernel = [r["total_kernel_us"] / 1e6 for r in rows]
    sol = [r["sol_s"] for r in rows]

    x = np.arange(n)
    width = 0.26
    fig, ax = plt.subplots(figsize=(2.6 * n, 4.5))
    ax.bar(x - width, wall, width, color=GRAY, label="wall_s (our system)")
    ax.bar(x, kernel, width, color=TEAL, label="kernel time (torch.profiler)")
    ax.bar(x + width, sol, width, color=BLUE, label="sol_s (our SOL estimate)")
    for xi, w, k, s in zip(x, wall, kernel, sol):
        ax.text(xi - width, w, f"{w:.1f}s", ha="center", va="bottom",
               fontsize=7.5)
        ax.text(xi, k, f"{k:.1f}s", ha="center", va="bottom", fontsize=7.5)
        ax.text(xi + width, s, f"{s:.1f}s", ha="center", va="bottom",
               fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(queries, fontsize=9)
    ax.set_ylabel("seconds")
    ax.set_title("Total query time: our system vs. profiled kernel time "
                "vs. the SOL floor")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "torch_profiler_totals.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = load()
    fig_comparison(rows)
    fig_unmodeled_share(rows)
    fig_totals(rows)
    print(f"[plot_torch_profiler_comparison] wrote 3 PNGs to {OUT}")
