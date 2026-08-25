"""Plots for 2026-08-24-join-gate-and-reshard.md, from the committed
summary results/join_bench.json.

    uv run --with matplotlib python reports/make_join_bench_plots.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.style.use(Path(__file__).parent / "quail.mplstyle")
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_colors import BLUE, GRAY, GREEN  # noqa: E402

HERE = Path(__file__).parent
DATA = json.loads((HERE.parent / "results" / "join_bench.json")
                  .read_text())
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)


def gate_plot():
    g = DATA["gate"]
    formula = [120, g["prediction"]["live_after_gate1"],
               g["prediction"]["live_after_gate2"]]
    planted = [120, g["planted"]["live_after_gate1"],
               g["planted"]["live_after_gate2"]]
    meas = g["measured_live"]
    x = range(3)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    w = 0.27
    ax.bar([i - w for i in x], formula, w, color=GRAY,
           label="formula expectation n(1−(1−s)^partners)")
    ax.bar(list(x), planted, w, color=GREEN,
           label="planted truth of the drawn corpus")
    ax.bar([i + w for i in x], meas, w, color=BLUE, label="measured")
    for i in x:
        ax.text(i - w, formula[i] + 2, f"{formula[i]:.0f}",
                ha="center")
        ax.text(i, planted[i] + 2, f"{planted[i]}", ha="center")
        ax.text(i + w, meas[i] + 2, f"{meas[i]}", ha="center")
    ax.axhline(120, color=GRAY, lw=0.8, ls="--")
    ax.text(0.55, 112, "no gates: 120 anchors at every stage",
            ha="left", fontsize=8, color="#666666")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["entering stage 1", "entering stage 2",
                        "entering stage 3"])
    ax.set_ylabel("live anchor documents (count)")
    ax.set_ylim(0, 158)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "join_gate_formula.png", dpi=150)


def orientation_plot():
    r = DATA["reshard"]
    labels = ["short doc anchored,\nlong doc streams\n(forced baseline)",
              "long doc anchored,\nshort doc streams\n(planner's pick)"]
    vals = [r["baseline"]["observed_selectivity"][0],
            r["free"]["observed_selectivity"][0]]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar(labels, vals, 0.5, color=[GREEN, BLUE])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.4f}", ha="center")
    planted = r["planted"]["stage1_selectivity"]
    ax.axhline(planted, color=GRAY, lw=0.8, ls="--")
    ax.text(-0.42, planted + 0.03, f"planted truth {planted}",
            ha="left", fontsize=8, color="#666666")
    ax.set_ylabel("stage-1 observed selectivity (fraction TRUE)")
    ax.set_ylim(0, 1.12)
    fig.tight_layout()
    fig.savefig(OUT / "join_orientation_accuracy.png", dpi=150)


def reshard_plot():
    r = DATA["reshard"]
    labels = ["shared anchor\n(forced, one group)",
              "planner's choice\n(two groups, barrier)"]
    tokens = [r["baseline"]["fresh_tokens"], r["free"]["fresh_tokens"]]
    walls = [r["baseline"]["wall_s"], r["free"]["wall_s"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, vals, unit, fmt in (
            (ax1, tokens, "fresh tokens computed", "{:,.0f}"),
            (ax2, walls, "wall seconds", "{:.1f}")):
        ax.bar(labels, vals, 0.55, color=[GREEN, BLUE])
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, fmt.format(v), ha="center")
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(vals) * 1.18)
    ax1.text(1, tokens[0] * 0.45, f"{r['token_ratio']:.1f}x fewer",
             ha="center", fontsize=11, color="#444444")
    ax2.text(1, walls[0] * 0.45, f"{r['wall_ratio']:.1f}x faster",
             ha="center", fontsize=11, color="#444444")
    fig.tight_layout()
    fig.savefig(OUT / "join_reshard_cost.png", dpi=150)


gate_plot()
reshard_plot()
orientation_plot()
print("wrote", OUT / "join_gate_formula.png")
print("wrote", OUT / "join_reshard_cost.png")
print("wrote", OUT / "join_orientation_accuracy.png")
