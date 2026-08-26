"""Parallel judge-pass plots.

Reads results/parallel_judge_pass_sf0.1.json and writes
reports/plots/judge_pass_selectivity.png and
reports/plots/judge_pass_wall.png.

    uv run --with matplotlib python reports/make_parallel_judge_pass_plots.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, GRAY, LIGHT_GRAY, ORANGE, DARK

DATA = json.loads(
    (ROOT / "results" / "parallel_judge_pass_sf0.1.json").read_text())

WORKLOAD_NAME = {"imdb": "IMDB", "biodex": "BioDEX",
                 "fever": "FEVER", "lepard": "LePaRD"}


def selectivity_plot():
    """Positive rate per predicate, before and after the trailing
    ANSWER= cue was removed. Dots rather than bars: the axis is
    logarithmic, where bar length no longer encodes the value."""
    rows = DATA["selectivity_before_and_after_stripping_the_cue"]
    # LePaRD's citation join is labelled from the dataset's passage ids,
    # so no prompt reaches the model and it cannot move
    rows = [r for r in rows if r["legacy_code"] != "LEPJOIN"]
    rows.sort(key=lambda r: r["true_percent_after"])

    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    for i, r in enumerate(rows):
        before, after = r["true_percent_before"], r["true_percent_after"]
        if after != before:
            ax.plot([before, after], [i, i], color=LIGHT_GRAY, lw=2.4,
                    zorder=1, solid_capstyle="round")
            ax.plot([before], [i], "o", color=GRAY, ms=6, zorder=2)
        ax.plot([after], [i], "o", color=BLUE, ms=6, zorder=3)

        label = f"{after:.2f}"
        if before and after / before >= 2:
            label += f"   {after / before:.0f}x"
        ax.text(after * 1.25, i, label, va="center", fontsize=8.5,
                color=ORANGE if before and after / before >= 2 else DARK)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{WORKLOAD_NAME[r['workload']]}  {r['legacy_code']}" for r in rows],
        fontsize=8.5)
    ax.set_ylim(-1.4, len(rows) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(0.045, 400)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.set_xlabel("rows labelled TRUE, percent of rows judged "
                  "(log scale, spans three orders of magnitude)")
    ax.plot([0.09], [-1.0], "o", color=GRAY, ms=6)
    ax.text(0.105, -1.0, "before", va="center", fontsize=8.5, color="#888888")
    ax.plot([0.4], [-1.0], "o", color=BLUE, ms=6)
    ax.text(0.47, -1.0, "after", va="center", fontsize=8.5, color=DARK)
    fig.savefig(OUT / "judge_pass_selectivity.png", dpi=150)
    plt.close(fig)


def wall_plot():
    """Where the 33.8 minutes went, and why four containers bought less
    than four times the speed."""
    w = DATA["workloads"]
    order = sorted(w, key=lambda k: w[k]["total_wall_s"])
    base_load = min(v["boot_s"] for v in w.values()) / 60

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for i, k in enumerate(order):
        load = w[k]["boot_s"] / 60
        judging = w[k]["wall_minus_boot_s"] / 60
        ax.barh(i, base_load, 0.55, color=GRAY)
        ax.barh(i, load - base_load, 0.55, left=base_load, color=ORANGE)
        ax.barh(i, judging, 0.55, left=load, color=BLUE)
        ax.text(load + judging + 0.5, i, f"{load + judging:.1f} min",
                va="center", fontsize=8.5, color=DARK)

    top = len(order) - 1
    slowest_load = w[order[top]]["boot_s"] / 60
    # "contention" sits a line higher: its segment is too narrow for
    # the label to clear "model load" on the same line
    for x, dy, align, text, color in (
            (0, 0.5, "left", "model load", "#777777"),
            (slowest_load, 1.0, "right", "contention", ORANGE),
            (slowest_load + w[order[top]]["wall_minus_boot_s"] / 120,
             0.5, "center", "judging", BLUE)):
        ax.text(x, top + dy, text, ha=align, fontsize=8, color=color)

    parallel = DATA["result"]["wall_s_parallel"] / 60
    serial = DATA["result"]["wall_s_serial_reconstructed"] / 60
    extra = w[order[top]]["boot_s"] / 60 - base_load
    ax.axvline(parallel, color=DARK, lw=0.8, ls=":", ymax=0.82)
    ax.text(0, -1.35,
            f"All four finish at {parallel:.1f} min. One container doing the "
            f"same work would take {serial:.1f} min, so the split is worth "
            f"{DATA['result']['speedup_over_serial']:.2f}x.\n"
            f"BioDEX sets the wall time: it is "
            f"{100 * w['biodex']['wall_minus_boot_s'] / sum(v['wall_minus_boot_s'] for v in w.values()):.0f}%"
            f" of the judging, and {extra:.1f} min of its load is four "
            f"containers pulling one checkpoint at once.",
            fontsize=8.5, color="#555555", va="top")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([WORKLOAD_NAME[k] for k in order], fontsize=9)
    ax.set_ylim(-2.6, len(order) + 0.5)
    ax.set_xlim(0, 46)
    ax.set_xlabel("minutes")
    fig.savefig(OUT / "judge_pass_wall.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    selectivity_plot()
    wall_plot()
    print("wrote", OUT / "judge_pass_selectivity.png")
    print("wrote", OUT / "judge_pass_wall.png")
