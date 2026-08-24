"""Tiered-boot plots.

Reads results/boot_tiered.json (tests/gpu/boot_profile.py), the old
results/boot_profile.json (the swept-warmup boot this change
replaces), results/m1_filter1.json and results/baseline_filter1.json
(the query references), and writes reports/plots/boot_tiered.png.

    uv run --with matplotlib python reports/make_boot_plots.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
RESULTS = ROOT / "results"
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, GRAY, GREEN, ORANGE, RED, TEAL, DARK

# boot phases from the profiler-off control: py-spy adds ~4.6 s to
# the warm phase
nospy = json.load(open(RESULTS / "boot_tiered_nospy.json"))
stock_q = json.load(open(RESULTS / "baseline_filter1.json"))

new_cold = nospy["touch"]["cold"]


def mean_s(agg, key):
    return agg[key]["mean"]


PHASES = [("load_model", "load_model_s", BLUE),
          ("arena", "arena_s", TEAL),
          ("warm_kernels", "warm_kernels_s", ORANGE)]

fig, (ax_boot, ax_q) = plt.subplots(
    1, 2, figsize=(10.5, 3.4),
    gridspec_kw={"width_ratios": [1.5, 1]})

# ---- left: what a cold container boot spends its time on ------------
left = 0.0
for name, key, color in PHASES:
    val = mean_s(new_cold, key)
    if not val:
        continue
    ax_boot.barh(0, val, left=left, height=0.42, color=color)
    if val >= 2.5:
        ax_boot.text(left + val / 2, 0, f"{val:.1f}",
                     ha="center", va="center", fontsize=9.5,
                     color="white", fontweight="bold")
    left += val
total = mean_s(new_cold, "boot_s")
ax_boot.text(total + 0.6, 0, f"{total:.1f} s", va="center",
             fontsize=10, color=DARK, fontweight="bold")

ax_boot.set_xlim(0, total * 1.22)
ax_boot.set_ylim(-0.45, 0.75)
ax_boot.set_yticks([])
ax_boot.set_xlabel("cold container boot (seconds, trial mean)")
ax_boot.legend(
    handles=[Patch(facecolor=c, label=n) for n, _, c in PHASES],
    loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
    fontsize=8.5, handlelength=1.1)

# ---- right: first query after boot vs stock vLLM --------------------
best = nospy["query"]["best_wall"]["no_arena"]
stock_wall = min(r["wall"] for r in stock_q["runs"])
bars = ax_q.bar(["Quail", "stock vLLM"], [best, stock_wall],
                color=[GREEN, GRAY], width=0.45)
for bar, v in zip(bars, [best, stock_wall]):
    ax_q.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
              f"{v:.1f}", ha="center", fontsize=10,
              fontweight="bold", color=DARK)
ax_q.text(bars[1].get_x() + bars[1].get_width() / 2,
          stock_wall / 2, f"+{(stock_wall / best - 1) * 100:.0f}%",
          ha="center", fontsize=9.5, color="white",
          fontweight="bold")
ax_q.set_ylim(0, stock_wall * 1.18)
ax_q.set_ylabel("filter query wall (seconds)")
ax_q.set_title("10k-document single-stage filter", fontsize=10,
               loc="left")

fig.tight_layout()
fig.savefig(OUT / "boot_tiered.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT / 'boot_tiered.png'}")


# ---- figure 2: inside the touch pass (py-spy timeline) --------------
tl = json.load(open(RESULTS / "boot_touch_timeline.json"))

CAT_COLORS = {
    "waiting for the GPU (warm chunk running)": BLUE,
    "gemm + quant kernel launches": ORANGE,
    "attention + triton kernel launches": GREEN,
    "deepgemm cache reads (disk)": RED,
    "packing + admission (cpu)": GRAY,
    "other": DARK,
}

for model, entry in tl["models"].items():
    timeline = entry["timeline"]
    if timeline is None:
        continue
    cats = timeline["categories"]
    bin_s = timeline["bin_s"]
    n = timeline["n_bins"]
    xs = [i * bin_s for i in range(n)]
    fig2, ax = plt.subplots(figsize=(10.5, 3.0))
    bottom = [0.0] * n
    for c in cats:
        vals = [v / bin_s for v in timeline["matrix"][c]]
        if not any(vals):
            continue
        ax.bar(xs, vals, width=bin_s, bottom=bottom, align="edge",
               color=CAT_COLORS[c], label=c, linewidth=0)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xlim(0, n * bin_s)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("share of each 0.1 s bin")
    note = ("recorded under py-spy, which slows it: this phase is "
            "3.6 s without the profiler" if model == "qwen3-4b-fp8"
            else "recorded under py-spy; 11-22 s without it, "
                 "host-speed dependent")
    ax.set_xlabel(
        f"seconds into the touch pass, {model} ({note})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncol=3, fontsize=8, handlelength=1.1)
    fig2.tight_layout()
    suffix = "" if model == "qwen3-4b-fp8" else "_32b"
    fig2.savefig(OUT / f"boot_touch_timeline{suffix}.png", dpi=150,
                 bbox_inches="tight")
    print(f"wrote {OUT / f'boot_touch_timeline{suffix}.png'}")
