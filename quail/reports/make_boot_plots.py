"""Boot-time breakdown plot (Quail vs stock vLLM).

Reads results/boot_profile.json (3-trial mean/median from
tests/gpu/boot_profile.py) and writes reports/plots/boot_profile.png.

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
from plot_colors import QUAIL, STOCK, GOOD, TEAL, ORANGE, DARK


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


data = load("boot_profile.json")
q_cold = data["quail"]["cold"]
s_cold = data["stock"]["cold"]


def mean_s(side, key):
    return side[key]["mean"]


phases = [
    ("load_model", mean_s(q_cold, "load_model_s"), QUAIL),
    ("arena", mean_s(q_cold, "arena_s"), TEAL),
    ("warm_kernels", mean_s(q_cold, "warm_kernels_s"), ORANGE),
]
q_total = mean_s(q_cold, "boot_s")
s_total = mean_s(s_cold, "boot_s")

fig, (ax_q, ax_cmp) = plt.subplots(
    1, 2, figsize=(10.5, 3.5),
    gridspec_kw={"width_ratios": [1.4, 1]},
)

# ---- left: Quail cold boot phases -----------------------------------
left = 0.0
for name, val, color in phases:
    if val <= 0:
        continue
    ax_q.barh(0, val, left=left, height=0.48, color=color,
              edgecolor="white", linewidth=1.2)
    if val >= 3.0:
        ax_q.text(left + val / 2, 0, f"{val:.1f} s",
                  ha="center", va="center", fontsize=10.5,
                  color="white", fontweight="bold")
    left += val

ax_q.set_xlim(0, q_total * 1.08)
ax_q.set_ylim(-0.5, 0.5)
ax_q.set_yticks([])
ax_q.set_xlabel("seconds")
ax_q.set_title(
    f"Quail cold phases  ·  {q_total:.1f} s",
    fontsize=11, loc="left")
ax_q.spines["left"].set_visible(False)
ax_q.grid(False)

handles = [
    Patch(facecolor=c,
          label=(f"{n}  {v:.2f} s" if v < 1 else f"{n}  {v:.1f} s"))
    for n, v, c in phases
]
ax_q.legend(handles=handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8.5,
            handlelength=1.1)

# ---- right: total cold boot comparison ------------------------------
bars = ax_cmp.bar(
    ["Quail", "Stock vLLM"], [q_total, s_total],
    color=[GOOD, STOCK], width=0.5, edgecolor="white", linewidth=0.8,
)
for b, v, med in zip(bars, [q_total, s_total],
                      [q_cold["boot_s"]["median"],
                       s_cold["boot_s"]["median"]]):
    ax_cmp.text(
        b.get_x() + b.get_width() / 2, v + s_total * 0.02,
        f"{v:.1f} s", ha="center", va="bottom",
        fontsize=10.5, fontweight="bold", color=DARK,
    )
    ax_cmp.text(
        b.get_x() + b.get_width() / 2, v + s_total * 0.09,
        f"median {med:.1f}", ha="center", va="bottom",
        fontsize=8, color="#888888",
    )

speedup = s_total / q_total
ax_cmp.text(
    0, q_total + s_total * 0.18,
    f"{speedup:.0f}x faster", ha="center", fontsize=10.5,
    fontweight="bold", color=GOOD,
)

ax_cmp.set_ylim(0, s_total * 1.28)
ax_cmp.set_ylabel("seconds")
ax_cmp.set_title("Cold boot total", fontsize=11, loc="left")
ax_cmp.text(
    0.5, -0.18,
    "warm boot = 0.0 s on both  ·  3 H100 SXM trials",
    transform=ax_cmp.transAxes, ha="center", fontsize=8,
    color="#888888",
)

fig.subplots_adjust(bottom=0.24, wspace=0.35)
out = OUT / "boot_profile.png"
fig.savefig(out)
print("wrote", out)
