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
from plot_colors import BLUE, GRAY, GREEN, TEAL, ORANGE, DARK


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


data = load("boot_profile.json")
q_cold = data["quail"]["cold"]
s_cold = data["stock"]["cold"]


def mean_s(side, key):
    return side[key]["mean"]


phases = [
    ("load_model", mean_s(q_cold, "load_model_s"), BLUE),
    ("arena", mean_s(q_cold, "arena_s"), TEAL),
    ("warm_kernels", mean_s(q_cold, "warm_kernels_s"), ORANGE),
]
q_total = mean_s(q_cold, "boot_s")
s_total = mean_s(s_cold, "boot_s")

fig, (ax_q, ax_cmp) = plt.subplots(
    1, 2, figsize=(10.5, 3.2),
    gridspec_kw={"width_ratios": [1.4, 1]},
)

# ---- left: Quail cold boot phases -----------------------------------
left = 0.0
for name, val, color in phases:
    if val <= 0:
        continue
    ax_q.barh(0, val, left=left, height=0.4, color=color)
    if val >= 3.0:
        ax_q.text(left + val / 2, 0, f"{val:.1f} s",
                  ha="center", va="center", fontsize=10.5,
                  color="white", fontweight="bold")
    left += val

ax_q.set_xlim(0, q_total * 1.08)
ax_q.set_ylim(-0.45, 0.45)
ax_q.set_yticks([])
ax_q.set_xticks([])
ax_q.set_title(
    f"Quail cold phases  ·  {q_total:.1f} s total",
    fontsize=11, loc="left")

handles = [
    Patch(facecolor=c,
          label=(f"{n}  {v:.2f} s" if v < 1 else f"{n}  {v:.1f} s"))
    for n, v, c in phases
]
ax_q.legend(handles=handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8.5,
            handlelength=1.1)

# ---- right: total cold boot comparison ------------------------------
speedup = s_total / q_total
bars = ax_cmp.bar(
    ["Quail", "Stock vLLM"], [q_total, s_total],
    color=[GREEN, GRAY], width=0.45,
)

ax_cmp.text(
    bars[0].get_x() + bars[0].get_width() / 2,
    q_total + s_total * 0.03,
    f"{q_total:.1f} s  ({speedup:.0f}x faster)",
    ha="center", va="bottom", fontsize=10, fontweight="bold",
    color=GREEN,
)
ax_cmp.text(
    bars[1].get_x() + bars[1].get_width() / 2,
    s_total + s_total * 0.03,
    f"{s_total:.1f} s",
    ha="center", va="bottom", fontsize=10, fontweight="bold",
    color=DARK,
)

ax_cmp.set_ylim(0, s_total * 1.18)
ax_cmp.set_yticks([])
ax_cmp.set_title("Cold boot total", fontsize=11, loc="left")
ax_cmp.text(
    0.5, -0.15,
    "warm boot = 0.0 s on both  ·  3 H100 SXM trials",
    transform=ax_cmp.transAxes, ha="center", fontsize=8,
    color="#999999",
)

fig.subplots_adjust(bottom=0.22, wspace=0.3)
out = OUT / "boot_profile.png"
fig.savefig(out)
print("wrote", out)
