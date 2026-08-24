"""Cold-cache principled-warmup boot phases vs the committed
warm-cache profile.

    uv run --with matplotlib python reports/make_boot_cold_plots.py
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
from plot_colors import BLUE, ORANGE, DARK, GRAY

with open(ROOT / "results" / "boot_profile_cold.json") as f:
    cold = json.load(f)
with open(ROOT / "results" / "boot_profile.json") as f:
    warm = json.load(f)

c = cold["quail"]["cold"]
w = warm["quail"]["cold"]
rows = [
    ("this run, cold cache",
     [("load", c["load_model_s"], BLUE),
      ("warmup", c["warm_kernels_s"], ORANGE)]),
    ("committed, warm cache",
     [("load", w["load_model_s"]["mean"], BLUE),
      ("warmup", w["warm_kernels_s"]["mean"], ORANGE)]),
]

fig, ax = plt.subplots(figsize=(6.4, 2.6))
for i, (label, parts) in enumerate(rows):
    left = 0.0
    for name, val, color in parts:
        ax.barh(i, val, left=left, height=0.45, color=color)
        if val >= 8:
            ax.text(left + val / 2, i, f"{name} {val:.1f} s",
                    ha="center", va="center", fontsize=9,
                    color="white")
        elif val >= 1:
            ax.text(left + val + 3, i, f"{name} {val:.1f} s",
                    ha="left", va="center", fontsize=9, color=DARK)
        left += val
    if i == 0:
        ax.text(left + 4, i, f"total {left:.1f} s",
                ha="left", va="center", fontsize=9, color=DARK)

ax.set_yticks([0, 1])
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("seconds")
ax.set_xlim(0, 280)
ax.axvline(c["warm_kernels_s"], color=GRAY, linewidth=0.6)
fig.savefig(OUT / "boot_profile_cold.png", dpi=150)
print("wrote", OUT / "boot_profile_cold.png")
