"""Boot and first-query after skipping warm_kernels.

    uv run --with matplotlib python reports/make_boot_query_skip_plots.py
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
from plot_colors import BLUE, GRAY, GREEN, DARK, ORANGE


def load():
    with open(ROOT / "results" / "boot_query_after_skip.json") as f:
        return json.load(f)


data = load()
q_boot = data["quail_boot"]["boot_s"]
s_boot = data["stock_boot"]["boot_s"]
q_query = data["quail_query"]["wall_s"]
s_query = data["stock_query"]["wall_s_mean"]
q_warmed = data["quail_query"]["warmed_reference_s"]

fig, (ax_b, ax_q) = plt.subplots(
    1, 2, figsize=(10.2, 3.4),
    gridspec_kw={"width_ratios": [1, 1.15]},
)

# ---- boot ----------------------------------------------------------
bars = ax_b.bar(["Quail", "Stock vLLM"], [q_boot, s_boot],
                color=[GREEN, GRAY], width=0.5)
ax_b.text(bars[0].get_x() + bars[0].get_width() / 2, q_boot + 6,
          f"{q_boot:.1f} s", ha="center", va="bottom", fontsize=10,
          fontweight="bold", color=GREEN)
ax_b.text(bars[1].get_x() + bars[1].get_width() / 2, s_boot + 6,
          f"{s_boot:.1f} s", ha="center", va="bottom", fontsize=10,
          fontweight="bold", color=DARK)
ax_b.annotate(
    f"{s_boot / q_boot:.1f}x",
    xy=(0, q_boot), xytext=(0.55, (q_boot + s_boot) / 2),
    fontsize=9, color=DARK,
    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
ax_b.set_ylabel("seconds")
ax_b.set_ylim(0, s_boot * 1.18)
ax_b.tick_params(axis="y", length=0)

# ---- query ---------------------------------------------------------
xs = [0, 1, 2]
vals = [q_query, q_warmed, s_query]
colors = [BLUE, ORANGE, GRAY]
labels = ["Quail\nfirst query", "Quail\nwarmed ref.", "Stock vLLM\ncommitted"]
qb = ax_q.bar(xs, vals, color=colors, width=0.55)
for i, (bar, v) in enumerate(zip(qb, vals)):
    ax_q.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
              f"{v:.1f} s", ha="center", va="bottom", fontsize=10,
              fontweight="bold", color=DARK)
ax_q.set_xticks(xs)
ax_q.set_xticklabels(labels)
ax_q.set_ylabel("seconds")
ax_q.set_ylim(0, max(vals) * 1.22)
ax_q.tick_params(axis="y", length=0)
ax_q.annotate(
    f"+{q_query - q_warmed:.1f} s first-use cubin load",
    xy=(0, q_query), xytext=(0.35, q_query + 4),
    fontsize=8, color=DARK)

fig.savefig(OUT / "boot_query_after_skip.png", dpi=150)
print("wrote", OUT / "boot_query_after_skip.png")
