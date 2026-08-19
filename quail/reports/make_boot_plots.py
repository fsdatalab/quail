"""Boot-time breakdown plot (Quail vs stock vLLM).

Reads results/boot_profile.json (3-trial mean/median from
tests/gpu/boot_profile.py) and writes reports/plots/boot_profile.png.

    uv run --with matplotlib python reports/make_boot_plots.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "plots"
OUT.mkdir(exist_ok=True)

BLUE = "#2979FF"
LIGHT = "#90CAF9"
TEAL = "#00897B"
ORANGE = "#FB8C00"
GRAY = "#9E9E9E"
DARK = "#424242"


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


data = load("boot_profile.json")
q_cold = data["quail"]["cold"]
s_cold = data["stock"]["cold"]


def m(phase_key, side_cold):
    return side_cold[phase_key]["mean"]


# Stacked cold phases. Stock has no weight/KV split in this run
# (log markers were null), so it is one bar.
quail_phases = [
    ("load_model", m("load_model_s", q_cold), BLUE),
    ("arena", m("arena_s", q_cold), TEAL),
    ("pipeline", m("pipeline_s", q_cold), LIGHT),
    ("warm_kernels", m("warm_kernels_s", q_cold), ORANGE),
]
stock_total = m("boot_s", s_cold)

fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
y = [1, 0]  # Quail on top, stock below
height = 0.55

# Quail stacked
left = 0.0
for label, val, color in quail_phases:
    if val <= 0:
        continue
    ax.barh(y[0], val, left=left, height=height, color=color,
            edgecolor="white", linewidth=0.6)
    # label segments wide enough to read
    if val >= 3.0:
        ax.text(left + val / 2, y[0], f"{label}\n{val:.1f} s",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")
    left += val
q_total = m("boot_s", q_cold)
ax.text(q_total + 3, y[0],
        f"{q_total:.1f} s  (median {q_cold['boot_s']['median']:.1f})",
        va="center", fontsize=9.5, color=DARK)

# Stock single bar
ax.barh(y[1], stock_total, height=height, color=GRAY,
        edgecolor="white", linewidth=0.6)
ax.text(stock_total / 2, y[1],
        f"LLM(...)  {stock_total:.0f} s",
        ha="center", va="center", fontsize=9.5, color="white",
        fontweight="bold")
ax.text(stock_total + 3, y[1],
        f"{stock_total:.1f} s  (median {s_cold['boot_s']['median']:.1f})",
        va="center", fontsize=9.5, color=DARK)

ax.set_yticks(y)
ax.set_yticklabels(["Quail cold", "Stock vLLM cold"], fontsize=10)
ax.set_xlabel("seconds (mean of 3 H100 SXM trials; warm boot = 0 on both)",
              fontsize=9)
ax.set_xlim(0, max(stock_total, q_total) * 1.18)
ax.set_title("Cold boot breakdown: Quail vs stock vLLM",
             fontsize=12, fontweight="bold", loc="left")
ax.spines[["top", "right"]].set_visible(False)

# Tiny arena label outside the stack (too thin to print inside)
arena = m("arena_s", q_cold)
if 0 < arena < 3.0:
    load = m("load_model_s", q_cold)
    ax.annotate(f"arena {arena:.2f} s",
                xy=(load + arena / 2, y[0] + height / 2),
                xytext=(load + 8, y[0] + 0.55),
                fontsize=7.5, color=TEAL,
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.8))

legend = [
    Patch(facecolor=BLUE, label="load_model"),
    Patch(facecolor=TEAL, label="arena"),
    Patch(facecolor=ORANGE, label="warm_kernels"),
    Patch(facecolor=GRAY, label="stock LLM(...) total"),
]
ax.legend(handles=legend, loc="lower right", fontsize=8,
          framealpha=0.95, ncol=2)

note = ("Stock weight_load / kv_profile markers were null this run; "
        "bar is full LLM(...) wall (engine init + DeepGEMM warmup + "
        "CUDA graphs). Quail warm and stock warm = 0.0 s.")
fig.text(0.01, -0.02, note, fontsize=7.5, color=DARK, wrap=True)

fig.tight_layout()
out = OUT / "boot_profile.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
