"""Ablation ladder figure (A0-A3).

    uv run --with matplotlib python reports/make_ablation_plots.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
RESULTS = ROOT / "results"
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import QUAIL, STOCK, GOOD, BAD, DARK


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


abl = load("ablation_forward.json")
stock = abl["stock"]
packed = abl["packed"]


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


rungs = [
    ("A0  stock, fp8 KV",              stock["A0"]["runs"],     STOCK),
    ("A1  stock, bf16 KV",             stock["A1"]["runs"],     STOCK),
    ("A2  Quail executor, vLLM kernels", packed["runs"]["A2"],  QUAIL),
    ("A3  Quail executor, our kernels",  packed["runs"]["A3"],  GOOD),
]
labels = [r[0] for r in rungs]
walls = [mean(r[1], "wall") for r in rungs]
colors = [r[2] for r in rungs]

fig, ax = plt.subplots(figsize=(9, 3.8))
y = list(range(len(rungs)))
ax.barh(y, walls, height=0.52, color=colors, edgecolor="white", linewidth=0.8)

for i, w in enumerate(walls):
    rows = rungs[i][1]
    tok = mean(rows, "tok_s") if "tok_s" in rows[0] else None
    txt = f" {w:.1f} s"
    if tok:
        txt += f"  ·  {tok / 1000:.0f}k tok/s"
    ax.text(w + 0.2, i, txt, va="center", fontsize=9.5, color=DARK,
            fontweight="bold")

deltas = [
    (0, "bf16 KV"),
    (1, "replace engine"),
    (2, "fused kernels"),
]
for i, label in deltas:
    d = walls[i + 1] - walls[i]
    sign = "+" if d > 0 else ""
    color = BAD if d > 0 else GOOD
    mid_x = max(walls[i], walls[i + 1]) + 0.2
    ax.annotate(
        f"  {label}  {sign}{d:.1f} s",
        xy=(mid_x, i + 0.5),
        fontsize=8, color=color, ha="left", va="center",
        fontweight="medium",
    )

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlim(0, 56)
ax.set_xlabel("seconds  (10k docs, 5 filters, one H100)")
ax.set_title("Forward-pass ablation", loc="left")
ax.grid(axis="x", alpha=0.3)

fig.savefig(OUT / "ablation_ladder.png")
print("wrote", OUT / "ablation_ladder.png")
