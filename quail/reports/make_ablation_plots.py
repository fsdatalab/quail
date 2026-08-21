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
from plot_colors import BLUE, GRAY, GREEN, RED, DARK


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


abl = load("ablation_forward.json")
stock = abl["stock"]
packed = abl["packed"]


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


LIGHT_GRAY = "#B0B0B0"

rungs = [
    ("A0  stock vLLM, fp8 KV",      stock["A0"]["runs"],       LIGHT_GRAY),
    ("A1  stock vLLM, bf16 KV",      stock["A1"]["runs"],       LIGHT_GRAY),
    ("A2  our executor, vLLM kernels", packed["runs"]["A2"],    BLUE),
    ("A3  our executor, our kernels",  packed["runs"]["A3"],    GREEN),
]
labels = [r[0] for r in rungs]
walls = [mean(r[1], "wall") for r in rungs]
colors = [r[2] for r in rungs]

fig, ax = plt.subplots(figsize=(9, 3.6))
y = list(range(len(rungs)))
ax.barh(y, walls, height=0.55, color=colors, edgecolor="white")

for i, w in enumerate(walls):
    rows = rungs[i][1]
    tok = mean(rows, "tok_s") if "tok_s" in rows[0] else None
    txt = f"{w:.1f} s"
    if tok:
        txt += f"   {tok / 1000:.0f}k tok/s"
    ax.text(w + 0.3, i, txt, va="center", fontsize=9.5, color=DARK)

deltas = [
    (0, "bf16 KV"),
    (1, "replace engine"),
    (2, "fused kernels"),
]
for i, label in deltas:
    d = walls[i + 1] - walls[i]
    sign = "+" if d > 0 else ""
    color = RED if d > 0 else GREEN
    ax.annotate(f"{label}: {sign}{d:.1f} s",
                xy=((walls[i] + walls[i + 1]) / 2, i + 0.5),
                fontsize=8.5, color=color, ha="center", va="center")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlim(0, 52)
ax.set_xlabel("seconds (10k documents, 5 filters, one H100)", fontsize=9)
ax.set_title("Forward-pass ablation", fontsize=12, fontweight="bold",
             loc="left")

fig.savefig(OUT / "ablation_ladder.png")
print("wrote", OUT / "ablation_ladder.png")
