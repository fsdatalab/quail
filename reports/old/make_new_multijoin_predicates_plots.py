"""Plot Qwen3 32B ground-truth selectivity for the 4 predicates added to
support the new multi-join queries (IMDB-8/9/11, BIO-6, FEV-7/C/D).

    uv run --with matplotlib python reports/old/make_new_multijoin_predicates_plots.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "plots"
OUT.mkdir(parents=True, exist_ok=True)
ARTIFACT_STEM = "20260826T040743Z-new-multijoin-predicates-ground-truth"

plt.style.use(HERE.parent / "quail.mplstyle")
sys.path.insert(0, str(HERE.parent))
from plot_colors import BLUE, GREEN, ORANGE

WORKLOAD_COLORS = {"imdb": BLUE, "biodex": ORANGE, "fever": GREEN}

with open(ROOT / "results" / "benchmark" / f"{ARTIFACT_STEM}.json") as f:
    data = json.load(f)

rows = sorted(data["label_sets"], key=lambda row: row["true_percent"])
labels = [f"{row['workload']} {row['legacy_code']}" for row in rows]
values = [max(row["true_percent"], 0.02) for row in rows]
colors = [WORKLOAD_COLORS[row["workload"]] for row in rows]

fig, ax = plt.subplots(figsize=(7.0, 3.2))
y = range(len(rows))
ax.hlines(y, 0.02, values, color=colors, linewidth=2.0)
ax.scatter(values, y, color=colors, s=32, zorder=3)
for index, row in enumerate(rows):
    label = "0%" if row["true_percent"] == 0 else f"{row['true_percent']:.3g}%"
    ax.annotate(label, (values[index], index), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=8.5)

ax.set_xscale("log")
ax.set_xlim(0.02, 20)
ax.set_yticks(list(y))
ax.set_yticklabels(labels)
ax.set_xlabel("TRUE labels (% of predicate rows, log scale)")
ax.set_ylabel("new predicate")
fig.tight_layout()
path = OUT / f"{ARTIFACT_STEM}-selectivity.png"
fig.savefig(path, dpi=150)
print(f"wrote {path}")
