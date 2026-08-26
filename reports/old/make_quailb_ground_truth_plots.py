"""Plot Qwen3 32B ground-truth selectivity for QUAIL-B at sf=0.1.

Superseded: these selectivities predate PR #56, which removed a
duplicated ANSWER= cue from the templates. Kept so the old report's
figure can be rebuilt.

    uv run --with matplotlib python \
        reports/old/make_quailb_ground_truth_plots.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "plots" / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)
ARTIFACT_STEM = "20260825T082145Z-quailb-qwen32b-ground-truth-sf0.1"

plt.style.use(HERE.parent / "quail.mplstyle")
sys.path.insert(0, str(HERE.parent))
from plot_colors import BLUE, GREEN, ORANGE, TEAL

with open(ROOT / "results" / "benchmark"
          / f"{ARTIFACT_STEM}.json") as f:
    data = json.load(f)

WORKLOAD_COLORS = {
    "imdb": BLUE,
    "biodex": ORANGE,
    "fever": GREEN,
    "lepard": TEAL,
}

rows = sorted(data["label_sets"], key=lambda row: row["true_percent"])
labels = [f"{row['workload']} {row['legacy_code']}" for row in rows]
values = [row["true_percent"] for row in rows]
colors = [WORKLOAD_COLORS[row["workload"]] for row in rows]

fig, ax = plt.subplots(figsize=(8.0, 6.5))
y = range(len(rows))
ax.hlines(y, 0.04, values, color=colors, linewidth=2.0)
ax.scatter(values, y, color=colors, s=28, zorder=3)
for index, value in enumerate(values):
    label = f"{value:.3g}%" if value < 1 else f"{value:g}%"
    ax.annotate(label, (value, index), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=8.5)

ax.set_xscale("log")
ax.set_xlim(0.04, 170)
ax.set_yticks(list(y))
ax.set_yticklabels(labels)
ax.set_xlabel("TRUE labels (% of predicate rows, log scale)")
ax.set_ylabel("predicate")
fig.tight_layout()
path = OUT / f"{ARTIFACT_STEM}-selectivity.png"
fig.savefig(path, dpi=150)
print(f"wrote {path}")
