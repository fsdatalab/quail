"""Single-stage fast-path figure (issue #6, on the unified filter
path): wall time of the arena path against the fast path on the
10,000-document one-question workload.

    uv run --with matplotlib python reports/make_single_stage_fast_path_plots.py

Reads results/m1_filter1.json; writes one PNG into reports/plots/.
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
from plot_colors import BLUE, DARK, GRAY

with open(RESULTS / "m1_filter1.json") as f:
    data = json.load(f)

MODES = (("arena", "arena path\n(paged unified)", GRAY),
         ("no_arena", "fast path\n(one causal segment)", BLUE))

fig, ax = plt.subplots(figsize=(5.6, 3.4))
for i, (mode, label, color) in enumerate(MODES):
    walls = [r["wall"] for r in data["runs"] if r["mode"] == mode]
    best = min(walls)
    ax.bar(i, best, width=0.55, color=color)
    # individual repetitions as dots so the spread is visible
    ax.plot([i] * len(walls), walls, "o", color=DARK, markersize=4,
            zorder=3)
    tok_s = max(r["tok_s"] for r in data["runs"] if r["mode"] == mode)
    ax.text(i, best / 2, f"{best:.1f} s\n{tok_s / 1e3:.0f}k tok/s",
            ha="center", va="center", color="white", fontsize=10)

saved = data["saved_s"]
best_arena = data["best_wall"]["arena"]
pct = 100 * saved / best_arena
ax.annotate(
    f"{saved:.1f} s saved ({pct:.1f}%)\n{data['flips']} answer flips",
    xy=(1, data["best_wall"]["no_arena"]),
    xytext=(0.5, best_arena * 1.02), ha="center", va="bottom",
    fontsize=10, color=DARK)

ax.set_xticks(range(len(MODES)))
ax.set_xticklabels([label for _, label, _ in MODES])
ax.set_ylabel("wall time (s), best of "
              f"{max(r['rep'] for r in data['runs']) + 1} repetitions")
ax.set_ylim(0, best_arena * 1.22)
fig.tight_layout()
fig.savefig(OUT / "single_stage_fast_path.png", dpi=150)
print(f"wrote {OUT / 'single_stage_fast_path.png'}")
