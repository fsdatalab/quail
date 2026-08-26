"""vllm-opbench baseline plots.

Reads the numbers embedded below (sourced from the tee'd Modal run
logs and quail's own committed benchmark logs - see
2026-08-25-vllm-opbench-baseline.md for exact source lines) and
writes two PNGs to reports/plots/.

    uv run --with matplotlib python reports/make_vllm_opbench_plots.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, GRAY, RED, DARK, ORANGE

# ---------------------------------------------------------------- data
# quail wall_s: (point_estimate, other_observed_or_None) - point
# estimate is the "full run" log in both cases; the second value is
# an independent rerun of the identical query, kept as a range
# whisker where the two disagree (join queries only; filters agreed
# closely and quail_range collapses to a point).
QUERIES = ["filter-reports", "join-reports", "join-claims", "join-imdb"]
LABELS = {"filter-reports": "filter-reports\n(F7, 200 reports)",
          "join-reports": "join-reports\n(REACTION, 122.8k pairs)",
          "join-claims": "join-claims\n(SUPPORT, 5.7k pairs)",
          "join-imdb": "join-imdb\n(DISCUSS_ASPECT, 60k pairs)"}

quail = {
    "filter-reports": {"4b": (9.03, 10.14), "32b": (55.72, 53.8)},
    # join-reports/join-imdb: fresh rerun on current code (2026-08-25),
    # checked against ground truth - see the report's Stability
    # section. Replaces two disagreeing Aug-20 reference numbers. 32B
    # warm passes never ran (Modal client heartbeat bug right after
    # both cold passes finished) - 32B values here are cold-only.
    "join-reports": {"4b": (138.6, 140.7), "32b": (812.2, None)},
    "join-claims": {"4b": (4.86, None), "32b": (13.46, None)},
    "join-imdb": {"4b": (52.4, 55.0), "32b": (358.1, None)},
}

# vllm-opbench generate-only wall (GPU serving time, no CPU build,
# no Modal RPC overhead) - the number comparable to quail's own
# wall_s in spirit, since quail's wall_s is also GPU-serving time.
vllm_opbench = {
    "filter-reports": {"4b": 10.75, "32b": 74.05},
    "join-reports": {"4b": 217.69, "32b": 468.11},   # clean, post-OOM-fix reruns
    "join-claims": {"4b": 22.87, "32b": 175.61},
    "join-imdb": {"4b": 33.43, "32b": 250.68},
}

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
model_x = np.arange(2)
width = 0.32

for ax, q in zip(axes, QUERIES):
    q_vals = [quail[q]["4b"][0], quail[q]["32b"][0]]
    q_range = [quail[q]["4b"][1], quail[q]["32b"][1]]
    v_vals = [vllm_opbench[q]["4b"], vllm_opbench[q]["32b"]]

    b1 = ax.bar(model_x - width / 2, q_vals, width, color=BLUE,
               label="quail")
    b2 = ax.bar(model_x + width / 2, v_vals, width, color=RED,
               label="vllm-opbench")

    for i, (v, r) in enumerate(zip(q_vals, q_range)):
        if r is not None and abs(r - v) > 0.01:
            lo, hi = sorted([v, r])
            ax.plot([model_x[i] - width / 2] * 2, [lo, hi],
                   color=DARK, linewidth=1.2)
            ax.plot([model_x[i] - width / 2 - 0.06,
                     model_x[i] - width / 2 + 0.06], [hi, hi],
                   color=DARK, linewidth=1.2)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                   f"{h:.0f}", ha="center", va="bottom", fontsize=8,
                   color=DARK)

    ax.set_xticks(model_x)
    ax.set_xticklabels(["4B", "32B"])
    ax.set_title(LABELS[q], fontsize=9.5)
    ax.set_ylabel("wall time (s)" if q == QUERIES[0] else "")

axes[0].legend(loc="upper left", fontsize=8.5)
fig.suptitle(
    "quail vs vllm-opbench, generate-only wall time (sf=0.1, H100)\n"
    "join-reports/join-imdb 4B whiskers: cold vs. warm pass of a fresh "
    "rerun, now stable. Both queries at 32B: fresh cold-only rerun "
    "(warm pass never ran - Modal client heartbeat bug), checked "
    "against ground truth instead - see report.",
    fontsize=9, y=1.08)
fig.savefig(OUT / "vllm_opbench_vs_quail.png")
plt.close(fig)

# ------------------------------------------- join-imdb, 3-way, both models
fig2, ax2 = plt.subplots(figsize=(6, 3.8))
systems = ["quail", "stock vLLM\n(quail's checkpoint,\nsync client)",
          "vllm-opbench\n(base checkpoint,\nload-time fp8)"]
colors = [BLUE, ORANGE, RED]
data_4b = [quail["join-imdb"]["4b"][0], 34.59, vllm_opbench["join-imdb"]["4b"]]
data_32b = [quail["join-imdb"]["32b"][0], 206.74,
           vllm_opbench["join-imdb"]["32b"]]

x = np.arange(3)
w = 0.32
b1 = ax2.bar(x - w / 2, data_4b, w, color=colors, alpha=1.0)
b2 = ax2.bar(x + w / 2, data_32b, w, color=colors, alpha=0.55,
            hatch="//")
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 6, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8.5, color=DARK)
ax2.set_xticks(x)
ax2.set_xticklabels(systems, fontsize=8.5)
ax2.set_ylabel("wall time (s)")
ax2.set_title(
    "join-imdb (DISCUSS_ASPECT, 5000x12=60k pairs)\nsolid = 4B, "
    "hatched = 32B", fontsize=10)
fig2.savefig(OUT / "vllm_opbench_join_imdb_3way.png")
plt.close(fig2)

print(f"wrote {OUT / 'vllm_opbench_vs_quail.png'}")
print(f"wrote {OUT / 'vllm_opbench_join_imdb_3way.png'}")
