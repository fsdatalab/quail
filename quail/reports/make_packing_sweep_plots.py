"""Packing-sweep figures (issue #25): per-chunk GPU rate against
attended context, and measured vs predicted per query.

    uv run --with matplotlib python reports/make_packing_sweep_plots.py

Reads results/packing_sweep.json; writes two PNGs into reports/plots/.
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
from plot_colors import BLUE, DARK, GRAY, GREEN, RED

KIND_COLOR = {"filter": BLUE, "join": RED, "chain": GREEN}
KIND_MARK = {"filter": "o", "join": "^", "chain": "s"}

with open(ROOT / "results" / "packing_sweep.json") as f:
    data = json.load(f)

MODELS = [m for m in ("qwen3-4b-fp8", "qwen3-32b-fp8")
          if m in data["models"]]
SHORT = {"qwen3-4b-fp8": "Qwen3 4B fp8", "qwen3-32b-fp8": "Qwen3 32B fp8"}

# ------------------------------------------------- fig 1: rate vs context

fig, axes = plt.subplots(1, len(MODELS),
                         figsize=(4.6 * len(MODELS), 3.6))
axes = [axes] if len(MODELS) == 1 else list(axes)
for ax, model in zip(axes, MODELS):
    block = data["models"][model]
    fit = block["fits"]["filter"]
    a, a2c = fit["a_s_per_token"], fit["a2c_s_per_token2"]
    old = data["loaded_before"][model]
    xs_all = []
    for qid, q in sorted(block["queries"].items()):
        color = KIND_COLOR[q["kind"]]
        marker = KIND_MARK[q["kind"]]
        xs = [b["x_mean_ctx"] for b in q["chunk_bins"]]
        ys = [b["us_per_token"] for b in q["chunk_bins"]]
        xs_all += xs
        ax.plot(xs, ys, marker, color=color, markersize=4.5,
                linestyle="none", zorder=3)
    lo, hi = 0, max(xs_all) * 1.06
    grid = [lo + (hi - lo) * i / 60 for i in range(61)]
    # the causal fit in this x (mean attended context): t = a + 2*a2c*x
    ax.plot(grid, [(a + 2 * a2c * x) * 1e6 for x in grid],
            color=DARK, linewidth=1.2, zorder=2)
    ax.plot(grid, [(old["a_s_per_token"]
                    + 2 * old["a2_s_per_token2"] * x) * 1e6
                   for x in grid],
            color=GRAY, linewidth=1.2, linestyle="--", zorder=1)
    ax.set_title(SHORT[model])
    ax.set_xlabel("mean attended context per fresh token (tokens)")
    ax.set_ylabel("GPU time per fresh token (us)")
    # direct labels: the x axis counts cross context in full and a
    # segment's own context at half, so equal per-attended-token cost
    # would put join chunks ON the causal line
    y0, y1 = ax.get_ylim()
    ax.annotate("fit to the filter chunks", (hi * 0.60,
                (a + 2 * a2c * hi * 0.60) * 1e6),
                textcoords="offset points", xytext=(10, -16),
                fontsize=8, color=DARK,
                arrowprops=dict(arrowstyle="-", color=DARK, lw=0.6))
    y_old0 = (old["a_s_per_token"]
              + 2 * old["a2_s_per_token2"] * hi * 0.10) * 1e6
    ax.annotate("constants before this sweep", (hi * 0.10, y_old0),
                textcoords="offset points", xytext=(6, -18),
                ha="left", fontsize=8, color="#888888",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                lw=0.6))
    if "join" in block["fits"]:
        jf = block["fits"]["join"]
        slope_ratio = jf["a2x_s_per_token2"] / (2 * a2c)
        suffix_us = jf["per_suffix_s"] * 1e6
        jb = [b for q in block["queries"].values()
              if q["kind"] == "join" for b in q["chunk_bins"]]
        jb.sort(key=lambda b: b["x_mean_ctx"])
        mid = jb[len(jb) // 2]
        ax.annotate(
            f"join chunks: reading kept anchor KV\ncosts "
            f"{slope_ratio:.2f}x per attended token,\n"
            f"plus ~{suffix_us:.0f} us per suffix",
            xy=(mid["x_mean_ctx"], mid["us_per_token"]),
            xycoords="data", textcoords="axes fraction",
            xytext=(0.05, 0.66), ha="left", va="center",
            fontsize=8, color=RED,
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.6,
                            relpos=(1.0, 0.3)))
    ax.plot([], [], "o", color=BLUE, label="filter chunks")
    ax.plot([], [], "^", color=RED, label="join chunks")
    if any(q["kind"] == "chain" for q in block["queries"].values()):
        ax.plot([], [], "s", color=GREEN, label="chain chunks")
    ax.legend(loc="upper left")
fig.savefig(OUT / "packing_sweep_context.png", dpi=150)
plt.close(fig)

# --------------------------------------- fig 2: measured vs predicted

ORDER = ("IMDB-1", "BIO-1", "BIO-F3", "FEV-2", "IMDB-2", "LEP-2",
         "BIO-2")
n_q = max(len(data["models"][m]["queries"]) for m in MODELS)
fig, axes = plt.subplots(1, len(MODELS),
                         figsize=((1.1 + 0.82 * n_q) * len(MODELS),
                                  3.4))
axes = [axes] if len(MODELS) == 1 else list(axes)
for ax, model in zip(axes, MODELS):
    block = data["models"][model]
    order = [q for q in ORDER if q in block["queries"]]
    for i, qid in enumerate(order):
        q = block["queries"][qid]
        color = KIND_COLOR[q["kind"]]
        ax.bar(i, q["us_per_token_gpu"], width=0.55, color=color)
        ax.text(i, q["us_per_token_gpu"] / 2,
                f"{q['us_per_token_gpu']:.1f}", ha="center",
                va="center", color="white", fontsize=9)
        # the prediction from the constants loaded before this sweep
        ax.plot([i - 0.34, i + 0.34],
                [q["predicted_us_old"]] * 2, color=DARK,
                linewidth=1.4, zorder=3)
        delta = (q["us_per_token_gpu"] / q["predicted_us_old"] - 1) * 100
        ax.text(i, max(q["us_per_token_gpu"], q["predicted_us_old"])
                * 1.02, f"{delta:+.0f}%", ha="center", fontsize=8.5,
                color=DARK)
        # container repeats, when this query ran on several
        reps = block["containers"].get(qid, {}).get("by_container")
        if reps:
            ax.plot([i] * len(reps), list(reps.values()), "o",
                    color=DARK, markersize=3.5, zorder=4)
    ax.set_title(SHORT[model])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{q}\n{block['queries'][q]['kind']}"
                        for q in order])
    ax.set_ylabel("GPU time per fresh token (us)")
    ax.set_ylim(0, None)
    if model == MODELS[0]:
        ax.text(0.02, 0.98, "dash: predicted from the constants\n"
                "before this sweep\ndots: other containers",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color=DARK)
fig.savefig(OUT / "packing_sweep_queries.png", dpi=150)
plt.close(fig)
print(f"wrote {OUT}/packing_sweep_context.png and "
      f"{OUT}/packing_sweep_queries.png")
