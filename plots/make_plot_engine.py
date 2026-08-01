"""Measured H100 makespans against the ideal model, from the isolated grid.

Left panel: measured makespan per configuration and policy, cold cache.
Warm-arm runs (no reset, corpus KV resident from an earlier query) appear
as short horizontal dashes over their policy's bar slot. Right panel:
measured over ideal for the cold runs, with a hairline at 1.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; YELLOW = "#eda100"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5,
    "ytick.labelsize": 9, "axes.titlesize": 10.5, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 9,
})
df = pd.read_csv("results/engine/grid_analysis.csv")
cold = df[df["mode"] == "waves"]
warm = df[df["mode"] == "warm"]


def series_key(r):
    if r.policy == "task":
        return "task"
    if r.k == 1:
        return "k1"
    if r.k == r.n:
        return "kfull"
    return "k2"


COL = {"task": ("Task-first", BLUE), "k1": ("Pipeline (k=1)", ORANGE),
       "k2": ("Lookahead 2", YELLOW), "kfull": ("Full speculation", AQUA)}
KEYS = ("task", "k1", "k2", "kfull")
configs = [(2, 0.25), (2, 0.5), (2, 0.8), (3, 0.7), (3, 0.9),
           (4, 0.8), (4, 0.95)]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), dpi=200)
fig.subplots_adjust(top=0.76, bottom=0.26, left=0.06, right=0.99, wspace=0.22)
for ax, metric, ylab, title in (
        (axes[0], "measured", "measured makespan (s)",
         "Measured on the H100, cold cache (2,000 docs)"),
        (axes[1], "ratio", "measured / ideal",
         "Distance from the ideal model")):
    xs = np.arange(len(configs)) * 1.5
    for kidx, key in enumerate(KEYS):
        name, col = COL[key]
        vals, pos = [], []
        for ci, (n, s1) in enumerate(configs):
            sub = cold[(cold.n == n) & (cold.s1 == s1)]
            sub = sub[[series_key(r) == key for _, r in sub.iterrows()]]
            if len(sub):
                vals.append(float(sub[metric].iloc[0]))
                pos.append(xs[ci] + kidx * 0.3)
        ax.bar(pos, vals, width=0.26, color=col)
        stag = (0.85 if metric == "measured" else 0.11) * (kidx == 1)
        for xp, v in zip(pos, vals):
            ax.text(xp, v + (0.35 if metric == "measured" else 0.04) + stag,
                    f"{v:.1f}", ha="center", fontsize=6.8, color=INK2)
        if metric == "measured":
            for ci, (n, s1) in enumerate(configs):
                subw = warm[(warm.n == n) & (warm.s1 == s1)]
                subw = subw[[series_key(r) == key for _, r in subw.iterrows()]]
                if len(subw):
                    v = float(subw["measured"].iloc[0])
                    xp = xs[ci] + kidx * 0.3
                    ax.plot([xp - 0.16, xp + 0.16], [v, v], color=INK,
                            lw=1.6, solid_capstyle="butt", zorder=5)
    ax.set_xticks(xs + 0.45)
    ax.set_xticklabels([f"n={n}\ns={s}" for n, s in configs])
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
axes[1].axhline(1.0, color=BASE, lw=1)
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k][1]) for k in KEYS]
labels = [COL[k][0] for k in KEYS]
handles.append(plt.Line2D([0], [0], color=INK, lw=1.6))
labels.append("warm cache (corpus KV resident)")
fig.legend(handles, labels, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, 0.005), fontsize=8.5, columnspacing=1.3,
           handlelength=1.4)
fig.suptitle("Real engine, real H100: measured against the ideal model",
             x=0.06, y=0.97, ha="left", fontsize=13, color=INK,
             fontweight="bold")
fig.text(0.06, 0.885, "vLLM 0.26, Qwen3-4B-FP8, planted flag outcomes, "
         "prefix cache reset before every run so each policy pays its own "
         "prefill.", fontsize=9.5, color=INK2)
fig.text(0.06, 0.845, "Dashes: the same run without the reset, documents "
         "already resident from an earlier query over the corpus.",
         fontsize=9.5, color=INK2)
fig.savefig("results/plots/engine_measured.png")
print("done")
