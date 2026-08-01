import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"; BLUE = "#2a78d6"; ORANGE = "#eb6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "axes.titlesize": 10.5, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 9,
})
df = pd.read_csv("results/additive_traffic.csv")
df = df[(df.sweep == "B") & (df.s == 0.9)]
panels = [("Qwen3-4B-FP8", "H100-SXM-80GB", "Qwen3-4B on H100"),
          ("Qwen3-32B-FP8", "L40S-48GB", "Qwen3-32B on L40S")]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=200)
fig.subplots_adjust(top=0.74, bottom=0.14, left=0.08, right=0.97, wspace=0.26)
for ax, (m, dev, title) in zip(axes, panels):
    sub = df[(df.model == m) & (df.device == dev)]
    adv = {}
    for rule, col, name in (("tau_max", BLUE, "traffic overlapped (paper's rule)"),
                            ("tau_add", ORANGE, "traffic charged as time")):
        xs, ys = [], []
        for scale in (1, 2, 4, 10):
            g = sub[sub.scale == scale]
            pipe = float(g[g.policy == "pipe"][rule].iloc[0])
            spec = float(g[g.policy == "fullspec"][rule].iloc[0])
            xs.append(scale); ys.append(100 * (pipe - spec) / spec)
        ax.plot(xs, ys, color=col, lw=2, marker="o", ms=5,
                markerfacecolor=col, markeredgecolor=SURF, markeredgewidth=1)
        adv[name] = (col, ys[-1])
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 10])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(0.9, 11.5)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel("document length multiplier (mean 297 to 2,966 tokens)",
                  fontsize=9)
    ax.set_ylabel("speculation advantage over pipeline (%)", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y"); ax.grid(axis="x", visible=False); ax.tick_params(length=0)
h = [plt.Line2D([], [], color=BLUE, lw=2, marker="o", ms=5),
     plt.Line2D([], [], color=ORANGE, lw=2, marker="o", ms=5)]
axes[0].legend(h, ["traffic overlapped (paper's rule)",
                   "traffic charged as time"], loc="lower right", fontsize=8.5)
axes[0].annotate("speculation wins", xy=(1.05, 0.12), fontsize=8.5, color=INK2)
axes[0].annotate("pipeline wins", xy=(1.05, -0.4), fontsize=8.5, color=INK2)
fig.suptitle("Charging KV traffic flips the winner for long documents",
             x=0.08, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(0.08, 0.87, "Two filters, pass rate 0.9. Positive means full speculation beats the pipeline. Under the paper's",
         fontsize=9.5, color=INK2)
fig.text(0.08, 0.825, "overlap rule speculation never wins; charged as time, it wins on the 4B/H100 past 3x length.",
         fontsize=9.5, color=INK2)
fig.savefig("results/plots/additive_traffic.png")
print("done")
