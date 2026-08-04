import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; YELLOW = "#eda100"
COL = {"task": ("Task-first", BLUE), "k=1": ("Pipeline (k=1)", ORANGE),
       "k=2": ("Lookahead 2", YELLOW), "kn": ("Full speculation", AQUA)}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "axes.titlesize": 10.5, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 9,
})
OUT = "results/plots/"


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def spread(vals, gap):
    order = np.argsort(vals)
    out = np.array(vals, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


# ---------------------------------------- 5. four filters, partial speculation
df = pd.read_csv("results/multi_filters.csv")
panels = [("Qwen3-4B-FP8", "H100-SXM-80GB", "Qwen3-4B on H100"),
          ("Qwen3-32B-FP8", "L40S-48GB", "Qwen3-32B on L40S")]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=200)
fig.subplots_adjust(top=0.76, bottom=0.13, left=0.07, right=0.88, wspace=0.42)
for ax, (m, dev, title) in zip(axes, panels):
    sub = df[(df.model == m) & (df.device == dev)]
    ends = []
    for pol, key in (("task", "task"), ("k=1", "k=1"), ("k=2", "k=2"),
                     ("k=4", "kn")):
        name, col = COL[key]
        g = sub[sub.policy == pol].sort_values("s")
        ax.plot(g.s, g.tau, color=col, lw=2, marker="o", ms=4.5,
                markerfacecolor=col, markeredgecolor=SURF, markeredgewidth=1)
        ends.append((name, col, float(g.tau.iloc[-1])))
    lo, hi = ax.get_ylim()
    ysp = spread([e[2] for e in ends], (hi - lo) * 0.06)
    for (name, col, v), yy in zip(ends, ysp):
        ax.annotate(name, xy=(0.955, yy), fontsize=8, color=col, va="center",
                    annotation_clip=False)
    ax.set_xlim(0.46, 0.99)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel("pass rate per stage", fontsize=9)
    ax.set_ylabel("makespan (s)", fontsize=9)
    style(ax)
fig.suptitle("Four filters: strict pipelining beats every speculation depth",
             x=0.07, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(0.07, 0.86, "Validated schedules, ideal cost model. Lookahead 2 tracks the pipeline within 1 to 8 percent;",
         fontsize=9.5, color=INK2)
fig.text(0.07, 0.815, "full speculation is flat and loses everywhere. Task-first pays document rereads at every stage.",
         fontsize=9.5, color=INK2)
fig.savefig(OUT + "nfilter_partial_spec.png")
plt.close(fig)

# ------------------------------------------------------- 6. filter-count sweep
df = pd.read_csv("results/n_sweep.csv")
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=200)
fig.subplots_adjust(top=0.76, bottom=0.13, left=0.07, right=0.90, wspace=0.28)
for ax, (m, dev, title) in zip(axes, panels):
    sub = df[(df.model == m) & (df.device == dev)]
    series = {}
    for _i, r in sub.iterrows():
        if r.policy == "task":
            series.setdefault("task", {})[r.n] = r.tau
        elif r.k == r.n:
            series.setdefault("kn", {})[r.n] = r.tau
        if r.k == 2 and r.n > 2:
            series.setdefault("k=2", {})[r.n] = r.tau
        if r.k == 1:
            series.setdefault("k=1", {})[r.n] = r.tau
    ends = []
    for key in ("task", "k=1", "k=2", "kn"):
        name, col = COL[key]
        pts = sorted(series[key].items())
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2,
                marker="o", ms=4.5, markerfacecolor=col,
                markeredgecolor=SURF, markeredgewidth=1)
        ends.append((name, col, pts[-1][1]))
    lo, hi = ax.get_ylim()
    ysp = spread([e[2] for e in ends], (hi - lo) * 0.06)
    for (name, col, v), yy in zip(ends, ysp):
        ax.annotate(name, xy=(6.15, yy), fontsize=8, color=col, va="center",
                    annotation_clip=False)
    ax.set_xlim(0.8, 6.9)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel("number of filters", fontsize=9)
    ax.set_ylabel("makespan (s)", fontsize=9)
    style(ax)
fig.suptitle("More filters widen the gap between the policies",
             x=0.07, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(0.07, 0.86, "Per-stage pass rate 0.8, validated schedules. Task-first grows fastest because it rereads",
         fontsize=9.5, color=INK2)
fig.text(0.07, 0.815, "documents at every stage; the pipeline adds only 50 prompt tokens per surviving stage.",
         fontsize=9.5, color=INK2)
fig.savefig(OUT + "n_sweep.png")
plt.close(fig)

# ----------------------------------------------------------- 7. multi-GPU scaling
df = pd.read_csv("results/multi_gpu.csv")
cases = [("Qwen3-4B-FP8", 2, "Qwen3-4B, two filters, pass rate 0.5"),
         ("Qwen3-32B-FP8", 4, "Qwen3-32B, four filters, pass rate 0.8")]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=200)
fig.subplots_adjust(top=0.76, bottom=0.13, left=0.07, right=0.91, wspace=0.28)
for ax, (m, n, title) in zip(axes, cases):
    sub = df[(df.model == m) & (df.n == n)]
    ends = []
    for pol, key in (("task", "task"), ("k=1", "k=1"), (f"k={n}", "kn")):
        name, col = COL[key]
        g = sub[sub.policy == pol].sort_values("gpus")
        ax.plot(g.gpus, g.makespan, color=col, lw=2, marker="o", ms=5,
                markerfacecolor=col, markeredgecolor=SURF, markeredgewidth=1)
        base = float(g[g.gpus == 1].makespan.iloc[0])
        ax.plot([1, 8], [base, base / 8], color=BASE, lw=0.9,
                ls=(0, (3, 3)), zorder=0)
        ends.append((name, col, float(g[g.gpus == 8].makespan.iloc[0])))
    ysp = 10 ** spread(np.log10([e[2] for e in ends]), 0.09)
    for (name, col, v), yy in zip(ends, ysp):
        ax.annotate(name, xy=(8.6, yy), fontsize=8, color=col, va="center",
                    annotation_clip=False)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    yt = [1, 2, 4, 8, 16] if m.startswith("Qwen3-4B") else [16, 32, 64, 128, 256]
    ax.set_yticks(yt)
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(0.9, 13)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel("H100 GPUs (data parallel)", fontsize=9)
    ax.set_ylabel("makespan (s)", fontsize=9)
    style(ax)
fig.suptitle("Scaling out H100s: near-perfect division of the work",
             x=0.07, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(0.07, 0.86, "Each GPU holds full weights and its own KV cache; documents split by balanced token counts.",
         fontsize=9.5, color=INK2)
fig.text(0.07, 0.815, "Dotted gray lines are ideal 1/G scaling; the schedules sit on them to within 0.1 percent.",
         fontsize=9.5, color=INK2)
fig.savefig(OUT + "multi_gpu.png")
plt.close(fig)
print("done")
