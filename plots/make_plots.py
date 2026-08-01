import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; YELLOW = "#eda100"
POL = {"task": ("Task-first", BLUE), "pipe": ("Pipeline", ORANGE),
       "fullspec": ("Full speculation", AQUA)}

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
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def spread(vals, gap):
    """Nudge label y-positions apart by at least `gap`."""
    order = np.argsort(vals)
    out = np.array(vals, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


# ---------------------------------------------------- 1. latency vs pass rate
n = pd.read_csv("results/n10k_two_stage.csv")
piv = n.pivot_table(index=["model", "device", "policy"], columns="s1",
                    values="tau")
configs = [("Qwen3-4B-FP8", "H100-SXM-80GB", "Qwen3-4B on H100"),
           ("Qwen3-4B-FP8", "L40S-48GB", "Qwen3-4B on L40S"),
           ("Qwen3-32B-FP8", "H100-SXM-80GB", "Qwen3-32B on H100"),
           ("Qwen3-32B-FP8", "L40S-48GB", "Qwen3-32B on L40S")]
fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), dpi=200)
fig.subplots_adjust(hspace=0.42, wspace=0.24, top=0.86, bottom=0.09,
                    left=0.07, right=0.93)
S = [0.10, 0.25, 0.50, 0.75, 0.90]
for ax, (m, dev, title) in zip(axes.flat, configs):
    ends = {}
    for pol in ("task", "pipe", "fullspec"):
        name, col = POL[pol]
        ys = [piv.loc[(m, dev, pol), s] for s in S]
        ax.plot(S, ys, color=col, lw=2, marker="o", ms=4.5,
                markerfacecolor=col, markeredgecolor=SURF, markeredgewidth=1)
        ends[name] = (ys[-1], col)
    lo, hi = ax.get_ylim()
    ys0 = [v for v, _ in ends.values()]
    ysp = spread(ys0, (hi - lo) * 0.075)
    for (name, (v, col)), yy in zip(ends.items(), ysp):
        ax.annotate(name, xy=(0.90, v), xytext=(0.925, yy), fontsize=8.5,
                    color=col, va="center")
    ax.axvline(0.203, color=BASE, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlim(0.05, 1.22)
    ax.set_title(title, loc="left", color=INK)
    ax.set_ylabel("makespan (s)", fontsize=9)
    ax.set_xticks(S)
    style(ax)
axes.flat[0].annotate("break-even 0.20", xy=(0.203, axes.flat[0].get_ylim()[1]),
                      xytext=(0.23, axes.flat[0].get_ylim()[1] * 0.99),
                      fontsize=8, color=MUTED, va="top")
axes.flat[2].set_xlabel("pass rate of filter 1", fontsize=9)
axes.flat[3].set_xlabel("pass rate of filter 1", fontsize=9)
fig.suptitle("Finish time of the 10,000-document query, by policy",
             x=0.07, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(0.07, 0.895, "Ideal cost model, validated schedules. Task-first wins below "
         "a 0.20 pass rate, pipeline above it; full speculation never wins at this scale.",
         fontsize=9.5, color=INK2)
fig.savefig(OUT + "latency_vs_passrate.png")
plt.close(fig)

# ------------------------------------------------- 2. layers agreement (gaps)
lp = pd.read_csv("results/lp_two_stage.csv")
lp["gap_replay"] = 100 * (lp.t_construct - lp.LB) / lp.LB
cfgs = [("Qwen3-4B-FP8", "H100-SXM-80GB", "4B\nH100"),
        ("Qwen3-4B-FP8", "L40S-48GB", "4B\nL40S"),
        ("Qwen3-32B-FP8", "H100-SXM-80GB", "32B\nH100"),
        ("Qwen3-32B-FP8", "L40S-48GB", "32B\nL40S")]
fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=200)
fig.subplots_adjust(top=0.70, bottom=0.16, left=0.07, right=0.98)
xt, xl = [], []
x = 0
for m, dev, lab in cfgs:
    for s1 in (0.1, 0.5, 0.9):
        sub = lp[(lp.model == m) & (lp.device == dev) & (lp.s1 == s1)]
        for k, pol in enumerate(("task", "pipe", "fullspec")):
            v = float(sub[sub.method == pol].gap_replay.iloc[0])
            ax.bar(x + k * 0.28, v, width=0.24, color=POL[pol][1])
        xt.append(x + 0.28)
        xl.append(f"{s1:.1f}\n{lab.splitlines()[0]} {lab.splitlines()[1]}")
        x += 1.3
ax.set_xticks(xt)
ax.set_xticklabels([f"{l.splitlines()[0]}" for l in xl], fontsize=8)
for i, (m, dev, lab) in enumerate(cfgs):
    ax.text(i * 3.9 + 1.58, -0.22, lab.replace("\n", " on "), ha="center",
            fontsize=9, color=INK2)
ax.set_ylabel("percent above the lower bound", fontsize=9)
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
style(ax)
handles = [plt.Rectangle((0, 0), 1, 1, color=POL[p][1]) for p in POL]
ax.legend(handles, [POL[p][0] for p in POL], loc="lower left", ncol=3,
          bbox_to_anchor=(0, 1.01))
fig.suptitle("How close every replayed schedule sits to its certified floor",
             x=0.07, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(0.07, 0.86, "Replayed finite latency vs the resource lower bound, all 36 cells. "
         "Groups: pass rate of filter 1. Worst cell is 1.04%.",
         fontsize=9.5, color=INK2)
fig.savefig(OUT + "layers_agreement.png")
plt.close(fig)

# ----------------------------------------------------------- 3. small N exact
Ns = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
builder = {"task": [2.623, 5.983, 10.224, 17.270, 18.571, 31.673, 40.637,
                    50.694, 78.845, 111.015],
           "pipe": [2.802, 5.051, 9.656, 11.660, 16.574, 23.446, 36.322,
                    46.398, 69.470, 94.649],
           "fullspec": [2.093, 4.701, 9.688, 11.825, 17.687, 24.718, 39.296,
                        50.117, 75.215, 100.013]}
exactN = [2, 3, 4]
exact = {"task": [2.441, 3.922, 5.983], "pipe": [2.169, 3.116, 4.515],
         "fullspec": [2.093, 3.302, 4.701]}
fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=200)
fig.subplots_adjust(top=0.78, bottom=0.12, left=0.08, right=0.97)
endv = np.log10([builder[p][-1] for p in ("task", "pipe", "fullspec")])
endy = 10 ** spread(endv, 0.055)
for pol, ey in zip(("task", "pipe", "fullspec"), endy):
    name, col = POL[pol]
    ax.plot(Ns, builder[pol], color=col, lw=2, ls=(0, (4, 2.5)), alpha=0.85)
    ax.plot(exactN, exact[pol], color=col, lw=0, marker="o", ms=8,
            markerfacecolor=SURF, markeredgecolor=col, markeredgewidth=2)
    ax.annotate(name, xy=(66, ey), fontsize=9, color=col, va="center")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(Ns); ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_yticks([2, 5, 10, 20, 50, 100])
ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_xlim(1.8, 105)
ax.set_xlabel("documents in the query", fontsize=9)
ax.set_ylabel("makespan (ms)", fontsize=9)
style(ax)
ax.annotate("exact optimum\n(rings, N ≤ 4)", xy=(3, 3.116),
            xytext=(2.05, 7.4), fontsize=9, color=INK2,
            arrowprops=dict(arrowstyle="-", color=BASE, lw=0.8))
ax.annotate("speculation wins\nonly here", xy=(2, 2.093), xytext=(2.6, 1.55),
            fontsize=9, color=INK2,
            arrowprops=dict(arrowstyle="-", color=BASE, lw=0.8))
h = [plt.Line2D([], [], color=MUTED, lw=2, ls=(0, (4, 2.5))),
     plt.Line2D([], [], color=MUTED, lw=0, marker="o", ms=8,
                markerfacecolor=SURF, markeredgecolor=MUTED, markeredgewidth=2)]
ax.legend(h, ["schedule builder", "exact optimizer"], loc="upper left")
fig.suptitle("Small queries, real Qwen3-4B on H100 numbers",
             x=0.08, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(0.08, 0.895, "Dashed lines: the schedule builders. Rings: the exact optimizer, which holds documents",
         fontsize=9.5, color=INK2)
fig.text(0.08, 0.86, "back to hide filter waits and beats the builders by 12 to 24 percent below 8 documents.",
         fontsize=9.5, color=INK2)
fig.savefig(OUT + "smallN_exact.png")
plt.close(fig)

# -------------------------------------------------------- 4. lookahead, n = 4
rates = ["0.50", "0.80", "0.95"]
series = [("Task-first", BLUE, [473.8, 753.3, 946.8]),
          ("Lookahead 1 (pipeline)", ORANGE, [334.8, 381.7, 414.4]),
          ("Lookahead 2", YELLOW, [361.6, 395.7, 418.4]),
          ("Lookahead 4 (full spec.)", AQUA, [426.5, 426.5, 426.5])]
fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=200)
fig.subplots_adjust(top=0.72, bottom=0.10, left=0.08, right=0.98)
xs = np.arange(len(rates)) * 1.5
for k, (name, col, vals) in enumerate(series):
    pos = xs + k * 0.30
    ax.bar(pos, vals, width=0.26, color=col)
    for xp, v in zip(pos, vals):
        ax.text(xp, v + 12, f"{v:.0f}", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(xs + 0.45)
ax.set_xticklabels([f"pass rate {r} per stage" for r in rates], fontsize=9.5)
ax.set_ylabel("makespan (s)", fontsize=9)
ax.set_ylim(0, 1060)
style(ax)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _n, c, _v in series]
ax.legend(handles, [nm for nm, _c, _v in series], loc="lower left", ncol=4,
          bbox_to_anchor=(0, 1.01), fontsize=8.5)
fig.suptitle("Four filters on Qwen3-32B / L40S: pipeline still wins",
             x=0.08, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(0.08, 0.895, "Lookahead k runs k filter prompts per document without waiting.",
         fontsize=9.5, color=INK2)
fig.text(0.08, 0.855, "Speculation narrows to 1 percent of pipeline at 0.95 but never crosses under the ideal model.",
         fontsize=9.5, color=INK2)
fig.savefig(OUT + "lookahead_n4.png")
plt.close(fig)
print("done")
