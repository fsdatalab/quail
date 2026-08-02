"""The DocEngine story in one figure, all at 10,000 documents, cold.

Panel a: measured makespans, five systems from the naive vLLM baselines
to the strict in-engine scheduler, with the calibrated model prediction
as a hairline per configuration. Panel b: the mechanism, corpus read
multipliers. Panel c: contention at n=4 s=0.8 under the heavy
co-tenant. Panel d: the reasoning-filter analytical map (phase A).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; YELLOW = "#eda100"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.titlesize": 10.5, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 8,
})

CFGS = ["n=2\ns=0.5", "n=4\ns=0.8", "n=4\ns=0.95"]
SYSTEMS = [
    ("vLLM naive, task prompts", MUTED, [61.7, 122.8, 158.3]),
    ("vLLM naive, doc-first", ORANGE, [64.4, 111.3, 153.7]),
    ("blocked batches", YELLOW, [53.8, 74.0, 84.2]),
    ("client library", AQUA, [48.9, 52.5, 57.9]),
    ("in-engine, strict", BLUE, [48.5, 52.3, 54.7]),
]
PRED = [47.8, 54.7, 60.3]
MULT = [
    ([1.58, 3.03, 3.93]), ([1.62, 2.74, 3.82]), ([1.18, 1.27, 1.32]),
    ([1.17, 1.23, 1.29]), ([1.18, 1.23, 1.30]),
]

fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), dpi=200)
fig.subplots_adjust(top=0.87, bottom=0.07, left=0.06, right=0.985,
                    hspace=0.42, wspace=0.2)
(ax_a, ax_b), (ax_c, ax_d) = axes

xs = np.arange(3) * 1.9
for k, (name, col, vals) in enumerate(SYSTEMS):
    pos = xs + k * 0.3
    ax_a.bar(pos, vals, width=0.26, color=col)
    for xp, v in zip(pos, vals):
        ax_a.text(xp, v + 2, f"{v:.0f}", ha="center", fontsize=6.6,
                  color=INK2)
    ax_b.bar(pos, MULT[k], width=0.26, color=col)
for ci in range(3):
    ax_a.plot([xs[ci] - 0.2, xs[ci] + 1.4], [PRED[ci]] * 2, color=INK,
              lw=1.2, ls=(0, (4, 2)), zorder=5)
ax_a.set_xticks(xs + 0.6)
ax_a.set_xticklabels(CFGS)
ax_a.set_ylabel("makespan (s), 10k docs, cold")
ax_a.set_title("a. From naive engine use to the plan-governed engine",
               loc="left", color=INK)
ax_b.axhline(1.0, color=BASE, lw=1)
ax_b.set_xticks(xs + 0.6)
ax_b.set_xticklabels(CFGS)
ax_b.set_ylabel("corpus read multiplier")
ax_b.set_title("b. The mechanism: how many times the corpus is read",
               loc="left", color=INK)

ct_labels = ["stock\nalone", "stock +\nneighbor", "equal-rank\n+ neighbor",
             "plan-ranked\nalone", "plan-ranked\n+ neighbor"]
ct_vals = [53.5, 3600, 231.2, 52.2, 52.0]
ct_cols = [MUTED, MUTED, YELLOW, BLUE, BLUE]
pos = np.arange(5) * 0.75
bars = ax_c.bar(pos, ct_vals, width=0.55, color=ct_cols)
ax_c.set_yscale("log")
ax_c.set_ylim(30, 6000)
for xp, v, lab in zip(pos, ct_vals,
                      ["53.5", "never\nfinished", "231", "52.2", "52.0"]):
    ax_c.text(xp, v * 1.15, lab, ha="center", fontsize=7.2, color=INK2)
ax_c.set_xticks(pos)
ax_c.set_xticklabels(ct_labels, fontsize=7.5)
ax_c.set_ylabel("makespan (s), log scale")
ax_c.set_title("c. Contention: a heavy co-tenant on the same card "
               "(n=4, s=0.8)", loc="left", color=INK)

think = [0, 32, 128, 512]
series_d = [("task-first", MUTED, [127.7, 139.6, 175.0, 316.8]),
            ("pipeline", BLUE, [48.7, 60.6, 96.0, 237.8]),
            ("full speculation", AQUA, [52.2, 68.2, 116.2, 308.3])]
for name, col, vals in series_d:
    ax_d.plot(think, vals, color=col, lw=2, marker="o", ms=4, label=name)
ax_d.plot([0], [52.3], marker="D", ms=7, color=INK, ls="none",
          label="measured (strict engine)")
ax_d.set_xlabel("thinking tokens per filter call")
ax_d.set_ylabel("predicted makespan (s), calibrated")
ax_d.set_title("d. Phase A map: reasoning filters (n=4, s=0.8)",
               loc="left", color=INK)
ax_d.legend(loc="upper left")

for ax in (ax_a, ax_b, ax_c, ax_d):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _n, c, _v in SYSTEMS]
labels = [n for n, _c, _v in SYSTEMS]
handles.append(plt.Line2D([0], [0], color=INK, lw=1.2, ls=(0, (4, 2))))
labels.append("calibrated model prediction")
fig.legend(handles, labels, loc="upper left", ncol=3,
           bbox_to_anchor=(0.06, 0.955), fontsize=8.5)
fig.suptitle("DocEngine on one H100: the same query, five ways",
             x=0.06, y=0.985, ha="left", fontsize=14, color=INK,
             fontweight="bold")
fig.savefig("results/plots/docengine_story.png")
print("done")
