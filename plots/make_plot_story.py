"""The DocEngine story in one figure, all at 10,000 documents, cold.

Panel a: measured makespans, five systems from the naive vLLM baselines
to the strict in-engine scheduler, with the calibrated model prediction
as a hairline per configuration. Panel b: the mechanism, corpus read
multipliers. Panel c: contention at n=4 s=0.8 under the heavy
co-tenant. Panel d: the reasoning-filter analytical map (phase A).

Every plotted number is read from a banked results file, except the
two log-only contention arms declared in PROSE_ONLY below. The map:

  panel a bars      results/engine/scale10k.json.gz (task waves, naive
                    block-k1 waves, blocked block-k1 manifest),
                    client10k.json.gz (client), strict10k.json.gz
  panel a hairline  ideal column of client10k_analysis.csv times
                    KERNEL_FACTOR (the one calibration constant)
  panel b bars      read multiplier recomputed from each row's wave
                    token counters in the same three .gz files
  panel c bars      pinned10k.json.gz (stock alone), pinned10k_v3
                    .json.gz (plan-ranked alone and under the
                    neighbor), PROSE_ONLY for the two log-only arms
  panel d curves    results/reasoning_sweep.csv (n=4, s=0.8 rows)
  panel d diamond   strict10k.json.gz (n=4, s=0.8)
"""

import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENG = ROOT / "results" / "engine"

# STALE PENDING RE-BASELINE. These numbers exist only in prose and run
# logs (notes/RESULTS.md), with no banked results file. Like every
# number this figure reads, they were measured on the old slim image;
# the re-baseline flight (notes/PROPOSAL.md) replaces them.
PROSE_ONLY = {
    # Stock engine under the heavy co-tenant: never finished inside
    # the one-hour harness limit; plotted at the 3,600-second cutoff.
    "stock_with_neighbor_cutoff_s": 3600,
    # The intermediate pinned iteration (equal-rank requests): memory
    # defended, compute lost. Its file (pinned10k_hard.json.gz) was
    # never banked.
    "equal_rank_with_neighbor_s": 231.2,
}

# The calibration constant: the 275,000 tokens-per-second ideal-model
# ceiling over the 80,000 measured operating rate. The 80,000 anchor
# is stale (xengine.json: the same vLLM reads 97,220 on a CUDA 13
# devel image); recompute after the re-baseline flight.
KERNEL_FACTOR = 275_000 / 80_000

CONFIGS = [(2, 0.5), (4, 0.8), (4, 0.95)]


def load_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)["results"]


def pick(rows, n, s, policy, k, mode, junk=None):
    for r in rows:
        s0 = r["s"][0] if isinstance(r.get("s"), list) else r.get("s")
        if (r["n"] == n and abs(s0 - s) < 1e-9 and r["policy"] == policy
                and r["k"] == k and r["mode"] == mode
                and (junk is None or r.get("junk") == junk)):
            return r
    raise KeyError((n, s, policy, k, mode, junk))


def read_mult(row):
    corpus = sum(row["d_tok"])
    computed = sum(w["prompt_tokens"] - w["cached_tokens"]
                   for w in row["waves"])
    return computed / corpus


scale = load_rows(ENG / "scale10k.json.gz")
client = load_rows(ENG / "client10k.json.gz")
strict = load_rows(ENG / "strict10k.json.gz")
pinned1 = load_rows(ENG / "pinned10k.json.gz")
pinned3 = load_rows(ENG / "pinned10k_v3.json.gz")

selectors = [
    ("vLLM naive, task prompts", scale, dict(policy="task", k=0,
                                             mode="waves")),
    ("vLLM naive, doc-first", scale, dict(policy="block", k=1,
                                          mode="waves")),
    ("blocked batches", scale, dict(policy="block", k=1,
                                    mode="manifest")),
    ("client library", client, dict(policy="block", k=1, mode="client")),
    ("in-engine, strict", strict, dict(policy="block", k=1,
                                       mode="strict")),
]

with open(ENG / "client10k_analysis.csv") as f:
    ideal = {(int(r["n"]), float(r["s1"])): float(r["ideal"])
             for r in csv.DictReader(f)
             if r["policy"] == "block" and r["k"] == "1"}

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

COLORS = [MUTED, ORANGE, YELLOW, AQUA, BLUE]
SYSTEMS = []
MULT = []
for (name, rows, sel), col in zip(selectors, COLORS):
    picked = [pick(rows, n, s, **sel) for n, s in CONFIGS]
    SYSTEMS.append((name, col, [r["makespan"] for r in picked]))
    MULT.append([read_mult(r) for r in picked])
PRED = [ideal[cfg] * KERNEL_FACTOR for cfg in CONFIGS]

CFGS = ["n=2\ns=0.5", "n=4\ns=0.8", "n=4\ns=0.95"]

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
stock_alone = pick(pinned1, 4, 0.8, "block", 1, "stock", junk=False)
plan_alone = pick(pinned3, 4, 0.8, "block", 1, "pinned", junk=False)
plan_junk = pick(pinned3, 4, 0.8, "block", 1, "pinned", junk=True)
ct_vals = [stock_alone["makespan"],
           PROSE_ONLY["stock_with_neighbor_cutoff_s"],
           PROSE_ONLY["equal_rank_with_neighbor_s"],
           plan_alone["makespan"], plan_junk["makespan"]]
ct_text = [f"{ct_vals[0]:.1f}", "never\nfinished", f"{ct_vals[2]:.0f}",
           f"{ct_vals[3]:.1f}", f"{ct_vals[4]:.1f}"]
ct_cols = [MUTED, MUTED, YELLOW, BLUE, BLUE]
pos = np.arange(5) * 0.75
bars = ax_c.bar(pos, ct_vals, width=0.55, color=ct_cols)
ax_c.set_yscale("log")
ax_c.set_ylim(30, 6000)
for xp, v, lab in zip(pos, ct_vals, ct_text):
    ax_c.text(xp, v * 1.15, lab, ha="center", fontsize=7.2, color=INK2)
ax_c.set_xticks(pos)
ax_c.set_xticklabels(ct_labels, fontsize=7.5)
ax_c.set_ylabel("makespan (s), log scale")
ax_c.set_title("c. Contention: a heavy co-tenant on the same card "
               "(n=4, s=0.8)", loc="left", color=INK)

with open(ROOT / "results" / "reasoning_sweep.csv") as f:
    sweep = [r for r in csv.DictReader(f)
             if r["n"] == "4" and r["s"] == "0.8"]
sweep.sort(key=lambda r: int(r["think"]))
think = [int(r["think"]) for r in sweep]
series_d = [("task-first", MUTED, [float(r["task"]) for r in sweep]),
            ("pipeline", BLUE, [float(r["pipeline"]) for r in sweep]),
            ("full speculation", AQUA,
             [float(r["fullspec"]) for r in sweep])]
for name, col, vals in series_d:
    ax_d.plot(think, vals, color=col, lw=2, marker="o", ms=4, label=name)
strict_meas = pick(strict, 4, 0.8, "block", 1, "strict")["makespan"]
ax_d.plot([0], [strict_meas], marker="D", ms=7, color=INK, ls="none",
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
out = ROOT / "results" / "plots" / "docengine_story.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"wrote {out}")
