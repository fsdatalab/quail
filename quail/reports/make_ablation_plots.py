"""The forward-pass ablation ladder figure (A0-A3).

Reads results/ablation_forward.json (banked 2026-08-19) and writes
reports/plots/ablation_ladder.png. Run from the quail/ directory:

    uv run --with matplotlib python reports/make_ablation_plots.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "plots"
OUT.mkdir(exist_ok=True)

ACCENT = "#2979FF"
GRAY = "#9E9E9E"
GREEN = "#43A047"
DARK = "#424242"


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


abl = load("ablation_forward.json")
stock = abl["stock"]
packed = abl["packed"]


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


# rung order top -> bottom: A0, A1, A2, A3
rungs = [
    ("A0", "stock vLLM, fp8 KV\n(pipelined client)", stock["A0"]["runs"], GRAY),
    ("A1", "stock vLLM, bf16 KV\n(pipelined client)", stock["A1"]["runs"], GRAY),
    ("A2", "packed executor, vLLM kernels\nkept KV, 110k chunks, pinned staging",
     packed["runs"]["A2"], ACCENT),
    ("A3", "+ our three Triton kernels\n(the current executor)",
     packed["runs"]["A3"], GREEN),
]
walls = [mean(r, "wall") for _, _, r, _ in rungs]

fig, (ax, bx) = plt.subplots(
    2, 1, figsize=(10.2, 6.6), dpi=150,
    gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.62,
                 "top": 0.90, "bottom": 0.09})

# ---------------- panel A: the ladder --------------------------------
y = list(range(len(rungs)))
labels = [f"{name}  {desc}" for name, desc, _, _ in rungs]
cols = [c for _, _, _, c in rungs]
bars = ax.barh(y, walls, height=0.58, color=cols, edgecolor="white")
for i, (w, (_, _, rows, _)) in enumerate(zip(walls, rungs)):
    tok_s = mean(rows, "tok_s") if "tok_s" in rows[0] else None
    note = f"{w:.1f} s"
    if tok_s:
        note += f"  ({tok_s / 1000:.0f}k tok/s)"
    ax.text(w + 0.35, i, note, va="center", fontsize=10,
            color=DARK, fontweight="bold")

# per-step deltas, between adjacent bars
steps = [
    (0, "bf16 KV instead of fp8"),
    (1, "engine removed; kept KV our way;\nstill vLLM's kernels"),
    (2, "our three Triton kernels"),
]
for i, text in steps:
    d = walls[i + 1] - walls[i]
    sign = "−" if d < 0 else "+"
    color = GREEN if d < 0 else "#C62828"
    ax.text(46.6, i + 0.5, f"{text}: {sign}{abs(d):.1f} s",
            va="center", fontsize=8.6, color=color)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 58)
ax.set_xlabel("wall time, seconds (10,000-document five-filter workload, "
              "mean of 2 reps)", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("The forward-pass ablation ladder: stock vLLM to the packed "
             "executor", fontsize=12.5, loc="left", fontweight="bold",
             pad=12)

# ---------------- panel B: the packed pair is GPU-bound ---------------
# GPU-busy time against the wall-minus-GPU gap for A2 and A3: with
# pinned staging in the base configuration, both rungs sit on the GPU
# time, so the kernels are the whole difference between them.
names = ["A2", "A3"]
gpu = [mean(packed["runs"][n], "gpu_s") for n in names]
gap = [mean(packed["runs"][n], "wall") - g for n, g in zip(names, gpu)]

x = list(range(len(names)))
w = 0.32
b1 = bx.bar([i - w / 2 for i in x], gpu, width=w, color=ACCENT,
            label="GPU busy")
b2 = bx.bar([i + w / 2 for i in x], gap, width=w, color="#C62828",
            label="wall − GPU (unhidden CPU)")
for bars in (b1, b2):
    for b in bars:
        bx.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 0.5, f"{b.get_height():.1f}",
                ha="center", fontsize=8.6, color=DARK)
bx.set_xticks(x)
bx.set_xticklabels(names, fontsize=10)
bx.set_ylabel("seconds", fontsize=9.5)
bx.set_ylim(0, 50)
bx.legend(fontsize=8.6, ncol=2, frameon=False, loc="upper right")
bx.spines[["top", "right"]].set_visible(False)
bx.set_title("Both packed rungs sit on the GPU time - the kernels are "
             "the whole difference", fontsize=11, loc="left", pad=8)

fig.text(0.01, 0.005,
         "All rungs: committed 10k five-filter workload, one H100, "
         "Qwen3 4B fp8 weights. Packed rungs share one container boot; "
         "each stock rung has its own container. Data: "
         "results/ablation_forward.json.",
         fontsize=7.5, color=DARK)
fig.savefig(OUT / "ablation_ladder.png", bbox_inches="tight")
print("wrote", OUT / "ablation_ladder.png")
