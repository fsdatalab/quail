"""Ablation ladder figure (A0-A3).

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

BLUE = "#2979FF"
GRAY = "#B0B0B0"
GREEN = "#43A047"
RED = "#C62828"
DARK = "#333333"


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


abl = load("ablation_forward.json")
stock = abl["stock"]
packed = abl["packed"]


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


rungs = [
    ("A0  stock vLLM, fp8 KV",      stock["A0"]["runs"],       GRAY),
    ("A1  stock vLLM, bf16 KV",      stock["A1"]["runs"],       GRAY),
    ("A2  our executor, vLLM kernels", packed["runs"]["A2"],    BLUE),
    ("A3  our executor, our kernels",  packed["runs"]["A3"],    GREEN),
]
labels = [r[0] for r in rungs]
walls = [mean(r[1], "wall") for r in rungs]
colors = [r[2] for r in rungs]

fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
y = list(range(len(rungs)))
ax.barh(y, walls, height=0.55, color=colors, edgecolor="white")

for i, w in enumerate(walls):
    rows = rungs[i][1]
    tok = mean(rows, "tok_s") if "tok_s" in rows[0] else None
    txt = f"{w:.1f} s"
    if tok:
        txt += f"   {tok / 1000:.0f}k tok/s"
    ax.text(w + 0.3, i, txt, va="center", fontsize=9.5, color=DARK)

deltas = [
    (0, "bf16 KV"),
    (1, "replace engine"),
    (2, "fused kernels"),
]
for i, label in deltas:
    d = walls[i + 1] - walls[i]
    sign = "+" if d > 0 else ""
    color = RED if d > 0 else GREEN
    ax.annotate(f"{label}: {sign}{d:.1f} s",
                xy=((walls[i] + walls[i + 1]) / 2, i + 0.5),
                fontsize=8.5, color=color, ha="center", va="center")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlim(0, 52)
ax.set_xlabel("seconds (10k documents, 5 filters, one H100)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Forward-pass ablation", fontsize=12, fontweight="bold",
             loc="left")

fig.tight_layout()
fig.savefig(OUT / "ablation_ladder.png", bbox_inches="tight")
print("wrote", OUT / "ablation_ladder.png")

# ---------------- profile: where A2's time goes vs A3 -----------------
# Per-token GPU kernel time by bucket, from the chrome traces banked
# in results/ablation_profile.json (kernel events only).

prof = load("ablation_profile.json")
buckets = ["gemm", "attention", "small kernels", "KV/copies", "other"]


def bucketed(rung):
    cats = prof["rungs"][rung]["category_us_per_token"]
    small = cats.get("vllm_elementwise", 0) + cats.get("quant", 0) \
        + cats.get("triton_fused", 0)
    return [cats.get("gemm", 0), cats.get("attention", 0), small,
            cats.get("copies", 0), cats.get("other", 0)]


a2 = bucketed("A2")
a3 = bucketed("A3")

fig, ax = plt.subplots(figsize=(9, 3.2), dpi=150)
y = list(range(len(buckets)))
h = 0.34
ax.barh([i - h / 2 for i in y], a2, height=h, color=BLUE,
        label="A2: our executor, vLLM kernels")
ax.barh([i + h / 2 for i in y], a3, height=h, color=GREEN,
        label="A3: our executor, our kernels")
for i, (v2, v3) in enumerate(zip(a2, a3)):
    ax.text(v2 + 0.06, i - h / 2, f"{v2:.2f}", va="center", fontsize=8.5,
            color=DARK)
    ax.text(v3 + 0.06, i + h / 2, f"{v3:.2f}", va="center", fontsize=8.5,
            color=DARK)
ax.set_yticks(y)
ax.set_yticklabels(buckets, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("microseconds per token (GPU kernel time)", fontsize=9)
ax.set_xlim(0, 6.6)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Where the time goes per token: the fused kernels remove "
             "2.4 µs/token", fontsize=12, fontweight="bold", loc="left")
fig.tight_layout()
fig.savefig(OUT / "ablation_profile.png", bbox_inches="tight")
print("wrote", OUT / "ablation_profile.png")
