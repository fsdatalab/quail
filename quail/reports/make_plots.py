"""Plots for the 2026-08-18 reports (filter profiling, QUAIL-B).

Reads the measured JSON artifacts in results/ and writes PNGs into
reports/plots/. Run from the quail/ directory:

    uv run --with matplotlib python reports/make_plots.py
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
RED = "#E53935"
GREEN = "#43A047"
DARK = "#424242"


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


# ---- figure 1: GPU time breakdown (profiled 3k-doc filter) -----------

prof = load("profile_filter.json")
cat = prof["category_s"]

# The profiler's "copies" bucket counted the aten::index_copy_ op AND
# its kernel (1.86 s each); the true KV-write scatter cost is 1.86 s,
# and the rest of the copies bucket is ordinary memcpy.
kv_write = 1.86
other_copies = round(cat["copies"] - 2 * kv_write, 2)

rows = [
    ("GEMM\n(DeepGEMM fp8)", cat["gemm"], ACCENT),
    ("KV scatter\n(index_copy_)", kv_write, RED),
    ("Fused Triton\n(silu/norm)", cat["triton_fused"], ACCENT),
    ("Attention\n(flash_attn)", cat["attention"], ACCENT),
    ("fp8 quant", cat["quant"], ACCENT),
    ("Other copies", other_copies, GRAY),
    ("Other", cat["other"], GRAY),
]
fig, ax = plt.subplots(figsize=(8, 4.5))
names = [r[0] for r in rows][::-1]
vals = [r[1] for r in rows][::-1]
cols = [r[2] for r in rows][::-1]
bars = ax.barh(names, vals, color=cols, height=0.62)
ax.axvline(prof["ideal_gemm_s"], color="black", ls="--", lw=1,
           label=f"GEMM at peak ({prof['ideal_gemm_s']:.1f} s)")
for b, v in zip(bars, vals):
    ax.text(b.get_width() + 0.06, b.get_y() + b.get_height() / 2,
            f"{v:.2f} s", va="center", fontsize=9)
ax.set_xlabel("GPU self-time (seconds)")
ax.set_title("Where GPU time went, before KV scatter fix\n"
             "(3,000-doc filter, 1.15M tokens)",
             fontsize=11)
ax.set_xlim(0, 7.2)
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "profile_categories.png", dpi=150)
plt.close(fig)

# ---- figure 1b: full wall time breakdown (stacked bar) ---------------
# One horizontal stacked bar showing where the entire 29 s went:
# each kernel category from the GPU busy time, plus the idle time.

gpu_idle = prof["gap_wall_minus_busy_s"]
attn_quant = cat["attention"] + cat["quant"]
segments = [
    ("GEMM", cat["gemm"], ACCENT),
    ("KV scatter", kv_write, RED),
    ("Fused Triton", cat["triton_fused"], "#42A5F5"),
    ("Attn + quant", attn_quant, "#66BB6A"),
    ("Other GPU", other_copies + cat["other"], "#78909C"),
    ("GPU idle\n(CPU scheduling)", gpu_idle, "#E0E0E0"),
]
fig, ax = plt.subplots(figsize=(10, 3.8))
left = 0
for label, val, color in segments:
    ax.barh(0, val, left=left, height=0.55, color=color,
            edgecolor="white", linewidth=0.5,
            label=f"{label} ({val:.1f} s)")
    left += val
ax.set_xlim(0, prof["region_wall_s"] * 1.01)
ax.set_xlabel("seconds")
ax.set_yticks([])
ax.set_ylim(-0.5, 0.8)
ax.set_title(
    f"Full wall time breakdown (before KV scatter fix): "
    f"{prof['region_wall_s']:.0f} s total, "
    f"{gpu_idle:.0f} s idle ({gpu_idle / prof['region_wall_s']:.0%})",
    fontsize=11, pad=10)
ax.legend(loc="upper right", fontsize=8.5, ncol=3,
          framealpha=0.9)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "profile_busy_idle.png", dpi=150)
plt.close(fig)

# ---- figure 2: quail vs stock vLLM, before and after fix ------------

pre1 = load("m1_filter_final1.json")["runs"]
pre2 = load("m1_filter_final2.json")["runs"]
post = load("m1_filter.json")["runs"]
stockf = load("baseline_filter4.json")["runs"]
pre_wall = sum(r["wall"] for r in pre1 + pre2) / 4
post_wall = sum(r["wall"] for r in post) / 2
stock_wall = sum(r["wall"] for r in stockf) / 2

joinb = load("baseline_join.json")["runs"]
gate = load("dispatch_gate.json")
stockj = sum(r["wall"] for r in joinb) / 2
packedj = gate["gpus1"]["join"]["wall_s"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
labels = ["Stock vLLM\n(tuned)", "Quail\nbefore fix", "Quail\nafter fix"]
vals = [stock_wall, pre_wall, post_wall]
cols = [GRAY, ACCENT, GREEN]
bars = a1.bar(labels, vals, color=cols, width=0.55)
for b, v in zip(bars, vals):
    a1.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f} s",
            ha="center", fontsize=10)
a1.set_ylabel("wall time (seconds, lower is better)")
a1.set_title("5-filter chain, 10,000 documents", fontsize=11)
a1.set_ylim(0, 46)
a1.spines[["top", "right"]].set_visible(False)

labels = ["Stock vLLM\n(prefix caching)", "Quail packed"]
vals = [stockj, packedj]
bars = a2.bar(labels, vals, color=[GRAY, GREEN], width=0.5)
for b, v in zip(bars, [stockj, packedj]):
    a2.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f} s",
            ha="center", fontsize=10)
a2.set_title("72,000-pair join", fontsize=11)
a2.set_ylabel("wall time (seconds, lower is better)")
a2.set_ylim(0, 103)
a2.text(1, packedj + 12, f"{stockj / packedj:.1f}x faster",
        ha="center", fontsize=10, color=GREEN)
a2.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "vs_stock.png", dpi=150)
plt.close(fig)

# ---- figure 3: QUAIL-B SF=0.1, cold vs warm -------------------------

suite = load("quailb_sf0.1.json")
cold = {q["query"]: q for q in suite["passes"]["cold"]["queries"]}
warm = {q["query"]: q for q in suite["passes"]["warm"]["queries"]}
qids = list(cold)
y = range(len(qids))
cw = [cold[q]["wall_s"] for q in qids]
ww = [warm[q]["wall_s"] for q in qids]
restored = [sum(s.get("restored_docs", 0)
                for s in (warm[q].get("store") or {}).values())
            for q in qids]

fig, ax = plt.subplots(figsize=(9, 6.8))
h = 0.38
ax.barh([i - h / 2 for i in y], cw, height=h, color=GRAY,
        label="Cold pass (store disabled)")
ax.barh([i + h / 2 for i in y], ww, height=h, color=ACCENT,
        label="Warm pass (store enabled)")
for i, q in enumerate(qids):
    ax.text(cw[i] * 1.04, i - h / 2, f"{cw[i]:.0f} s",
            va="center", fontsize=7.5, color=DARK)
    label = f"{ww[i]:.0f} s"
    if restored[i]:
        label += f" ({restored[i]} docs restored)"
    ax.text(ww[i] * 1.04, i + h / 2, label,
            va="center", fontsize=7.5, color=ACCENT)
ax.set_yticks(list(y))
descs = [f"{q}: {cold[q]['desc']}" for q in qids]
ax.set_yticklabels(descs, fontsize=8)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(8, 3000)
ax.set_xlabel("wall time (seconds, log scale)")
ax.set_title(
    f"QUAIL-B at SF=0.1: cold pass {suite['passes']['cold']['pass_wall_s']:.0f} s, "
    f"warm pass {suite['passes']['warm']['pass_wall_s']:.0f} s",
    fontsize=11)
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "quailb_cold_warm.png", dpi=150)
plt.close(fig)

print("wrote", *[p.name for p in sorted(OUT.glob("*.png"))])
