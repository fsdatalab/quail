"""The stock-vLLM against packed-pipeline kernel figure.

Panel A: per-token GPU kernel time by kernel role, from the banked
profiled windows (engine: results/engine/fusion_ab.json control cell;
ours: round-4 packed_custom_qk cell). Panel B: end-to-end throughput
with the wrong-answer counts.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e8e8e6"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BLUE_DARK = "#104281"

# microseconds per token, banked profiled windows (381,315 tokens)
stock = {"mm": 5.521, "attn": 0.813, "kvw": 0.427, "between": 4.004}
ours = {"mm": 5.800, "attn": 0.679, "kvw": 0.0, "between": 1.999}
stock_total = 10.77
ours_total = 8.48

fig, (ax, bx) = plt.subplots(
    2, 1, figsize=(9.6, 7.0), dpi=200,
    gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.66,
                 "top": 0.84, "bottom": 0.10})
fig.patch.set_facecolor(SURFACE)

# ---------------- Panel A: kernel time composition ----------------
ax.set_facecolor(SURFACE)
rows = [("our packed pipeline", ours, ours_total),
        ("stock vLLM engine", stock, stock_total)]
for y, (label, d, total) in enumerate(rows):
    left = 0.0
    segs = [("mm", BLUE, None), ("attn", ORANGE, None),
            ("kvw", ORANGE, "///"), ("between", AQUA, None)]
    for key, color, hatch in segs:
        w = d[key]
        if w <= 0:
            continue
        ax.barh(y, w, left=left, height=0.52, color=color,
                hatch=hatch, edgecolor=SURFACE, linewidth=1.6)
        if w > 0.55:
            ax.text(left + w / 2, y, f"{w:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        left += w
    ax.text(left + 0.12, y, f"{total:.2f} µs/token",
            ha="left", va="center", fontsize=10, color=INK,
            fontweight="bold")

ax.set_yticks([0, 1])
ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=INK)
ax.set_xlim(0, 12.6)
ax.set_ylim(-0.85, 2.05)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9)
ax.set_xlabel("GPU kernel time, microseconds per token", fontsize=9.5,
              color=INK2)

ax.annotate("same kernels both rows:\nDeepGEMM multiplies + FlashAttention-3",
            xy=(3.0, 1.55), fontsize=8.8, color=INK2, ha="center")
ax.annotate("KV-cache write\n(ours has no KV)",
            xy=(stock["mm"] + stock["attn"] + stock["kvw"] / 2, 0.72),
            xytext=(8.3, 1.62), fontsize=8.8, color=INK2, ha="center",
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
ax.annotate("engine's separate norm,\nquantize, and silu kernels",
            xy=(8.9, 0.42), fontsize=8.8, color=INK2, ha="center")
ax.annotate("the same work, done by\nour 3 fused kernels",
            xy=(7.6, -0.55), fontsize=8.8, color=INK2, ha="center")

legend = [Patch(facecolor=BLUE, label="DeepGEMM multiplies"),
          Patch(facecolor=ORANGE, label="FlashAttention-3"),
          Patch(facecolor=ORANGE, hatch="///", edgecolor=SURFACE,
                label="KV-cache write"),
          Patch(facecolor=AQUA, label="between-multiply kernels")]
ax.legend(handles=legend, loc="lower right", bbox_to_anchor=(1.0, 1.02),
          ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
ax.set_title("Where each pipeline's GPU time goes",
             fontsize=11.5, color=INK, loc="left", pad=30,
             fontweight="bold")

# ---------------- Panel B: end-to-end throughput ----------------
bx.set_facecolor(SURFACE)
configs = [
    ("packed loop, our 3 kernels\n(no engine, no KV)", 121045, BLUE_DARK),
    ("packed loop, vLLM's kernels\n(no engine, no KV)", 100308, BLUE),
    ("vLLM engine\n(fp8 KV cache)", 96946, BLUE),
]
for y, (label, rate, color) in enumerate(configs):
    bx.barh(y, rate / 1000, height=0.52, color=color,
            edgecolor=SURFACE, linewidth=1.6)
    bx.text(rate / 1000 + 1.2, y, f"{rate:,} tokens/s",
            va="center", fontsize=10, color=INK, fontweight="bold")
bx.set_yticks(range(len(configs)))
bx.set_yticklabels([c[0] for c in configs], fontsize=9.5, color=INK)
bx.set_xlim(0, 152)
bx.invert_yaxis()
bx.xaxis.grid(True, color=GRID, linewidth=0.8)
bx.set_axisbelow(True)
for spine in ("top", "right", "left"):
    bx.spines[spine].set_visible(False)
bx.spines["bottom"].set_color(GRID)
bx.tick_params(colors=INK2, labelsize=9)
bx.set_xlabel("thousand input tokens per second, one H100, "
              "10,000-document filter", fontsize=9.5, color=INK2)
bx.set_title("End-to-end, same corpus, one container",
             fontsize=11.5, color=INK,
             loc="left", pad=10, fontweight="bold")

fig.suptitle("One filter on one H100: stock vLLM against the packed "
             "forward pass", fontsize=13.5, color=INK, x=0.055,
             y=0.97, ha="left", fontweight="bold")
fig.text(0.055, 0.008,
         "Kernel times from profiled 381,315-token windows (containers "
         "differ by about ±3%); stock profiled at its committed fp8-KV "
         "setting. All three throughput rungs ran in one container "
         "(round-6 ladder). Same kernels without the engine or cache is "
         "+3.5%; the three custom kernels are the rest.",
         fontsize=7.8, color=INK2)

fig.savefig("/home/user/quail-exploration/results/plots/"
            "stock_vs_packed_kernels.png",
            bbox_inches="tight", facecolor=SURFACE)
print("saved")
