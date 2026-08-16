"""Every kernel in one decoder layer, in order, stock against ours.

Stock's sequence is the compiled engine's, read from the banked
profiles (fusion_ab control cell kernel names); ours is round 4's
packed_custom_qk path. Gray links map each stock kernel to what
replaced it; converging links are the fusions.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
LINK = "#c9c8c4"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

MM, ATTN, BETWEEN = BLUE, ORANGE, AQUA

# (label line 1, label line 2, role color, hatch, ours-made)
stock = [
    ("input norm + residual add", "compiler-fused triton", BETWEEN, None),
    ("quantize to 8-bit", "per_token_group_quant", BETWEEN, None),
    ("qkv multiply", "DeepGEMM  6144x2560", MM, None),
    ("q/k head norm + rotate", "compiler-fused triton", BETWEEN, None),
    ("KV-cache write (fp8)", "reshape_and_cache_flash", ATTN, "///"),
    ("attention", "FlashAttention-3", ATTN, None),
    ("quantize to 8-bit", "per_token_group_quant", BETWEEN, None),
    ("o multiply", "DeepGEMM  2560x4096", MM, None),
    ("post-attention norm + add", "compiler-fused triton", BETWEEN, None),
    ("quantize to 8-bit", "per_token_group_quant", BETWEEN, None),
    ("gate_up multiply", "DeepGEMM  19456x2560", MM, None),
    ("silu and multiply", "compiler-fused triton", BETWEEN, None),
    ("quantize to 8-bit", "per_token_group_quant", BETWEEN, None),
    ("down multiply", "DeepGEMM  2560x9728", MM, None),
]

ours = [
    ("norm + add + quantize", "add_rms_norm_quant - OURS", BETWEEN, True),
    ("qkv multiply", "DeepGEMM  6144x2560", MM, False),
    ("q/k norm + rotate + layout", "qk_norm_rope - OURS", BETWEEN, True),
    ("attention, no KV written", "FlashAttention-3", ATTN, False),
    ("quantize to 8-bit", "per_token_group_quant", BETWEEN, False),
    ("o multiply", "DeepGEMM  2560x4096", MM, False),
    ("norm + add + quantize", "add_rms_norm_quant - OURS", BETWEEN, True),
    ("gate_up multiply", "DeepGEMM  19456x2560", MM, False),
    ("silu + multiply + quantize", "silu_mul_quant - OURS", BETWEEN, True),
    ("down multiply", "DeepGEMM  2560x9728", MM, False),
]

# stock indices (0-based) feeding each ours box; None = removed
mapping = [
    ([0, 1], 0), ([2], 1), ([3], 2), ([5], 3), ([6], 4), ([7], 5),
    ([8, 9], 6), ([10], 7), ([11, 12], 8), ([13], 9),
]
removed = 4  # KV-cache write has no counterpart

ROW = 1.0
BOX_H = 0.74
LX, LW = 0.4, 4.0
RX, RW = 7.2, 4.0

fig, ax = plt.subplots(figsize=(9.8, 11.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 11.8)
ax.set_ylim(-14.9, 1.9)
ax.axis("off")


def box(x, y, w, label1, label2, color, hatch=None, ours_made=False):
    edge = INK if ours_made else ("#f4b596" if hatch else SURFACE)
    patch = FancyBboxPatch(
        (x, y - BOX_H / 2), w, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color, hatch=hatch, edgecolor=edge,
        linewidth=2.0 if ours_made else 1.2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + 0.13, label1, ha="center", va="center",
            fontsize=9.2, color="white", fontweight="bold")
    ax.text(x + w / 2, y - 0.20, label2, ha="center", va="center",
            fontsize=7.4, color="white", family="monospace")


stock_y = {}
for i, (l1, l2, color, hatch) in enumerate(stock):
    y = -i * ROW
    stock_y[i] = y
    box(LX, y, LW, l1, l2, color, hatch=hatch)

for sources, j in mapping:
    y = sum(stock_y[s] for s in sources) / len(sources)
    l1, l2, color, ours_made = ours[j]
    box(RX, y, RW, l1, l2, color, ours_made=ours_made)
    for s in sources:
        ax.plot([LX + LW + 0.06, RX - 0.06], [stock_y[s], y],
                color=LINK, linewidth=1.4, zorder=0)

ax.text(RX - 0.25, stock_y[removed], "removed - ours keeps no KV",
        ha="left", va="center", fontsize=8.6, color=INK2, style="italic")

ax.text(LX + LW / 2, 1.15, "stock vLLM engine", ha="center",
        fontsize=11.5, color=INK, fontweight="bold")
ax.text(LX + LW / 2, 0.72, "14 kernels per layer", ha="center",
        fontsize=9, color=INK2)
ax.text(RX + RW / 2, 1.15, "our packed pipeline", ha="center",
        fontsize=11.5, color=INK, fontweight="bold")
ax.text(RX + RW / 2, 0.72, "10 kernels per layer", ha="center",
        fontsize=9, color=INK2)

# side rail marking the repeat
ax.plot([0.12, 0.12], [-13.55, 0.55], color=INK2, linewidth=1.0)
ax.text(0.02, -6.5, "one decoder layer, repeated 36 times per chunk",
        rotation=90, va="center", ha="center", fontsize=8.6, color=INK2)

legend_items = [
    (MM, None, False, "matrix multiplies (DeepGEMM) - same both sides"),
    (ORANGE, None, False, "attention (FlashAttention-3) - same both sides"),
    (ORANGE, "///", False, "KV-cache write - stock only"),
    (AQUA, None, False, "kernels between the multiplies"),
    (AQUA, None, True, "black outline: written by us (Triton)"),
]
ly = -14.15
for k, (color, hatch, outlined, text) in enumerate(legend_items):
    x0 = 0.4 + (k % 2) * 5.8
    y0 = ly - (k // 2) * 0.52
    ax.add_patch(FancyBboxPatch(
        (x0, y0 - 0.13), 0.42, 0.26,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor=color, hatch=hatch,
        edgecolor=INK if outlined else SURFACE,
        linewidth=1.8 if outlined else 1.0))
    ax.text(x0 + 0.56, y0, text, ha="left", va="center", fontsize=8.6,
            color=INK)

ax.set_title(
    "Every kernel in one decoder layer, in execution order",
    fontsize=13, color=INK, loc="left", pad=14, fontweight="bold")
fig.text(0.055, 0.012,
         "Stock sequence read from the compiled engine's profiled kernel "
         "names (fp8-KV production boot); ours is the round-4 path. "
         "Converging links are the three fusions. Layer 0 runs its first "
         "norm unfused in both; the final norm and the 12-row output "
         "projection follow the last layer.",
         fontsize=7.8, color=INK2)

fig.savefig("/home/user/quail-exploration/results/plots/"
            "kernel_sequence.png", bbox_inches="tight",
            facecolor=SURFACE)
print("saved")
