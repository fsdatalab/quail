"""Draw DiffusionGemma's forward pass before and after expert fusion.

Run from the repository root:
    uv run --with matplotlib python reports/make_expert_fusion_diagram.py

The figure describes code, not measured kernel launch counts. The unfused
column is revision 189bfaa. The fused column is revision f76f7ed. It adds
expert normalization plus
quantization, and expert GELU plus multiplication plus quantization.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from plot_colors import BLUE, DARK, GRAY, GREEN

ROOT = Path(__file__).parent
plt.style.use(ROOT / "quail.mplstyle")

# Aligned rows preserve the location of the operations removed by fusion.
COMMON = {
    0: "Token embeddings + canvas normalization\nPrompt + one fixed canvas token",
    1: "Input normalization + FP8 quantization\nFuse the prior layer's residual addition",
    2: "QKV projection\nFP8 matrix multiplication",
    3: "Q/K/V normalization + rotary positions\nqkv_norm_rope",
    4: "Write KV + run attention\nSliding or full attention, by layer",
    5: "FP8 quant + output projection\nPost-attention normalization",
    6: "Residual addition + dense input norm + FP8 quant\nscale_add_rms_norm_quant",
    7: "Dense gate and up projection\nFP8 matrix multiplication",
    8: "Dense GELU + multiplication + FP8 quant\ngelu_mul_quant",
    9: "Dense down projection + normalization\nKeep the dense output for the branch sum",
    11: "Router projection + select eight experts\n_gemma4_routing_kernel",
    13: "Assign rows + expert gate and up projection\nfused_moe_kernel, w13",
    16: "Expert down projection\nfused_moe_kernel, w2",
    17: "Sum the eight weighted expert outputs\nmoe_sum",
    18: "Expert output norm + add dense output + norm\nfused_add_rms_norm",
    19: "Continue to the next layer\nInclude the scaled residual",
    20: "After 30 layers, apply the final normalization\nCompare TRUE and FALSE at the canvas",
}
LEFT = dict(COMMON)
LEFT.update({
    10: "Expert input norm + router input norm\nrms_norm2",
    12: "Expert input FP8 quantization\ndynamic_per_token_scaled_fp8_quant",
    14: "Expert GELU + multiplication\nact_and_mul",
    15: "Expert activation FP8 quantization\ndynamic_per_token_scaled_fp8_quant",
})
RIGHT = dict(COMMON)
RIGHT.update({
    10: "Expert input norm + FP8 quant + router norm\nrms_norm2, QUANTIZE=True",
    14: "Expert GELU + multiplication + FP8 quant\ngelu_mul_quant, ROUND_ACTIVATION=True",
})

WIDTH, STEP, HEIGHT = 5.7, 0.70, 0.54
TOP = 15.4
fig, ax = plt.subplots(figsize=(12.8, 17.2))
fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02)
ax.set(xlim=(0, 12.4), ylim=(-0.35, 17.0))
ax.axis("off")
ax.text(6.2, 16.75, "DiffusionGemma forward pass", ha="center",
        va="center", fontsize=23, weight="bold", color=DARK)
ax.text(6.2, 16.36, "Quail, 26B-A4B FP8  |  Green borders mark the new fusions",
        ha="center", va="center", fontsize=15, color=DARK)

for col, (heading, rows) in enumerate((("Before", LEFT), ("After", RIGHT))):
    x = 0.25 + col * 6.2
    center = x + WIDTH / 2
    ax.text(center, 15.92, heading, ha="center", va="center",
            fontsize=19, weight="bold")
    previous_bottom = None
    for row, label in sorted(rows.items()):
        top = TOP - row * STEP
        height = HEIGHT + (STEP if col == 1 and row == 14 else 0)
        changed = col == 1 and row in (10, 14)
        color = GREEN if changed else BLUE if row in (0, 20) else GRAY
        if previous_bottom is not None:
            ax.add_patch(FancyArrowPatch(
                (center, previous_bottom), (center, top),
                arrowstyle="-|>", mutation_scale=11,
                color=DARK, linewidth=1.0, shrinkA=2, shrinkB=2))
        ax.add_patch(Rectangle((x, top - height), WIDTH, height,
                               facecolor="white", edgecolor=color,
                               linewidth=2.7 if changed else 1.15))
        ax.text(center, top - height / 2, label, ha="center", va="center",
                fontsize=12.8, linespacing=1.35, color=DARK,
                weight="bold" if changed else "normal")
        previous_bottom = top - height

ax.text(6.2, 0.38,
        "Read each column from top to bottom. Boxes may group several kernels.",
        ha="center", fontsize=12.5)
ax.text(6.2, 0.08,
        "The fusions keep the BF16 rounding steps and FP8 scales of the separate operations.",
        ha="center", fontsize=12.5)
ax.text(6.2, -0.22,
        "Expert selection, matrix multiplications, and the weighted sum use the same kernels.",
        ha="center", fontsize=12.5)
output = ROOT / "plots" / "diffusion_gemma_expert_fusion.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=300)
print(output)
