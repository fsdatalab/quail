"""Draw the kernel categories in DiffusionGemma's forward pass.

Run from the repository root:
    uv run --with matplotlib python reports/make_expert_fusion_diagram.py

The columns describe Quail revisions 189bfaa and f76f7ed. Colors follow the
kernel diagram in PR #133. Boxes are operations, not measured launch counts.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from plot_colors import DARK

ROOT = Path(__file__).parent
plt.style.use(ROOT / "quail.mplstyle")

# The category colors match the forward-pass figure in PR #133.
COLORS = {
    "quail": ("#d6f2e3", "#1f8a5a", "Quail fused Triton"),
    "gemm": ("#dbe8f8", "#2f6db5", "CUTLASS FP8 matrix multiplication"),
    "moe": ("#e6dcf5", "#6e4bb3", "vLLM expert kernel, Quail tile settings"),
    "attention": ("#fdebc8", "#d08a1a", "FlashAttention 3 or 4"),
    "kv": ("#dff1fb", "#3b8fc4", "Quail KV write"),
    "vllm": ("#e7e7e7", "#777777", "Separate vLLM operation"),
    "vfused": ("#f0f3d6", "#7a8a1f", "vLLM fused CUDA operation"),
    "cublas": ("#fbe0e6", "#c04a6a", "cuBLAS BF16 matrix multiplication"),
}

COMMON = {
    0: ("Input norm + FP8 quant, with prior residual add\nscale_add_rms_norm_quant", "quail"),
    1: ("QKV projection\nCUTLASS FP8 matrix multiplication", "gemm"),
    2: ("Q/K/V norms + rotary positions\nqkv_norm_rope", "quail"),
    3: ("Write this layer's KV\nkv_row_scatter", "kv"),
    4: ("Attention over the layer's KV pages\nFA3 for sliding layers; FA4 for full layers", "attention"),
    5: ("Output projection input\nPer-token FP8 quantization", "vllm"),
    6: ("Attention output projection\nCUTLASS FP8 matrix multiplication", "gemm"),
    7: ("Post-attention normalization\nrms_norm", "vllm"),
    8: ("Residual add + dense input norm + FP8 quant\nscale_add_rms_norm_quant", "quail"),
    9: ("Dense gate and up projection\nCUTLASS FP8 matrix multiplication", "gemm"),
    10: ("Dense GELU + multiplication + FP8 quant\ngelu_mul_quant", "quail"),
    11: ("Dense down projection\nCUTLASS FP8 matrix multiplication", "gemm"),
    12: ("Normalize the dense branch's output\nrms_norm", "vllm"),
    14: ("Router projection\nBF16 matrix multiplication", "cublas"),
    15: ("Select eight experts per token\n_gemma4_routing_kernel", "vllm"),
    17: ("Assign rows to experts\nmoe_align_block_size + sort", "vllm"),
    18: ("Expert gate and up projection\nfused_moe_kernel, w13", "moe"),
    21: ("Expert down projection\nfused_moe_kernel, w2", "moe"),
    22: ("Sum the eight weighted expert outputs\nmoe_sum", "vllm"),
    23: ("Normalize the expert branch's output\nrms_norm", "vllm"),
    24: ("Add dense and expert outputs + normalize\nfused_add_rms_norm", "vfused"),
}
LEFT = dict(COMMON)
LEFT.update({
    13: ("Expert input norm + router input norm\nrms_norm2", "quail"),
    16: ("Expert input\nPer-token FP8 quantization", "vllm"),
    19: ("Expert GELU + multiplication\nact_and_mul", "vllm"),
    20: ("Expert activation\nPer-token FP8 quantization", "vllm"),
})
RIGHT = dict(COMMON)
RIGHT.update({
    13: ("Expert input norm + FP8 quant + router norm\nrms_norm2, QUANTIZE=True", "quail"),
    19: ("Expert GELU + multiplication + FP8 quant\ngelu_mul_quant, ROUND_ACTIVATION=True", "quail"),
})

WIDTH, STEP, HEIGHT = 5.4, 0.72, 0.57
X_POSITIONS = (0.2, 6.8)
TOP = 20.4
fig, ax = plt.subplots(figsize=(12.8, 24.0))
fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02)
ax.set(xlim=(0, 12.4), ylim=(-1.1, 22.6))
ax.axis("off")
ax.text(6.2, 22.25, "DiffusionGemma forward-pass kernels", ha="center",
        va="center", fontsize=23, weight="bold", color=DARK)
ax.text(6.2, 21.80, "Quail, 26B-A4B FP8. Colors identify every kernel category.",
        ha="center", va="center", fontsize=15, color=DARK)
ax.text(6.2, 21.42, "Prompt and one canvas token enter the layer sequence below.",
        ha="center", va="center", fontsize=13.5)

for col, (heading, rows) in enumerate((
        ("Before expert fusion", LEFT), ("With expert fusion", RIGHT))):
    x = X_POSITIONS[col]
    center = x + WIDTH / 2
    ax.text(center, 20.94, heading, ha="center", va="center",
            fontsize=19, weight="bold")
    previous_bottom = None
    for row, (label, kind) in sorted(rows.items()):
        top = TOP - row * STEP
        height = HEIGHT + (STEP if col == 1 and row == 19 else 0)
        fill, border, _ = COLORS[kind]
        if previous_bottom is not None:
            ax.add_patch(FancyArrowPatch(
                (center, previous_bottom), (center, top),
                arrowstyle="-|>", mutation_scale=11,
                color=DARK, linewidth=1.0, shrinkA=2, shrinkB=2))
        ax.add_patch(FancyBboxPatch(
            (x, top - height), WIDTH, height,
            boxstyle="round,pad=0.015,rounding_size=0.045",
            facecolor=fill, edgecolor=border, linewidth=1.5))
        ax.text(center, top - height / 2, label, ha="center", va="center",
                fontsize=13.2, linespacing=1.35, color=DARK)
        previous_bottom = top - height

# The fused norm has two outputs. The BF16 router input follows the main
# path; the FP8 expert input bypasses the router and enters the first expert
# matrix multiplication after the router assigns rows to experts.
right_x = X_POSITIONS[1]
bypass_x = right_x - 0.34
source_y = TOP - 13 * STEP - HEIGHT / 2
target_y = TOP - 18 * STEP - HEIGHT / 2
ax.plot([right_x, bypass_x], [source_y, source_y], color="#1f8a5a", lw=1.6)
ax.plot([bypass_x, bypass_x], [source_y, target_y], color="#1f8a5a", lw=1.6)
ax.add_patch(FancyArrowPatch(
    (bypass_x, target_y), (right_x, target_y),
    arrowstyle="-|>", mutation_scale=12, color="#1f8a5a", linewidth=1.6,
    shrinkA=0, shrinkB=2))
ax.text(bypass_x - 0.09, (source_y + target_y) / 2,
        "FP8 expert input", rotation=90, ha="center", va="center",
        fontsize=11.5, color="#1f8a5a",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

ax.text(6.2, 2.10,
        "Repeat for 30 layers, then apply the final norm and compare TRUE and FALSE at the canvas.",
        ha="center", fontsize=12.5)
ax.text(6.2, 1.78,
        "The first layer only normalizes and quantizes its input; later layers include the prior residual add.",
        ha="center", fontsize=12.0)
ax.text(6.2, 1.46,
        "Each box names an operation. Assignment and attention may use multiple kernel launches.",
        ha="center", fontsize=12.0)

for i, (fill, border, label) in enumerate(COLORS.values()):
    x = 0.3 + (i % 2) * 6.2
    y = 0.96 - (i // 2) * 0.43
    ax.add_patch(Rectangle((x, y - 0.10), 0.30, 0.20,
                           facecolor=fill, edgecolor=border, linewidth=1.4))
    ax.text(x + 0.43, y, label, va="center", fontsize=13)

ax.text(6.2, -0.85,
        "Both existing and new Quail fusions are green. Expert selection and matrix kernels are unchanged.",
        ha="center", fontsize=12.3)
output = ROOT / "plots" / "diffusion_gemma_expert_fusion.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=300)
print(output)
