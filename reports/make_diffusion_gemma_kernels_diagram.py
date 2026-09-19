"""Draw the kernels one DiffusionGemma layer runs on three paths.

    uv run --with matplotlib python reports/make_diffusion_gemma_kernels_diagram.py

Writes reports/plots/diffusion_gemma_kernels_per_layer.png. The
columns are Quail's unfused reference path (deleted after the fused
path matched it), Quail's fused single-call path (the default), and
stock vLLM's compiled graph, from the profile saved at
/results/ablations/diffusion_gemma_stock_kernels.json on the
quail-results volume.
"""

import textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent / "plots" / "diffusion_gemma_kernels_per_layer.png"
K = {"gemm": ("#dbe8f8", "#2f6db5"), "moe": ("#e6dcf5", "#6e4bb3"), "fa": ("#fdebc8", "#d08a1a"),
     "kv": ("#dff1fb", "#3b8fc4"), "vllm": ("#e7e7e7", "#777777"), "vfused": ("#f0f3d6", "#7a8a1f"), "inductor": ("#fbe3e3", "#b8434a"),
     "quail": ("#d6f2e3", "#1f8a5a"), "cublas": ("#fbe0e6", "#c04a6a")}
COLS = ["vLLM kernels called one by one\n(Quail reference path, measured 09-18)",
        "Quail fused, single call\n(default)",
        "Stock vLLM, compiled\n(-O2 Inductor, profiled 09-19)"]
V, Q, G, F, KV, M, C, VF, I = "vllm", "quail", "gemm", "fa", "kv", "moe", "cublas", "vfused", "inductor"
moe_rows = [[("moe_align_block_size + sort", V), ("moe_align_block_size + sort", V), ("radix sort x2, expert offsets, gemm starts", V)],
            [("per_token fp8 quant", V), ("per_token fp8 quant", V), ("expandInputRows (fp8 gather)", V)],
            [("fused_moe_kernel (w13), Triton, vLLM tile table", V), ("fused_moe_kernel (w13), Triton, Quail tile table", M), ("cutlass_3x_group_gemm (w13)", G)],
            [("act_and_mul (gelu_tanh)", V)] * 3,
            [("per_token fp8 quant", V)] * 3,
            [("fused_moe_kernel (w2), Triton, vLLM tile table", V), ("fused_moe_kernel (w2), Triton, Quail tile table", M), ("cutlass_3x_group_gemm (w2)", G)],
            [("moe_sum (top-8 weighted sum)", V), ("moe_sum (top-8 weighted sum)", V), ("finalizeMoeRouting (top-8 sum)", V)]]
# each stage: list of rows; each row: 3 entries (text, kind) or None, so the same kernel sits on the same row
stages = [
 ("add + norm\n+ quant", [
    [("rms_norm (input)", V), ("scale_add_rms_norm_quant", Q), ("Inductor: add + scalar + rms_norm + fp8 quant", I)],
    [("+ residual, x layer_scalar", V), None, None],
    [("per_token fp8 quant", V), None, None]]),
 ("QKV matmul", [[("cutlass fp8 gemm (qkv_proj)", G)] * 3]),
 ("Q/K/V norms\n+ rotary", [
    [("copy (k)", V), ("qkv_norm_rope", Q), ("Inductor: q/k head norms + rotary (2 kernels)", I)],
    [("copy (v)", V), None, None], [("rms_norm (q heads)", V), None, None],
    [("rms_norm (k heads)", V), None, None], [("rms_norm (v heads, no weight)", V), None, None],
    [("rotary_embedding", V), None, None]]),
 ("KV page write", [[("kv_row_scatter (the layer's pool)", KV), ("kv_row_scatter (the layer's pool)", KV), ("reshape_and_cache_flash", V)]]),
 ("attention", [
    [("FA3 causal + block table, window 1024 [25 sliding]", F), ("FA3 causal + block table, window 1024 [25 sliding]", F), ("FA4 causal + block table, window 1024 [25 sliding]", F)],
    [("FA4 causal + block table, 512-wide heads [5 full]", F)] * 3]),
 ("quant for o_proj", [[("per_token fp8 quant", V)] * 3]),
 ("output matmul", [[("cutlass fp8 gemm (o_proj)", G)] * 3]),
 ("norm, add,\nnorm + quant", [
    [("rms_norm (post_attention)", V), ("rms_norm (post_attention)", V), ("Inductor: rms_norm + add + rms_norm + fp8 quant", I)],
    [("+ residual", V), ("scale_add_rms_norm_quant (scale 1)", Q), None],
    [("rms_norm (pre_feedforward)", V), None, None], [("per_token fp8 quant", V), None, None]]),
 ("dense MLP\n(2112 wide)", [
    [("cutlass fp8 gemm (gate_up_proj)", G)] * 3,
    [("act_and_mul (gelu_tanh)", V), ("gelu_mul_quant", Q), ("Inductor: gelu + fp8 quant", I)],
    [("per_token fp8 quant", V), None, None],
    [("cutlass fp8 gemm (down_proj)", G)] * 3]),
 ("norms before\nthe experts", [
    [("rms_norm (post_ffn_1, dense)", V), ("rms_norm (post_ffn_1, dense)", V), ("Inductor: the three norms + router scale", I)],
    [("rms_norm (pre_ffn_2, residual)", V), ("rms_norm2 (pre_ffn_2 + router, one read)", Q), None],
    [("rms_norm (router, residual)", V), None, None], [("x router scale", V), None, None]]),
 ("router", [[("bf16 gemm (router.proj, cuBLAS)", C)] * 3, [("_gemma4_routing_kernel (top-8 of 128)", V)] * 3]),
 ("experts\n(128 x 704 wide,\ntop 8)", moe_rows),
 ("sum branches,\nnorm, add", [
    [("rms_norm (post_ffn_2, routed)", V), ("rms_norm (post_ffn_2, routed)", V), ("Inductor: norm + add + norm + add (+ scalar in\nthe next layer's kernel)", I)],
    [("dense + routed", V), ("fused_add_rms_norm (post_feedforward)", VF), None],
    [("rms_norm (post_feedforward)", V), None, None],
    [("+ residual, x layer_scalar", V), ("(scalar folded into the norm weight)", None), None]]),
 ("after the layers", [
    [None, None, ("canvas denoising: lm_head GEMM + log_softmax /\nargmax kernels per step, 15% of stock's time", I)]]),
]
launches = [40, 26, 27]
BOX_H, GAP, STAGE_GAP = 0.34, 0.07, 0.24
col_w, col_x0, col_gap = 5.4, 1.9, 0.3
heights = [len(rows) * (BOX_H + GAP) - GAP for _, rows in stages]
total_h = sum(heights) + STAGE_GAP * (len(stages) - 1)
top = total_h + 1.4
fig_w = col_x0 + 3 * col_w + 2 * col_gap + 0.4
fig, ax = plt.subplots(figsize=(fig_w * 0.95, (top + 3.1) * 0.95))
fig.patch.set_facecolor("white")
ax.set_xlim(0, fig_w); ax.set_ylim(-2.8, top + 0.2); ax.axis("off")
ax.text(0.2, top + 0.05, "Kernels per layer: DiffusionGemma 26B-A4B fp8, one H100", fontsize=15, weight="bold", va="bottom")
for ci, name in enumerate(COLS):
    x = col_x0 + ci * (col_w + col_gap)
    ax.add_patch(FancyBboxPatch((x, -0.35), col_w, total_h + 1.4, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#f7f8fa", ec="#cfd3da", lw=1.2, zorder=0))
    ax.text(x + col_w / 2, total_h + 0.85, name, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x + col_w / 2, total_h + 0.3, "%d launches / layer" % launches[ci], ha="center", va="center", fontsize=9.5, color="#555")
    ax.plot([x + 0.15, x + col_w - 0.15], [total_h + 0.06, total_h + 0.06], color="#cfd3da", lw=1)
y = total_h
for (label, rows), h in zip(stages, heights):
    ax.text(col_x0 - 0.25, y - h / 2, label, ha="right", va="center", fontsize=9.5, color="#333")
    by = y
    for row in rows:
        for ci, cell in enumerate(row):
            if cell is None:
                continue
            text, kind = cell
            x = col_x0 + ci * (col_w + col_gap) + 0.2
            if kind is None:
                ax.text(x + (col_w - 0.4) / 2, by - BOX_H / 2, text, ha="center", va="center", fontsize=8, color="#777", style="italic")
                continue
            fc, ec = K[kind]
            ax.add_patch(FancyBboxPatch((x, by - BOX_H), col_w - 0.4, BOX_H, boxstyle="round,pad=0.01,rounding_size=0.06", fc=fc, ec=ec, lw=1.4))
            ax.text(x + (col_w - 0.4) / 2, by - BOX_H / 2, text, ha="center", va="center", fontsize=8.1, family="monospace", color="#222")
        by -= BOX_H + GAP
    if y - h - STAGE_GAP > 0.05:
        for ci in range(3):
            xc = col_x0 + ci * (col_w + col_gap) + 0.2 + (col_w - 0.4) / 2
            ax.annotate("", xy=(xc, y - h - STAGE_GAP + 0.03), xytext=(xc, y - h - 0.02), arrowprops=dict(arrowstyle="-|>", color="#888", lw=1))
    y -= h + STAGE_GAP
items = [("gemm", "CUTLASS fp8 GEMM (vLLM)"), ("moe", "Triton fused MoE (vLLM kernel, Quail-tuned tile table)"), ("fa", "FlashAttention 3 / 4 (varlen, block table over arena pages)"),
         ("kv", "KV page write (Quail Triton)"), ("vllm", "vLLM op"), ("vfused", "vLLM fused CUDA op"), ("cublas", "cuBLAS bf16"), ("quail", "Quail fused Triton"), ("inductor", "Inductor-generated Triton (stock, compiled)")]
for i, (kind, text) in enumerate(items):
    xx = 0.2 + (i % 3) * 6.3; yy = -0.75 - (i // 3) * 0.42
    fc, ec = K[kind]
    ax.add_patch(FancyBboxPatch((xx, yy - 0.13), 0.3, 0.26, boxstyle="round,pad=0.01,rounding_size=0.05", fc=fc, ec=ec, lw=1.4))
    ax.text(xx + 0.42, yy, text, va="center", fontsize=9.3)
note = ("One layer of 30. Layer kinds: 25 sliding layers (head 256, 16 q / 8 kv heads, window 1024) and 5 full-attention layers (5, 11, 17, 23, 29: "
        "head 512, 2 kv heads with k = v). Every layer has the dense MLP and the experts. Quail runs FA3 on the sliding layers and FA4 on the full "
        "layers, reading the arena through FlashAttention's block table; stock vLLM runs FA4 on all 30 (its diffusion path needs FA4's per-sequence "
        "causal mask). Stock column from one profiled prefill pass at vLLM's defaults (-O2, Inductor, no vLLM fusion pass fired): the experts run "
        "CUTLASS grouped GEMM (31% of its GPU time), the dense GEMMs 13%, attention 4%, and the 256-row canvas denoising loop about 15%, which "
        "Quail's one-row canvas never runs. Record: /results/ablations/diffusion_gemma_stock_kernels.json. The default column runs the experts through vLLM's "
        "Triton fused MoE kernel with a tile table Quail tuned for 32k and 64k row chunks; the reference column, measured before that table existed, "
        "ran the same kernel with vLLM's own table, which stops at 16k rows. At chunk size the tuned tiles beat the CUTLASS grouped GEMM by 6% "
        "(/results/ablations/diffusion_gemma_moe_tiles*.json). A 35k-row chunk on the default column: "
        "0.363 s, 96k rows/s.")
ax.text(0.2, -2.25, "\n".join(textwrap.wrap(note, 185)), fontsize=8.8, color="#444", va="top")
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("ok")
