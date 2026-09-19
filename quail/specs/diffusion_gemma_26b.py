from dataclasses import replace

from .base import ModelSpec

# Gemma 4 attention runs with softmax scale 1.0 and a 1024-token
# sliding window on 25 of the 30 layers. The five full-attention
# layers (every sixth, starting at layer 5) use 512-wide heads with two
# KV heads; the sliding layers use 256-wide heads with eight.
SLIDING_KV = (8, 256)
FULL_KV = (2, 512)
LAYER_KV = tuple(FULL_KV if (index + 1) % 6 == 0 else SLIDING_KV
                 for index in range(30))
SLIDING_LAYERS = tuple(index for index in range(30) if (index + 1) % 6)

# Per layer: a dense MLP (3 x 2816 x 2112) beside 128 experts of
# 3 x 2816 x 704, eight of them active per token.
DENSE_MLP = 3 * 2816 * 2112
EXPERT = 3 * 2816 * 704
# Attention projections: q is 16 heads wide at the layer's head dim,
# k and v the layer's KV heads; o maps back to hidden.
SLIDING_ATTN = 2816 * (16 * 256 + 2 * 8 * 256) + 16 * 256 * 2816
FULL_ATTN = 2816 * (16 * 512 + 2 * 2 * 512) + 16 * 512 * 2816

DIFFUSION_GEMMA_26B_FP8 = ModelSpec(
    name="diffusion-gemma-26b-a4b-fp8",
    params=3.09e9,         # params one token multiplies, embeddings
    #                        aside: 25 sliding + 5 full attention
    #                        layers, each with the dense MLP and eight
    #                        active experts
    layers=30,
    hidden=2816,
    n_q=16,
    n_kv=8,
    d_head=256,
    ffn_width=10240,       # widest single GEMM: the full layers' qkv
    #                        projection (16 x 512 + 2 x 2 x 512)
    w_bytes=1.0,           # fp8 weights, per-channel scales
    hf_name="RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic",
    kv_bytes=2.0,
    w_mem_bytes=27_682_404_352,    # measured as-loaded footprint (the
    #                                probe cell): fp8 linears and
    #                                experts + bf16 embeddings, vision
    #                                tower, router, and
    #                                self-conditioning weights
    vocab=262_144,
    tied_head=True,
    weight_precision="fp8",
    arch="diffusion_gemma",
    layer_kv=LAYER_KV,
    sliding_window=1024,
    sliding_layers=SLIDING_LAYERS,
    # With thinking off the model opens its turn with an empty thinking
    # channel, four tokens: <|channel> thought \n <channel|>. The prompt
    # prefills that channel and the canvas holds the one answer row.
    # The checkpoint's canvas_length is 256; on QUAIL-B IMDB the
    # one-row canvas agrees with the reference within 0.01 of the
    # 256-row canvas on every query, at 3 to 8 times lower runtime
    # (runs 20260918T230410Z-cfe73359 and 20260918T155902Z-be7238e7 on
    # the quail-results volume).
    canvas_tokens=1,
    turn_prefix="<bos><|turn>user\n",
    turn_suffix="<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    canvas_answer_row=0,
    attn_params_per_layer=(25 * SLIDING_ATTN + 5 * FULL_ATTN) // 30,
    mlp_active_params_per_layer=DENSE_MLP + 8 * EXPERT,
    mlp_total_params_per_layer=DENSE_MLP + 128 * EXPERT,
    # on one 35k-row chunk the Triton experts took 0.206 s against
    # 0.218 s for vLLM's CUTLASS grouped GEMM
    # (/results/ablations/diffusion_gemma_layer_timing_triton_random*.json)
    moe_backend="triton",
    chunk_cap_tokens=65_536,    # the MoE kernels' token workspace and
    #                             the 512-wide attention heads grow
    #                             with the chunk; the kernel index cap
    #                             alone would allow 209k
)

# The checkpoint's own 256-row canvas, for reference runs: the model
# writes its empty thinking channel inside the canvas and the answer
# is read at canvas row 4.
DIFFUSION_GEMMA_26B_FP8_CANVAS256 = replace(
    DIFFUSION_GEMMA_26B_FP8, name="diffusion-gemma-26b-a4b-fp8-canvas256",
    turn_suffix="<turn|>\n<|turn>model\n",
    canvas_tokens=256, canvas_answer_row=4)

# Shorter canvases in that layout, as the benchmark measured them.
DIFFUSION_GEMMA_26B_FP8_CANVAS32 = replace(
    DIFFUSION_GEMMA_26B_FP8_CANVAS256,
    name="diffusion-gemma-26b-a4b-fp8-canvas32", canvas_tokens=32)
DIFFUSION_GEMMA_26B_FP8_CANVAS8 = replace(
    DIFFUSION_GEMMA_26B_FP8_CANVAS256,
    name="diffusion-gemma-26b-a4b-fp8-canvas8", canvas_tokens=8)
