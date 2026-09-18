from .base import ModelSpec

# Gemma 4 attention runs with softmax scale 1.0 and a 1024-token
# sliding window on 25 of the 30 layers. The five full-attention
# layers (every sixth, starting at layer 5) use 512-wide heads with two
# KV heads; the sliding layers use 256-wide heads with eight.
SLIDING_KV = (8, 256)
FULL_KV = (2, 512)
LAYER_KV = tuple(FULL_KV if (index + 1) % 6 == 0 else SLIDING_KV
                 for index in range(30))

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
    w_mem_bytes=27.5e9,    # estimate: fp8 linears and experts (24.5e9)
    #                        + bf16 embeddings (1.5e9) + bf16 vision
    #                        tower, router, and self-conditioning
    #                        weights; the confirmation cell prints the
    #                        loaded footprint
    vocab=262_144,
    tied_head=True,
    weight_precision="fp8",
    arch="diffusion_gemma",
    layer_kv=LAYER_KV,
    canvas_tokens=256,     # the checkpoint's canvas_length
    turn_prefix="<bos><|turn>user\n",
    # The empty thinking channel keeps the answer at the first canvas
    # row: with thinking off, the model may still open one.
    turn_suffix="<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    attn_params_per_layer=(25 * SLIDING_ATTN + 5 * FULL_ATTN) // 30,
    mlp_active_params_per_layer=DENSE_MLP + 8 * EXPERT,
    mlp_total_params_per_layer=DENSE_MLP + 128 * EXPERT,
    chunk_cap_tokens=65_536,    # the MoE kernels' token workspace and
    #                             the 512-wide attention heads grow
    #                             with the chunk; the kernel index cap
    #                             alone would allow 209k
)
