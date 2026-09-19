from dataclasses import replace

from .base import ModelSpec

DIFFUSION_GEMMA_26B_FP8 = ModelSpec(
    name="diffusion-gemma-26b-a4b-fp8",
    params=3.09e9,         # params one token multiplies, embeddings
    #                        aside: 25 sliding + 5 full attention
    #                        layers, each with the dense MLP and eight
    #                        active experts
    layers=30,
    hidden=2816,
    # Attention runs with softmax scale 1.0. 25 layers slide over a
    # 1024-token window with 256-wide heads; every sixth layer keeps
    # every token with 512-wide heads and two KV heads.
    n_q=16,
    n_kv=8,
    d_head=256,
    full_attention_period=6,
    full_n_kv=2,
    full_d_head=512,
    sliding_window=1024,
    # a dense MLP 2112 wide beside 128 experts 704 wide, eight active
    ffn_width=4224,
    experts=128,
    experts_active=8,
    expert_intermediate=704,
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
    prompt_format="gemma4-chat-nonthinking-v1",
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
