from .base import ModelSpec

DIFFUSION_GEMMA_26B_FP8 = ModelSpec(
    name="diffusion-gemma-26b-a4b-fp8",
    # Active parameters per token, excluding embeddings.
    params=3.09e9,
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
    # Loaded FP8 modules plus BF16 embeddings, router, and auxiliary weights.
    w_mem_bytes=27_682_404_352,
    vocab=262_144,
    tied_head=True,
    weight_precision="fp8",
    arch="diffusion_gemma",
    # Prefill the empty thinking channel so the next position holds the answer.
    canvas_tokens=1,
    turn_prefix="<bos><|turn>user\n",
    turn_suffix="<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    prompt_format="gemma4-chat-nonthinking-v1",
    moe_backend="triton",
    # Bound expert workspace and attention buffers for 512-wide heads.
    chunk_cap_tokens=65_536,
    # Images go through the Gemma 4 vision tower: 16 px patches, 3 x 3
    # pooled into one soft token, at most one of these budgets per
    # image, wrapped in <start_of_image> and <end_of_image>. The
    # checkpoint's own processor confirmed the budgets and the soft
    # token arithmetic (/results/ablations/pdf_probe_vision.json).
    input_modalities=frozenset({"text", "image"}),
    image_token_budgets=(70, 140, 280, 560, 1120),
    default_image_tokens=280,
    image_patch_pixels=16,
    image_pool_kernel=3,
    image_frame_tokens=2,
)
