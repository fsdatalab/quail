from .base import ModelSpec

# Eight attention layers (2, 5, 9, 13, 17, 21, 24, 27); the other 22 are
# short convolutions.
_ATTENTION_LAYERS = (2, 5, 9, 13, 17, 21, 24, 27)

LFM2_2_6B_BF16 = ModelSpec(
    name="lfm2-2.6b-bf16",
    params=2_434_928_640,  # attention, convolution-block, and MLP
    #                        projections plus the convolution weights;
    #                        norms and the embedding excluded
    layers=30,
    hidden=2048,
    n_q=32,
    n_kv=8,
    d_head=64,
    ffn_width=21504,       # gate_up output: 2 x 10752 intermediate
    w_bytes=2.0,           # bf16 weights
    hf_name="LiquidAI/LFM2-2.6B",
    revision="36ed799f7024ef169ba92ffe184821835e15cc66",
    kv_bytes=2.0,
    w_mem_bytes=5_138_544_640,    # 2 bytes x the hub's parameter count,
    #                               which includes norms and the tied
    #                               embedding; not yet measured as loaded
    vocab=65_536,
    tied_head=True,
    weight_precision="bf16",
    arch="lfm2",
    conv_width=3,          # L_cache: a short convolution reads 2 columns back
    layer_kinds=tuple(
        "full_attention" if layer in _ATTENTION_LAYERS else "short_conv"
        for layer in range(30)),
    turn_prefix="<|im_start|>user\n",
    turn_suffix="<|im_end|>\n<|im_start|>assistant\n",
    prompt_format="lfm2-chat-v1",
)
