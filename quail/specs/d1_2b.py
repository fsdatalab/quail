from .base import ModelSpec

# Every other layer is a short convolution; the attention layers
# alternate full (2, 6, ..., 22) and sliding (4, 8, ..., 20).
_LAYER_KINDS = (("short_conv", "short_conv", "full_attention")
                + ("short_conv", "sliding_attention",
                   "short_conv", "full_attention") * 5
                + ("short_conv",))

D1_2B_BF16 = ModelSpec(
    name="d1-2b-bf16",
    params=1_466_456_064,  # attention, convolution-block, and MLP
    #                        projections plus the convolution weights;
    #                        norms and the embedding excluded
    layers=24,
    hidden=2048,
    num_q_heads=16,
    num_kv_heads=4,
    d_head=128,
    ffn_width=15360,       # gate_up output: 2 x 7680 intermediate
    w_bytes=2.0,           # bf16 weights
    hf_name="LiquidAI/d1-2B",
    revision="10e7671d44e4b37700020b1eed3f594c5f2af0b4",
    kv_bytes=2.0,
    w_mem_bytes=3_457_603_072,    # the checkpoint's language-model
    #                               tensors, which Quail's LFM3 loader
    #                               loads and nothing else
    vocab=128_000,
    tied_head=True,
    weight_precision="bf16",
    arch="lfm3",
    sliding_window=256,
    layer_kinds=_LAYER_KINDS,
    conv_width=7,
    # Liquid trained the model with set attention (d1_attention "set")
    attention_mask="set",
    # the model was trained to answer yes or no at the end of its turn
    answer_spellings=(("yes", "Yes", "YES"), ("no", "No", "NO")),
    # calibration.json "default", the temperature Liquid fitted for
    # option answers
    answer_temperature=1.337398453966105,
    turn_prefix="<|startoftext|><|im_start|>user\n",
    bos_text="<|startoftext|>",
    # the chat template's assistant turn with thinking off
    turn_suffix="<|im_end|>\n<|im_start|>assistant\n<think></think>",
    prompt_layout="d1",
    prompt_format="d1-v1",
)
