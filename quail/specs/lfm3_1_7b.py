from .base import ModelSpec
from .prompts import TEXT_ANALYSIS_SYSTEM_PROMPT

_LAYER_TYPES = ("conv", "conv", "full_attention", "conv", "sliding_attention",
                "conv") + ("full_attention", "conv", "sliding_attention",
                           "conv") * 4 + ("full_attention", "conv")
_KINDS = {"conv": "short_conv", "full_attention": "full_attention",
          "sliding_attention": "sliding_attention"}

LFM3_1_7B_BF16 = ModelSpec(
    name="lfm3-1.7b-bf16",
    params=1_466_456_064,  # attention (with the per-head gate),
    #                        convolution-block, and MLP projections plus
    #                        the convolution weights; norms and the
    #                        embedding excluded
    layers=24,
    hidden=2048,
    num_q_heads=16,
    num_kv_heads=4,
    d_head=128,
    ffn_width=15360,       # w1 and w3 outputs: 2 x 7680 intermediate
    w_bytes=2.0,
    hf_name="LiquidAI/LFM3-1.7B",
    revision="40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
    kv_bytes=2.0,
    w_mem_bytes=3_457_603_072,    # 2 x the hub's parameter count; not
    #                               measured, vLLM cannot load LFM3
    vocab=128_000,
    tied_head=True,
    weight_precision="bf16",
    arch="lfm2",           # placeholder: no LFM3 forward pass exists
    conv_width=7,
    sliding_window=256,
    layer_kinds=tuple(_KINDS[kind] for kind in _LAYER_TYPES),
    turn_prefix=("<|startoftext|><|im_start|>system\n"
                 f"{TEXT_ANALYSIS_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"),
    bos_text="<|startoftext|>",
    turn_suffix="<|im_end|>\n<|im_start|>assistant\n<think></think>",
    prompt_format="lfm3-chat-text-analysis-v1",
)
