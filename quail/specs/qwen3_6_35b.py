from .base import ModelSpec
from .prompts import QWEN3_PROMPT_FORMAT, QWEN3_TURN_PREFIX, QWEN3_TURN_SUFFIX

_LAYER_KINDS = ("gated_delta_net",) * 3 + ("full_attention",)

QWEN3_6_35B_A3B_FP8 = ModelSpec(
    name="qwen3.6-35b-a3b-fp8",
    params=2_416_640_000,
    layers=40,
    hidden=2048,
    n_q=16,
    n_kv=2,
    d_head=256,
    ffn_width=1024,
    experts=256,
    experts_active=8,
    expert_intermediate=512,
    w_bytes=1.0,
    hf_name="Qwen/Qwen3.6-35B-A3B-FP8",
    revision="95a723d08a9490559dae23d0cff1d9466213d989",
    kv_bytes=2.0,
    # Loaded weights before KV or expert workspace allocation.
    # /results/ablations/qwen3.6-35b-a3b-fp8-quail-compare-quail.json
    w_mem_bytes=35_879_978_496,
    vocab=248_320,
    tied_head=False,
    weight_precision="fp8",
    arch="qwen3_5",
    layer_kinds=_LAYER_KINDS * 10,
    max_num_seqs=512,
    chunk_cap_tokens=32_768,
    moe_backend="triton",
    vision_tower=True,
    attn_output_gate=True,
    gdn_key_heads=16,
    gdn_value_heads=32,
    gdn_key_dim=128,
    gdn_value_dim=128,
    conv_width=4,
    turn_prefix=QWEN3_TURN_PREFIX,
    turn_suffix=QWEN3_TURN_SUFFIX,
    prompt_format=QWEN3_PROMPT_FORMAT,
)
