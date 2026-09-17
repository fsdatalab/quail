from .base import ModelSpec

QWEN3_RERANKER_4B_BF16 = ModelSpec(
    name="qwen3-reranker-4b-bf16",
    params=4_021_784_576,
    layers=36,
    hidden=2560,
    n_q=32,
    n_kv=8,
    d_head=128,
    ffn_width=19_456,
    w_bytes=2.0,
    hf_name="Qwen/Qwen3-Reranker-4B",
    revision="22e683669bc0f0bd69640a1354a6d0aebcfeede5",
    kv_bytes=2.0,
    w_mem_bytes=8_055_037_614,
    vocab=151_669,
    tied_head=True,
    weight_precision="bf16",
    attention_precision="bf16",
)
