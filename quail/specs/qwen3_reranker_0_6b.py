from .base import ModelSpec

QWEN3_RERANKER_0_6B_BF16 = ModelSpec(
    name="qwen3-reranker-0.6b-bf16",
    params=595_776_512,
    layers=28,
    hidden=1024,
    n_q=16,
    n_kv=8,
    d_head=128,
    ffn_width=6144,
    w_bytes=2.0,
    hf_name="Qwen/Qwen3-Reranker-0.6B",
    revision="e61197ed45024b0ed8a2d74b80b4d909f1255473",
    kv_bytes=2.0,
    w_mem_bytes=1_203_010_934,
    vocab=151_669,
    tied_head=True,
    weight_precision="bf16",
    attention_precision="bf16",
    role="reranker",
)
