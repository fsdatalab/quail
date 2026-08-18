from .base import ModelSpec

QWEN3_4B_FP8 = ModelSpec(
    name="qwen3-4b-fp8",
    params=3.6e9,          # dense params repeatedly used per token
    layers=36,
    hidden=2560,
    n_q=32,
    n_kv=8,
    d_head=128,
    ffn_width=19456,       # gate_up output: 2 x 9728 intermediate
    w_bytes=1.0,           # fp8 weights
    hf_name="Qwen/Qwen3-4B-FP8",
    kv_bytes=2.0,          # bf16 KV by default; the planner may pick fp8
    w_mem_bytes=4.5e9,     # measured footprint: fp8 weights + bf16
    #                        embeddings + block scales
)
