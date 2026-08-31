from .base import ModelSpec

QWEN3_32B_FP8 = ModelSpec(
    name="qwen3-32b-fp8",
    params=31.2e9,          # dense params repeatedly used per token
    #                         (Qwen3-32B non-embedding param count)
    layers=64,
    hidden=5120,
    n_q=64,
    n_kv=8,
    d_head=128,
    ffn_width=51200,       # gate_up output: 2 x 25600 intermediate
    w_bytes=1.0,           # fp8 weights
    hf_name="Qwen/Qwen3-32B-FP8",
    revision="aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df",
    w_mem_bytes=34.37e9,   # measured as-loaded footprint: fp8 weights
    #                        + two bf16 vocab matrices (embedding and
    #                        untied lm_head) + block scales. The head
    #                        (head_mem_bytes, 1.56e9) moves to CPU
    #                        memory at load, so budgets subtract it.
    vocab=151_936,
    tied_head=False,       # separate lm_head; moved to CPU at load
)
