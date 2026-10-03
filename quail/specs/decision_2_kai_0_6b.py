from .base import ModelSpec

DECISION_2_KAI_0_6B_BF16 = ModelSpec(
    name="decision-2.0-kai-0.6b-bf16",
    params=440_467_456,    # Qwen3-0.6B projections and norms; the
    #                        embedding and the decision head excluded
    layers=28,
    hidden=1024,
    n_q=16,
    n_kv=8,
    d_head=128,
    ffn_width=6144,        # gate_up output: 2 x 3072 intermediate
    w_bytes=2.0,           # bf16 weights after conversion
    hf_name="vllm-sr/Decision-2.0-Kai-0.6B",
    revision="881bee413681d80ebeac86afcda8b4138dae516e",
    kv_bytes=2.0,
    w_mem_bytes=1_196_313_424,    # 596,049,920 bf16 backbone weights,
    #                               embedding included, plus the fp32
    #                               decision head file; not measured
    vocab=151_936,
    tied_head=True,
    weight_precision="bf16",
    attention_precision="bf16",
    role="decision",
    prompt_layout="decision2-noul",
    prompt_format="decision2-noul-v1",
    # the same shapes as the 0.6B reranker, whose warm-up OOMed at its
    # memory-bound chunk of 349k tokens on an H100
    chunk_cap_tokens=65_536,
)
