from .base import ModelSpec

# One Gated DeltaNet layer's projections: in_proj_qkvz (hidden to 16
# key heads, 16 query heads, 48 value heads, and 48 gate heads of 128),
# in_proj_ba, out_proj, and the short convolution.
LINEAR_ATTENTION_PARAMS = (5120 * (2 * 16 * 128 + 2 * 48 * 128)
                           + 5120 * 2 * 48
                           + 48 * 128 * 5120
                           + (2 * 16 * 128 + 48 * 128) * 4)

QWEN3_8_27B_FP8 = ModelSpec(
    name="qwen3.8-27b-fp8",
    params=24.35e9,        # dense params repeatedly used per token:
    #                        48 linear-attention layers, 16 gated
    #                        full-attention layers, and 64 MLPs
    layers=64,
    hidden=5120,
    # The 16 full-attention layers: gated attention whose q_proj also
    # emits a sigmoid gate, so its output is 2 x n_q x d_head columns;
    # the cost model's projection count leaves the gate out, about
    # 0.5e9 of the params. Rotary covers a quarter of each head.
    n_q=24,
    n_kv=4,
    d_head=256,
    full_attention_period=4,
    linear_attention_params=LINEAR_ATTENTION_PARAMS,
    ffn_width=34816,       # gate_up output: 2 x 17408 intermediate
    w_bytes=1.0,           # fp8 weights in [128, 128] blocks
    hf_name="Qwen/Qwen3.8-27B-FP8",
    revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    kv_bytes=2.0,
    # Estimated from the checkpoint's 30.9e9 bytes of safetensors
    # without the 0.7e9-byte multi-token-prediction layer, which vLLM
    # loads only for speculative decoding, and the 0.8e9-byte vision
    # tower, which it skips in language_model_only mode: fp8
    # projections and the bf16 embedding table and untied head.
    # Replace with the measured footprint.
    w_mem_bytes=29.4e9,
    vocab=248_320,
    tied_head=False,
    # vLLM's model type; Quail's executor has no forward pass for it,
    # so the Quail backend refuses this model and the stock vLLM
    # backend runs it
    arch="qwen3_5",
    language_model_only=True,
)
