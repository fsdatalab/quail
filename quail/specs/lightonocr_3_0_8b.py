from .base import ModelSpec

# One Gated DeltaNet layer's projections: in_proj_qkvz (hidden to 16
# key heads, 16 query heads, 16 value heads, and 16 gate heads of 128),
# in_proj_ba, out_proj, and the short convolution.
LINEAR_ATTENTION_PARAMS = (1024 * (2 * 16 * 128 + 2 * 16 * 128)
                           + 1024 * 2 * 16
                           + 16 * 128 * 1024
                           + (2 * 16 * 128 + 16 * 128) * 4)

LIGHTONOCR_3_0_8B_BF16 = ModelSpec(
    name="lightonocr-3-0.8b",
    params=0.497e9,        # dense params repeatedly used per token:
    #                        18 linear-attention layers, 6 gated
    #                        full-attention layers, and 24 MLPs; the
    #                        tied embedding and the vision tower are
    #                        outside it
    layers=24,
    hidden=1024,
    # The 6 full-attention layers: gated attention whose q_proj also
    # emits a sigmoid gate. Rotary covers a quarter of each head, in
    # three interleaved sections for an image's time, row, and column.
    n_q=8,
    n_kv=2,
    d_head=256,
    full_attention_period=4,
    linear_attention_params=LINEAR_ATTENTION_PARAMS,
    linear_key_heads=16,
    linear_value_heads=16,
    linear_head_dim=128,
    linear_conv_kernel=4,
    ffn_width=7168,        # gate_up output: 2 x 3584 intermediate
    w_bytes=2.0,           # bf16 weights
    hf_name="lightonai/LightOnOCR-3-0.8B",
    revision="36d8636925327c671bd8e479c517a004f7566bd9",
    kv_bytes=2.0,
    # 852,985,920 bf16 parameters in the safetensors, vision tower and
    # tied embedding included
    w_mem_bytes=1.71e9,
    vocab=248_320,
    tied_head=True,
    weight_precision="bf16",
    # vLLM's model type, which names the Quail executor's pipeline
    arch="qwen3_5",
)
