from .base import ModelSpec

# Every fourth layer is attention (layers 3, 7, ..., 31); the other 24
# are Gated DeltaNet.
_LAYER_KINDS = ("gated_delta_net",) * 3 + ("full_attention",)

QWEN3_5_4B_BF16 = ModelSpec(
    name="qwen3.5-4b-bf16",
    params=3_569_876_992,  # attention, Gated DeltaNet, and MLP
    #                        projections plus the convolution weights;
    #                        norms and the embedding excluded
    layers=32,
    hidden=2560,
    n_q=16,
    n_kv=4,
    d_head=256,
    ffn_width=18432,       # gate_up output: 2 x 9216 intermediate
    w_bytes=2.0,           # bf16 weights
    hf_name="Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    kv_bytes=2.0,
    w_mem_bytes=8_411_152_384,    # derived from the config as 2 bytes
    #                               x (projections + convolution +
    #                               embedding); not yet measured as loaded
    vocab=248_320,
    tied_head=True,
    weight_precision="bf16",
    arch="qwen3_5",
    layer_kinds=_LAYER_KINDS * 8,
    max_num_seqs=3_000,    # vLLM 0.26 on an H100 at 0.91 memory holds 3,052
    #                        state blocks; 4,096 sequences raised ValueError
    vision_tower=True,
    attn_output_gate=True,
    gdn_key_heads=16,
    gdn_value_heads=32,
    gdn_key_dim=128,
    gdn_value_dim=128,
    conv_width=4,
)
