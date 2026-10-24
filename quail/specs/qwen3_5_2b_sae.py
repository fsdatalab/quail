from .base import ModelSpec

# Every fourth layer is attention; the others are Gated DeltaNet. The
# checkpoint has 24 layers; this spec loads layers 0 to 11 (9 Gated
# DeltaNet and 3 attention layers) and reads the residual stream after
# layer 11, where the Qwen-Scope SAE was trained (hook point
# resid_post). The spec serves AI.SAE_FEATURES only.
_LAYER_KINDS = ("gated_delta_net",) * 3 + ("full_attention",)

QWEN3_5_2B_SAE_BF16 = ModelSpec(
    name="qwen3.5-2b-sae-bf16",
    params=686_579_712,    # attention, Gated DeltaNet, and MLP
    #                        projections plus the convolution weights of
    #                        layers 0 to 11; norms and the embedding
    #                        excluded
    layers=12,
    checkpoint_layers=24,
    hidden=2048,
    num_q_heads=8,
    num_kv_heads=2,
    d_head=256,
    ffn_width=12288,       # gate_up output: 2 x 6144 intermediate
    w_bytes=2.0,           # bf16 weights
    hf_name="Qwen/Qwen3.5-2B-Base",
    revision="b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
    kv_bytes=2.0,
    w_mem_bytes=4_040_000_000,    # measured: torch.cuda.memory_allocated
    #                               after load_model in
    #                               experiments/cells/sae_agreement.py
    #                               (3,897,935,872) plus the SAE encoder
    #                               (138,682,368). Near the full 24-layer
    #                               size: the dropped tail layers appear
    #                               to stay referenced, so budgets assume
    #                               they are resident
    vocab=248_320,
    tied_head=True,
    weight_precision="bf16",
    arch="qwen3_5",
    layer_kinds=_LAYER_KINDS * 3,
    max_num_seqs=3_000,    # the qwen3.5-4b engine bound; the 2B state is
    #                        smaller, so this does not overcommit
    vision_tower=True,
    attn_output_gate=True,
    gdn_key_heads=16,
    gdn_value_heads=16,
    gdn_key_dim=128,
    gdn_value_dim=128,
    conv_width=4,
    # a base checkpoint: prompts are raw text with no chat turn, as the
    # quail-bench feature questions are rendered
    residual_readout=True,
    sae_hf_name="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100",
    sae_revision="027267657257a8d490296286e8fab41e1c1a1a3d",
    sae_file="layer11.sae.pt",
    sae_num_features=32_768,
    sae_keep=100,
)
