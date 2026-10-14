from .base import ModelSpec

# Every fourth layer is attention (layers 3, 7, ..., 63); the other 48
# are Gated DeltaNet.
_LAYER_KINDS = ("gated_delta_net",) * 3 + ("full_attention",)

QWEN3_8_27B_FP8 = ModelSpec(
    name="qwen3.8-27b-fp8",
    params=24_352_522_240,  # attention, Gated DeltaNet, and MLP
    #                         projections plus the convolution weights;
    #                         norms and the embedding excluded; the
    #                         same count as Qwen3.5-27B-FP8
    layers=64,
    hidden=5120,
    n_q=24,
    n_kv=4,
    d_head=256,
    ffn_width=34816,       # gate_up output: 2 x 17408 intermediate
    w_bytes=1.0,           # fp8 weights, block scales in bf16
    hf_name="Qwen/Qwen3.8-27B-FP8",
    revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    kv_bytes=2.0,
    w_mem_bytes=29_638_887_936,    # vLLM's model memory after loading
    #                                the text path, 0.6 percent above
    #                                the 29,468,003,328 bytes of the
    #                                checkpoint's text tensors;
    #                                /results/ablations/
    #                                qwen3.8-27b-fp8-vllm-limits-768.json
    vocab=248_320,
    tied_head=False,
    weight_precision="fp8",
    arch="qwen3_5",
    layer_kinds=_LAYER_KINDS * 16,
    max_num_seqs=768,      # 640 and 768 started; 896 raised
    #                        ValueError, 801 Mamba cache blocks held
    #                        at 0.91 memory; the records are
    #                        /results/ablations/
    #                        qwen3.8-27b-fp8-vllm-limits-<setting>.json
    vision_tower=True,
    attn_output_gate=True,
    gdn_key_heads=16,
    gdn_value_heads=48,
    gdn_key_dim=128,
    gdn_value_dim=128,
    conv_width=4,
    turn_prefix="<|im_start|>user\n",
    turn_suffix="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    prompt_format="qwen3_5-chat-nonthinking-v1",
)
