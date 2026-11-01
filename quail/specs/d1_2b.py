from .base import ModelSpec

# Every other layer is a short convolution; the attention layers
# alternate full (2, 6, ..., 22) and sliding (4, 8, ..., 20).
_LAYER_KINDS = (("short_conv", "short_conv", "full_attention")
                + ("short_conv", "sliding_attention",
                   "short_conv", "full_attention") * 5
                + ("short_conv",))

# calibration.json "tables" "found:text": a text read's raw P(found)
# against the share found on held-out rows
_FOUND_TEXT = (
    (0.005682162615729877, 0.0),
    (0.020574209407310595, 0.011363636363636364),
    (0.05274184431288053, 0.022727272727272728),
    (0.12593568082560191, 0.11363636363636363),
    (0.2609023091129281, 0.25),
    (0.45711776140061294, 0.42045454545454547),
    (0.6460263153368776, 0.5568181818181818),
    (0.7728411636569283, 0.7727272727272727),
    (0.852111120115627, 0.8522727272727273),
    (0.9055508368394591, 0.9204545454545454),
    (0.9416752389886163, 0.9318181818181818),
    (0.9675597308034246, 0.9772727272727273),
    (0.9927885918454691, 0.9924242424242424),
    (0.9997205003886156, 1.0),
)

D1_2B_BF16 = ModelSpec(
    name="d1-2b-bf16",
    params=1_466_456_064,  # attention, convolution-block, and MLP
    #                        projections plus the convolution weights;
    #                        norms and the embedding excluded
    layers=24,
    hidden=2048,
    num_q_heads=16,
    num_kv_heads=4,
    d_head=128,
    ffn_width=15360,       # gate_up output: 2 x 7680 intermediate
    w_bytes=2.0,           # bf16 weights
    hf_name="LiquidAI/d1-2B",
    revision="10e7671d44e4b37700020b1eed3f594c5f2af0b4",
    kv_bytes=2.0,
    w_mem_bytes=3_457_603_072,    # the checkpoint's language-model
    #                               tensors, which Quail's LFM3 loader
    #                               loads and nothing else
    vocab=128_000,
    tied_head=True,
    weight_precision="bf16",
    arch="lfm3",
    sliding_window=256,
    layer_kinds=_LAYER_KINDS,
    conv_width=7,
    # Liquid trained the model with set attention (d1_attention "set")
    attention_mask="set",
    set_size=16,           # d1_set_size in the checkpoint's config
    # the model was trained to answer yes or no at the end of its turn
    answer_spellings=(("yes", "Yes", "YES"), ("no", "No", "NO")),
    # calibration.json "default", the temperature Liquid fitted for
    # option answers
    answer_temperature=1.337398453966105,
    read_found=_FOUND_TEXT,
    read_span_tokens=512,  # pointer.py SPAN_CAPS["text"]
    turn_prefix="<|startoftext|><|im_start|>user\n",
    bos_text="<|startoftext|>",
    # the chat template's assistant turn with thinking off
    turn_suffix="<|im_end|>\n<|im_start|>assistant\n<think></think>",
    prompt_layout="d1",
    prompt_format="d1-v1",
)
