"""Hybrid model specs against their config.json files."""

import json
from pathlib import Path

from quail.backends.vllm import DefaultVLLMEngine, VLLMEngine
from quail.cost.dense_decoder_cost import (
    attention_projection_params,
    gdn_projection_params,
    mlp_params,
)
from quail.specs import (
    LFM2_2_6B_BF16,
    MODELS,
    QWEN3_5_27B_FP8,
    QWEN3_8_27B_FP8,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _config(path):
    return json.loads((FIXTURES / path).read_text())


def test_qwen3_5_27b_matches_the_config():
    spec = QWEN3_5_27B_FP8
    config = _config("qwen3_5_configs/qwen3.5-27b-fp8.json")["text_config"]
    assert spec.layers == config["num_hidden_layers"]
    assert spec.hidden == config["hidden_size"]
    assert spec.n_q == config["num_attention_heads"]
    assert spec.n_kv == config["num_key_value_heads"]
    assert spec.d_head == config["head_dim"]
    assert spec.ffn_width == 2 * config["intermediate_size"]
    assert spec.vocab == config["vocab_size"]
    assert spec.gdn_key_heads == config["linear_num_key_heads"]
    assert spec.gdn_value_heads == config["linear_num_value_heads"]
    assert spec.gdn_key_dim == config["linear_key_head_dim"]
    assert spec.gdn_value_dim == config["linear_value_head_dim"]
    assert spec.conv_width == config["linear_conv_kernel_dim"]
    assert [spec.layer_kind(i) for i in range(spec.layers)] == [
        {"linear_attention": "gated_delta_net"}.get(kind, kind)
        for kind in config["layer_types"]]
    assert MODELS[spec.name] is spec


def test_qwen3_5_27b_weights_follow_the_checkpoint():
    spec = QWEN3_5_27B_FP8
    # 48 Gated DeltaNet layers of 10,240 convolution channels, width 4
    conv_weights = 48 * 10_240 * 4
    assert spec.params == (attention_projection_params(spec)
                           + gdn_projection_params(spec)
                           + mlp_params(spec) + conv_weights)
    # fp8 projections, plus bf16 embedding, lm_head, convolution, and
    # the two per-head gates; the headers also hold 2.2e6 norm elements
    in_proj_ba = 48 * spec.hidden * 2 * spec.gdn_value_heads
    in_bf16 = (2 * spec.vocab * spec.hidden + conv_weights + in_proj_ba)
    derived = spec.params - conv_weights - in_proj_ba + 2 * in_bf16
    assert 0 < 29_468_020_224 - derived < 5_000_000
    # vLLM's reported memory after loading, 0.6 percent above the tensors
    assert spec.w_mem_bytes == 29_638_887_936
    assert spec.w_mem_bytes / 29_468_020_224 < 1.01
    assert spec.head_mem_bytes == 2 * spec.vocab * spec.hidden
    assert len(spec.attention_layers) == 16
    # 16 layers x 2 x 4 KV heads x 256 x 2 bytes
    assert spec.kappa == 65_536


def test_qwen3_5_27b_is_text_only_for_stock_vllm():
    spec = QWEN3_5_27B_FP8
    for engine in (VLLMEngine(), DefaultVLLMEngine()):
        assert engine.llm_kwargs(spec)["limit_mm_per_prompt"] == {
            "image": 0, "video": 0}
    assert VLLMEngine().llm_kwargs(spec)["max_num_seqs"] == 768


def test_qwen3_8_27b_matches_the_config():
    spec = QWEN3_8_27B_FP8
    config = _config("qwen3_5_configs/qwen3.8-27b-fp8.json")["text_config"]
    assert spec.layers == config["num_hidden_layers"]
    assert spec.hidden == config["hidden_size"]
    assert spec.n_q == config["num_attention_heads"]
    assert spec.n_kv == config["num_key_value_heads"]
    assert spec.d_head == config["head_dim"]
    assert spec.ffn_width == 2 * config["intermediate_size"]
    assert spec.vocab == config["vocab_size"]
    assert spec.gdn_key_heads == config["linear_num_key_heads"]
    assert spec.gdn_value_heads == config["linear_num_value_heads"]
    assert spec.gdn_key_dim == config["linear_key_head_dim"]
    assert spec.gdn_value_dim == config["linear_value_head_dim"]
    assert spec.conv_width == config["linear_conv_kernel_dim"]
    assert [spec.layer_kind(i) for i in range(spec.layers)] == [
        {"linear_attention": "gated_delta_net"}.get(kind, kind)
        for kind in config["layer_types"]]
    assert MODELS[spec.name] is spec


def test_qwen3_8_27b_has_the_geometry_of_qwen3_5_27b():
    # the two checkpoints share every size the cost model reads
    new, old = QWEN3_8_27B_FP8, QWEN3_5_27B_FP8
    for field in ("params", "layers", "hidden", "n_q", "n_kv", "d_head",
                  "ffn_width", "vocab", "layer_kinds", "kappa",
                  "turn_prefix", "turn_suffix"):
        assert getattr(new, field) == getattr(old, field)
    # 24,328,929,280 fp8 and convolution elements in the headers, plus
    # 48 layers x 2 x 48 x 5,120 bf16 in_proj_a and in_proj_b elements
    assert new.params == 24_328_929_280 + 48 * 2 * 48 * 5120
    # vLLM's reported memory after loading, 0.6 percent above the
    # 29,468,003,328 bytes of text tensors
    assert new.w_mem_bytes == 29_638_887_936
    assert new.w_mem_bytes / 29_468_003_328 < 1.01
    assert new.max_num_seqs == 768


def test_lfm2_matches_the_config():
    spec = LFM2_2_6B_BF16
    config = _config("lfm2_configs/lfm2-2.6b.json")
    assert spec.layers == config["num_hidden_layers"]
    assert spec.hidden == config["hidden_size"]
    assert spec.n_q == config["num_attention_heads"]
    assert spec.n_kv == config["num_key_value_heads"]
    assert spec.d_head == config["hidden_size"] // config["num_attention_heads"]
    assert spec.ffn_width == 2 * config["intermediate_size"]
    assert spec.vocab == config["vocab_size"]
    assert spec.tied_head == config["tie_embedding"]
    assert [spec.layer_kind(i) for i in range(spec.layers)] == [
        {"conv": "short_conv"}.get(kind, kind)
        for kind in config["layer_types"]]
    assert MODELS[spec.name] is spec


def test_lfm2_parameters_follow_the_config():
    spec = LFM2_2_6B_BF16
    conv_layers = spec.layers - len(spec.attention_layers)
    # input projection to 3 x hidden, output projection, and a depthwise
    # convolution of width conv_L_cache
    conv_block = conv_layers * (4 * spec.hidden ** 2 + spec.hidden * 3)
    assert spec.params == (attention_projection_params(spec)
                           + mlp_params(spec) + conv_block)
    # hub count 2,569,272,320: embedding plus the norms on top
    assert spec.w_mem_bytes == 2 * 2_569_272_320
    assert len(spec.attention_layers) == 8
    # 8 layers x 2 x 8 KV heads x 64 x 2 bytes
    assert spec.kappa == 16_384


def test_lfm2_keeps_only_convolution_columns():
    spec = LFM2_2_6B_BF16
    assert len(spec.short_conv_layers) == 22
    assert len(spec.state_layers) == 30 - 8
    assert spec.gdn_layers == ()
    assert spec.conv_state_channels == spec.hidden == 2048
    # two bf16 columns of 2,048 channels in each of 22 layers
    assert spec.checkpoint_bytes == 22 * 2048 * 2 * 2


def test_lfm2_prices_its_short_convolution_projections():
    from quail.cost import budgets
    from quail.cost.dense_decoder_cost import (
        dense_params,
        short_conv_projection_params,
    )

    spec = LFM2_2_6B_BF16
    assert short_conv_projection_params(spec) == 22 * 4 * 2048 * 2048
    assert dense_params(spec) > short_conv_projection_params(spec)
    shapes = budgets._projection_shapes(spec)
    assert (2048, 3 * 2048, 22) in shapes and (2048, 2048, 22) in shapes
    assert not budgets.tree_attention_allowed(spec)


def test_a_model_without_short_convolutions_prices_none():
    from quail.cost.dense_decoder_cost import short_conv_projection_params

    assert short_conv_projection_params(QWEN3_5_27B_FP8) == 0


def test_lfm2_work_includes_its_convolution_projections():
    from quail.cost.dense_decoder_cost import (
        attention_projection_params,
        dense_decoder_components,
        short_conv_projection_params,
    )
    from quail.cost.work import Work

    spec = LFM2_2_6B_BF16
    work = Work(tokens=1_000)
    projection = next(
        c for c in dense_decoder_components(work, spec, passes=1.0)
        if c.name == "attn_proj")
    expected = (attention_projection_params(spec)
                + short_conv_projection_params(spec))
    assert projection.flops == 2.0 * expected * 1_000


def test_lfm2_prices_its_short_convolutions_on_the_cuda_cores():
    from quail.cost.dense_decoder_cost import (
        conv_flops_per_token,
        dense_decoder_components,
    )
    from quail.cost.work import Work

    spec = LFM2_2_6B_BF16
    conv = next(c for c in dense_decoder_components(
        Work(tokens=1_000), spec, passes=1.0) if c.name == "conv")
    # 2 FLOPs per tap per channel: width 3, 2,048 channels, 22 layers
    assert conv_flops_per_token(spec) == 2 * 3 * 2_048 * 22
    assert conv.flops == conv_flops_per_token(spec) * 1_000
    assert conv.precision == "fp32"


def test_lfm2_prompt_opens_with_the_beginning_of_text_token():
    from quail.logical.prompts import shared_preamble, without_bos

    spec = LFM2_2_6B_BF16
    assert spec.bos_text == "<|startoftext|>"
    # Quail tokenizes without special tokens, so its prefix carries the token
    assert shared_preamble(spec.turn_prefix).startswith(spec.bos_text)
    # vLLM adds the token itself, so the text it receives must not repeat it
    text = without_bos(shared_preamble(spec.turn_prefix), spec.bos_text)
    assert text == "<|im_start|>user\n" + shared_preamble("")
    assert not text.startswith(spec.bos_text)


def test_models_without_a_beginning_of_text_token_keep_their_prefix():
    from quail.logical.prompts import without_bos
    from quail.specs import QWEN3_4B_FP8, QWEN3_5_4B_BF16

    for spec in (QWEN3_4B_FP8, QWEN3_5_4B_BF16):
        assert spec.bos_text == ""
        assert without_bos(spec.turn_prefix, spec.bos_text) == spec.turn_prefix
