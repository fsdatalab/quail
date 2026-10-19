"""Dense and MoE Qwen hybrid specs against checkpoint configs and costs."""

import json
from pathlib import Path

import pytest

from quail.cost.dense_decoder_cost import (
    attention_projection_params,
    dense_decoder_components,
    gdn_projection_params,
    mlp_params,
)
from quail.cost.work import Work
from quail.specs import (
    H100_SXM,
    MODELS,
    QWEN3_4B_FP8,
    QWEN3_5_4B_BF16,
    QWEN3_5_4B_FP8,
)

CONFIG = (Path(__file__).parents[1] / "fixtures" / "qwen3_5_configs"
          / "qwen3.5-4b-bf16.json")
SPEC = QWEN3_5_4B_BF16


@pytest.mark.parametrize("name", ["qwen3.5-4b-bf16", "qwen3.6-35b-a3b-fp8"])
def test_spec_matches_the_config(name):
    spec = MODELS[name]
    config = json.loads((CONFIG.parent / f"{name}.json").read_text())["text_config"]
    assert spec.layers == config["num_hidden_layers"]
    assert spec.hidden == config["hidden_size"]
    assert spec.num_q_heads == config["num_attention_heads"]
    assert spec.num_kv_heads == config["num_key_value_heads"]
    assert spec.d_head == config["head_dim"]
    intermediate = config.get("intermediate_size",
                              config.get("shared_expert_intermediate_size"))
    assert spec.ffn_width == 2 * intermediate
    assert spec.vocab == config["vocab_size"]
    assert spec.tied_head == config["tie_word_embeddings"]
    assert spec.attn_output_gate == config["attn_output_gate"]
    assert spec.gdn_key_heads == config["linear_num_key_heads"]
    assert spec.gdn_value_heads == config["linear_num_value_heads"]
    assert spec.gdn_key_dim == config["linear_key_head_dim"]
    assert spec.gdn_value_dim == config["linear_value_head_dim"]
    assert spec.conv_width == config["linear_conv_kernel_dim"]
    assert [spec.layer_kind(i) for i in range(spec.layers)] == [
        {"linear_attention": "gated_delta_net"}.get(kind, kind)
        for kind in config["layer_types"]]
    assert MODELS[spec.name] is spec
    if spec.experts:
        assert spec.experts == config["num_experts"]
        assert spec.experts_active == config["num_experts_per_tok"]
        assert spec.expert_intermediate == config["moe_intermediate_size"]
        from quail.cost import budgets
        from quail.cost.dense_decoder_cost import mlp_weight_params
        from quail.specs import H100_SXM

        assert mlp_params(spec) == 40 * 3 * 2048 * (512 + 8 * 512)
        assert mlp_weight_params(spec) == 40 * 3 * 2048 * (512 + 256 * 512)
        parts = {p.name: p for p in dense_decoder_components(
            Work(tokens=100, pairs=0, kv_written=0, kv_read=0), spec, 1)}
        assert parts["router"].flops == 2 * 100 * 40 * 2048 * 256
        assert spec.widest_projection == 12288
        assert budgets.kernel_index_cap(spec) * 8 * 2048 < 2**31
        assert spec.act_per_token >= 2 * 8 * (2048 + 2048) + 32 * 2048
        assert budgets.chunk_budget(spec, H100_SXM) == spec.chunk_cap_tokens
        assert budgets.arena_pages(spec, H100_SXM)[0] > 0


def test_weights_match_the_plan():
    # equation 21: 2 x (embedding + MLP + attention + Gated DeltaNet) is
    # 8,411,152,384 bytes; vLLM's model memory after loading is 1.6
    # percent above it, and the spec carries the measured figure
    assert SPEC.W_mem == 8_545_764_864
    assert SPEC.W_mem / 8_411_152_384 < 1.02
    assert SPEC.widest_projection == 18_432
    conv_weights = 24 * (2 * 16 * 128 + 32 * 128) * 4
    assert conv_weights == 786_432
    assert SPEC.params == (attention_projection_params(SPEC)
                           + gdn_projection_params(SPEC)
                           + mlp_params(SPEC) + conv_weights)
    assert gdn_projection_params(SPEC) == 24 * 42_106_880
    assert attention_projection_params(SPEC) == 8 * 36_700_160


def test_dense_components_price_gdn_projections_with_the_attention_ones():
    work = Work(tokens=1_000, pairs=0, kv_written=0, kv_read=0)
    attn_proj, mlp, attention, *_ = dense_decoder_components(
        work, SPEC, passes=1.0)
    assert attn_proj.flops == 2.0 * 1_000 * (
        8 * 36_700_160 + 24 * 42_106_880)
    assert attn_proj.bytes_moved == 2.0 * (8 * 36_700_160 + 24 * 42_106_880)
    assert mlp.flops == 2.0 * 1_000 * 32 * 3 * 2560 * 9216
    assert attention.flops == 0
    fp8 = QWEN3_5_4B_FP8
    parts = {part.name: part for part in dense_decoder_components(work, fp8, 1)}
    assert parts["gdn_proj"].precision == "bf16"
    assert parts["gdn_proj"].bytes_moved == 2 * gdn_projection_params(fp8)
    assert parts["attn_proj"].precision == parts["mlp"].precision == "fp8"
    assert fp8.checkpoint_bytes == SPEC.checkpoint_bytes
    assert MODELS[fp8.name] is fp8


def test_vllm_takes_no_images_for_the_vision_checkpoint_only():
    from quail.baselines.vllm import DefaultVLLMEngine, VLLMEngine

    off = {"image": 0, "video": 0}
    for engine in (VLLMEngine(), DefaultVLLMEngine()):
        assert engine.llm_kwargs(SPEC)["limit_mm_per_prompt"] == off
        assert "limit_mm_per_prompt" not in engine.llm_kwargs(QWEN3_4B_FP8, H100_SXM)


def test_vllm_sequences_stay_under_the_state_blocks_it_holds():
    from quail.baselines.vllm import MAX_SEQUENCES, VLLMEngine

    assert VLLMEngine().llm_kwargs(SPEC)["max_num_seqs"] == 3_000 < MAX_SEQUENCES
    assert VLLMEngine().llm_kwargs(QWEN3_4B_FP8, H100_SXM)[
        "max_num_seqs"] == MAX_SEQUENCES


def test_a_sliding_hybrid_slides_only_its_attention_layers():
    from dataclasses import replace

    sliding = replace(
        SPEC, sliding_window=128,
        layer_kinds=tuple("sliding_attention" if kind == "full_attention" else kind
                          for kind in SPEC.layer_kinds))
    assert sliding.sliding_layer_set == frozenset(range(3, 32, 4))
    assert not any(sliding.is_full_layer(i) for i in range(32))


def test_gdn_core_and_convolution_are_priced_per_token():
    from quail.cost.dense_decoder_cost import (
        conv_flops_per_token,
        gdn_core_flops_per_token,
    )

    work = Work(tokens=1_000, pairs=0, kv_written=0, kv_read=0)
    names = [c.name for c in dense_decoder_components(work, SPEC, passes=1.0)]
    assert names == ["attn_proj", "mlp", "attention", "gdn_core", "state",
                     "conv"]
    # per token and value head: 6 C K + 4 C V + 6 K V + 2 C^2 / 3 with
    # C = 64 and K = V = 128, over 32 value heads and 24 layers
    per_head = 6 * 64 * 128 + 4 * 64 * 128 + 6 * 128 * 128 + 2 * 64 * 64 / 3
    assert gdn_core_flops_per_token(SPEC) == per_head * 32 * 24
    # 2 FLOPs per tap per channel: 8,192 channels, width 4, 24 layers
    assert conv_flops_per_token(SPEC) == 2 * 4 * 8_192 * 24


def test_a_pure_attention_model_prices_no_state_bytes():
    from quail.cost.work import ask
    from quail.specs import QWEN3_4B_FP8

    names = [c.name for c in dense_decoder_components(
        ask(100, 10), QWEN3_4B_FP8, passes=1.0)]
    assert "state" not in names


def test_one_review_through_a_filter_and_a_join_matches_the_hand_calculation():
    from quail.cost.sol import speed_of_light
    from quail.cost.work import ask, scan, stream
    from quail.specs import H100_SXM

    # 100-token review, 6-token preamble, 56-token filter question, then a
    # join: a 43-token frame and two aspects of 2 and 3 tokens, each with a
    # 13-token tail
    work = (scan(106, 56)
            + ask(106, 43) + Work(state_written=1.0)
            + stream(149, [15, 16]))
    assert work.tokens == 236
    assert work.pairs == 13_203 + 5_504 + 4_875
    assert work.kv_written == 236
    assert work.kv_read == 106 + 149
    assert (work.state_written, work.state_read) == (2, 3)
    sol = speed_of_light(work, SPEC, H100_SXM, chunk_tokens=116_508)
    assert sol.passes == 1
    assert [c.name for c in sol.components] == [
        "attn_proj", "mlp", "attention", "gdn_core", "state", "conv"]
    # one pass reads every weight once, so the three weight components
    # are limited by memory at this size
    assert sol.component("attn_proj").bound_by == "memory"
    assert sol.component("state").bytes_moved == 5 * 51_511_296
    assert sol.seconds == pytest.approx(2.2516e-3, rel=1e-3)
