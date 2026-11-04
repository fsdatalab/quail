"""Dense and MoE Qwen hybrid specs against checkpoint configs and costs."""

import json
from pathlib import Path

import pytest

from quail.common.dense_decoder_cost import (
    dense_decoder_components,
    mlp_params,
)
from quail.common.sol import speed_of_light
from quail.common.work import Work
from quail.specs import (
    H100_SXM,
    MODELS,
    QWEN3_5_4B_BF16,
)

CONFIG = (Path(__file__).parents[1] / "fixtures" / "qwen3_5_configs"
          / "qwen3.5-4b-bf16.json")
SPEC = QWEN3_5_4B_BF16


@pytest.mark.parametrize("name", ["qwen3.5-4b-bf16"])
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
        from quail.common import budgets
        from quail.common.dense_decoder_cost import mlp_weight_params
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
















def test_one_review_through_a_filter_and_a_join_matches_the_hand_calculation():
    from quail.common.work import ask, scan, stream

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
