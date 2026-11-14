"""Qwen hybrid specs against their checkpoint configs."""

import json
from pathlib import Path

import pytest

from quail.specs import MODELS, QWEN3_5_4B_BF16

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
