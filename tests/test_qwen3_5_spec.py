"""The Qwen3.5-4B spec against its config.json and the hybrid-model plan."""

import json
from pathlib import Path

from quail.cost.dense_decoder_cost import (
    attention_projection_params,
    dense_decoder_components,
    gdn_projection_params,
    mlp_params,
)
from quail.cost.work import Work
from quail.specs import MODELS, QWEN3_4B_FP8, QWEN3_5_4B_BF16

CONFIG = Path(__file__).parent / "fixtures" / "qwen3_5_configs" / "qwen3.5-4b-bf16.json"
SPEC = QWEN3_5_4B_BF16


def _text_config():
    return json.loads(CONFIG.read_text())["text_config"]


def test_spec_matches_the_config():
    config = _text_config()
    assert SPEC.layers == config["num_hidden_layers"]
    assert SPEC.hidden == config["hidden_size"]
    assert SPEC.n_q == config["num_attention_heads"]
    assert SPEC.n_kv == config["num_key_value_heads"]
    assert SPEC.d_head == config["head_dim"]
    assert SPEC.ffn_width == 2 * config["intermediate_size"]
    assert SPEC.vocab == config["vocab_size"]
    assert SPEC.tied_head == config["tie_word_embeddings"]
    assert SPEC.attn_output_gate == config["attn_output_gate"]
    assert SPEC.gdn_key_heads == config["linear_num_key_heads"]
    assert SPEC.gdn_value_heads == config["linear_num_value_heads"]
    assert SPEC.gdn_key_dim == config["linear_key_head_dim"]
    assert SPEC.gdn_value_dim == config["linear_value_head_dim"]
    assert SPEC.conv_width == config["linear_conv_kernel_dim"]
    assert [SPEC.layer_kind(i) for i in range(SPEC.layers)] == [
        {"linear_attention": "gated_delta_net"}.get(kind, kind)
        for kind in config["layer_types"]]
    assert MODELS[SPEC.name] is SPEC


def test_only_attention_layers_keep_kv():
    assert [i for i in range(SPEC.layers) if SPEC.keeps_kv(i)] == list(range(3, 32, 4))
    assert len(SPEC.gdn_layers) == 24
    # 8 layers x 2 x 4 KV heads x 256 x 2 bytes, 4.5x less than Qwen3-4B's 147,456
    assert SPEC.kappa == 32_768
    assert QWEN3_4B_FP8.kappa / SPEC.kappa == 4.5
    assert SPEC.kappa_sliding == 0


def test_weights_match_the_plan():
    # equation 21: 2 x (embedding + MLP + attention + Gated DeltaNet)
    assert SPEC.W_mem == 8_411_152_384
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
    attn_proj, mlp, attention = dense_decoder_components(work, SPEC, passes=1.0)
    assert attn_proj.flops == 2.0 * 1_000 * (
        8 * 36_700_160 + 24 * 42_106_880)
    assert attn_proj.bytes_moved == 2.0 * (8 * 36_700_160 + 24 * 42_106_880)
    assert mlp.flops == 2.0 * 1_000 * 32 * 3 * 2560 * 9216
    assert attention.flops == 0


def test_vllm_takes_no_images_for_the_vision_checkpoint_only():
    from quail.backends.vllm import DefaultVLLMEngine, VLLMEngine

    off = {"image": 0, "video": 0}
    for engine in (VLLMEngine(), DefaultVLLMEngine()):
        assert engine.llm_kwargs(SPEC)["limit_mm_per_prompt"] == off
        assert "limit_mm_per_prompt" not in engine.llm_kwargs(QWEN3_4B_FP8)


def test_vllm_sequences_stay_under_the_state_blocks_it_holds():
    from quail.backends.vllm import MAX_SEQUENCES, VLLMEngine

    assert VLLMEngine().llm_kwargs(SPEC)["max_num_seqs"] == 3_000 < MAX_SEQUENCES
    assert VLLMEngine().llm_kwargs(QWEN3_4B_FP8)["max_num_seqs"] == MAX_SEQUENCES


def test_turn_closes_the_empty_thinking_block_the_chat_template_writes():
    assert SPEC.turn_prefix == "<|im_start|>user\n"
    assert SPEC.turn_suffix == (
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_a_sliding_hybrid_slides_only_its_attention_layers():
    from dataclasses import replace

    sliding = replace(
        SPEC, sliding_window=128,
        layer_kinds=tuple("sliding_attention" if kind == "full_attention" else kind
                          for kind in SPEC.layer_kinds))
    assert sliding.sliding_layer_set == frozenset(range(3, 32, 4))
    assert not any(sliding.is_full_layer(i) for i in range(32))


def test_chunk_size_follows_the_plan():
    from quail.cost import budgets
    from quail.specs import H100_SXM

    # equation 22: 2^31 - 1 over the widest projection, 18,432 columns
    assert budgets.kernel_index_cap(SPEC) == 116_508
    # the memory bound before it, 412,529 tokens
    assert budgets.chunk_memory_bound(SPEC, H100_SXM) == 412_529
    assert budgets.chunk_budget(SPEC, H100_SXM) == 116_508
