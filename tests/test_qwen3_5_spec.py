"""The Qwen3.5-4B spec against its config.json and the hybrid-model plan."""

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
    # the memory bound before it: 412,529 tokens from the config's bytes,
    # 411,707 from vLLM's measured model memory
    assert budgets.chunk_memory_bound(SPEC, H100_SXM) == 411_707
    assert budgets.chunk_budget(SPEC, H100_SXM) == 116_508


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


def test_a_pure_attention_model_has_no_state_components():
    from quail.specs import QWEN3_4B_FP8

    work = Work(tokens=1_000, pairs=0, kv_written=0, kv_read=0)
    names = [c.name for c in dense_decoder_components(
        work, QWEN3_4B_FP8, passes=1.0)]
    assert names == ["attn_proj", "mlp", "attention"]


def test_state_component_prices_checkpoint_writes_and_reads_like_kv():
    from quail.cost.work import ask, scan, stream

    # a scan saves one checkpoint, an ask reads one, a stream reads one
    # per suffix
    assert scan(100, 10).state_written == 1 and scan(100, 10).state_read == 0
    assert ask(100, 10).state_read == 1 and ask(100, 10).state_written == 0
    assert stream(100, [10, 20, 30]).state_read == 3
    work = scan(100, 10) + stream(100, [10, 20, 30]) * 2
    assert (work.state_written, work.state_read) == (1, 6)
    state = next(c for c in dense_decoder_components(work, SPEC, passes=1.0)
                 if c.name == "state")
    assert state.flops == 0
    assert state.bytes_moved == SPEC.checkpoint_bytes * 7
    assert SPEC.checkpoint_bytes == 51_511_296


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
