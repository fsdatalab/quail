"""Tests for model specs and budget calculations on Qwen3-4B / H100."""

from dataclasses import replace

import pytest

from quail.cost import budgets
from quail.specs import (
    DIFFUSION_GEMMA_26B_FP8,
    H100_SXM,
    QWEN3_4B_FP8,
    QWEN3_5_4B_BF16,
    QWEN3_32B_FP8,
    RTX_PRO_6000_BLACKWELL_SERVER,
)


def test_model_weights_and_kv_memory_budgets():
    # Tied output weights must remain available for input embeddings.
    assert QWEN3_4B_FP8.head_mem_bytes == 0.0
    assert QWEN3_4B_FP8.W_resident == QWEN3_4B_FP8.W_mem
    # The separate 32B output head stays resident for AI.CLASSIFY: its
    # 1,555,824,640 bytes are 5,936 KV tokens (262,144 bytes each, in
    # whole 16-token pages) the arena does not get.
    head = QWEN3_32B_FP8.head_mem_bytes
    assert head == 151_936 * 5_120 * 2
    assert QWEN3_32B_FP8.W_resident == QWEN3_32B_FP8.W_mem
    assert budgets.arena_tokens(QWEN3_32B_FP8, H100_SXM) == 112_304 - 5_936
    # the chunk budget does not move: the 32B chunk is capped by the
    # int32 kernel index, not by memory
    kept = replace(QWEN3_32B_FP8, tied_head=True)
    assert budgets.chunk_budget(QWEN3_32B_FP8, H100_SXM) == \
        budgets.chunk_budget(kept, H100_SXM)

    tokens = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM)
    assert tokens == 362_240     # 22,640 pages of 16 tokens
    assert tokens // 400 == 905
    fp8 = budgets.arena_tokens(QWEN3_4B_FP8.with_kv_bytes(1.0), H100_SXM)
    assert fp8 == pytest.approx(2 * tokens, rel=0.01)
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        assert budgets.arena_tokens(model, RTX_PRO_6000_BLACKWELL_SERVER) \
            > budgets.arena_tokens(model, H100_SXM)


def test_arena_pages_needs_room_for_one_chunk_of_sliding_kv():
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    full, sliding = budgets.arena_pages(DIFFUSION_GEMMA_26B_FP8, H100_SXM)
    assert full > 0 and sliding >= budgets.transient_sliding_pages(
        budgets.chunk_budget(DIFFUSION_GEMMA_26B_FP8, H100_SXM),
        DIFFUSION_GEMMA_26B_FP8.sliding_window)
    small = replace(H100_SXM, mem_bytes=40e9)
    with pytest.raises(ValueError, match="one chunk's sliding KV"):
        budgets.arena_pages(DIFFUSION_GEMMA_26B_FP8, small)


def test_layer_kind_follows_the_existing_attention_layout():
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    gemma = DIFFUSION_GEMMA_26B_FP8
    # a dense model keeps every token in every layer
    assert {QWEN3_4B_FP8.layer_kind(i) for i in range(QWEN3_4B_FP8.layers)} \
        == {"full_attention"}
    # DiffusionGemma slides in 25 layers and keeps every token in each
    # sixth layer
    kinds = [gemma.layer_kind(i) for i in range(gemma.layers)]
    assert [i for i, kind in enumerate(kinds) if kind == "full_attention"] \
        == [5, 11, 17, 23, 29]
    assert kinds.count("sliding_attention") == 25
    # the kinds agree with the helpers they sit beside
    assert all((kind == "full_attention") == gemma.is_full_layer(i)
               for i, kind in enumerate(kinds))
    assert {i for i, kind in enumerate(kinds) if kind == "sliding_attention"} \
        == gemma.sliding_layer_set


def test_explicit_layer_kinds_replace_the_default_rule():
    kinds = ("short_conv", "short_conv", "full_attention", "short_conv")
    spec = replace(QWEN3_4B_FP8, layers=4, layer_kinds=kinds)
    assert [spec.layer_kind(i) for i in range(4)] == list(kinds)


def test_layer_kinds_must_name_every_layer():
    with pytest.raises(ValueError,
                       match="layer_kinds has 2 entries for 36 layers"):
        replace(QWEN3_4B_FP8, layer_kinds=("full_attention", "short_conv"))


def test_a_sliding_window_needs_layer_kinds():
    with pytest.raises(ValueError, match="sliding_window needs layer_kinds"):
        replace(QWEN3_4B_FP8, sliding_window=128)


def test_tree_attention_is_refused_for_a_model_with_a_gdn_layer():
    # one Gated DeltaNet layer is the only difference from Qwen3-4B
    kinds = ("gated_delta_net",) + ("full_attention",) * 35
    hybrid = replace(QWEN3_4B_FP8, layer_kinds=kinds)
    assert hybrid.params == QWEN3_4B_FP8.params
    assert not hybrid.canvas_tokens
    assert budgets.tree_attention_allowed(QWEN3_4B_FP8)
    assert not budgets.tree_attention_allowed(hybrid)


def test_attention_layers_are_the_layers_that_store_kv():
    assert QWEN3_5_4B_BF16.attention_layers == tuple(range(3, 32, 4))
    assert QWEN3_4B_FP8.attention_layers == tuple(range(36))


def test_budgets_of_models_without_gdn_layers_keep_their_values():
    for model, knee, crossover in [
            (QWEN3_4B_FP8, 416.4278836608033, 12321.0),
            (QWEN3_32B_FP8, 343.53425314744567, 29760.0),
            (DIFFUSION_GEMMA_26B_FP8, 445.28202599846264, 6402.0)]:
        assert budgets.compute_knee(model, H100_SXM) == knee
        assert budgets.attention_crossover(model, H100_SXM) == crossover


def _hybrid_projections():
    """(in, out, layers) of Qwen3.5-4B's dense projections, from its config."""
    return [(2560, (2 * 16 + 2 * 4) * 256, 8),    # qkv with the query gate
            (4096, 2560, 8),                       # attention output
            (2560, 2 * 9216, 32), (9216, 2560, 32),    # MLP
            (2560, 2 * 2048 + 2 * 4096 + 2 * 32, 24),  # GDN qkv, z, b, a
            (4096, 2560, 24)]                      # GDN output


def test_compute_knee_counts_each_projection_in_the_layers_that_have_it():
    ridge = H100_SXM.peak_flops / H100_SXM.hbm_bw
    tot_p = sum(layers * i * o for i, o, layers in _hybrid_projections()) / 32
    tot_io = sum(layers * (i + o) for i, o, layers in _hybrid_projections()) / 32
    expected = ridge * tot_p * 2.0 / (2.0 * tot_p - ridge * tot_io * 2)
    assert budgets.compute_knee(QWEN3_5_4B_BF16, H100_SXM) == pytest.approx(
        expected, rel=1e-12)


def test_projection_time_sums_each_projection_over_its_layers():
    chunk = 4096
    expected = 0.0
    for din, dout, layers in _hybrid_projections():
        params = din * dout
        moved = params * 2.0 + chunk * (din + dout) * budgets.ACT_BYTES
        expected += layers * max(2.0 * params * chunk / H100_SXM.peak_flops,
                                 moved / H100_SXM.hbm_bw)
    assert budgets._projection_time(
        QWEN3_5_4B_BF16, H100_SXM, chunk) == pytest.approx(expected, rel=1e-12)


def test_attention_time_covers_only_the_attention_layers():
    model, chunk, context = QWEN3_5_4B_BF16, 1024, 4000
    flops = 4.0 * chunk * context * 16 * 256
    moved = context * 32_768 / 8 + 2.0 * chunk * 16 * 256 * budgets.ACT_BYTES
    expected = max(flops / H100_SXM.peak_flops, moved / H100_SXM.hbm_bw) * 8
    assert budgets._attention_time(
        model, H100_SXM, chunk, context) == pytest.approx(expected, rel=1e-12)
    # the same layers with all 32 storing KV take 4 times as long
    dense = replace(model, layer_kinds=())
    assert budgets._attention_time(dense, H100_SXM, chunk, context) \
        == pytest.approx(4 * expected, rel=1e-12)


def test_attention_path_choice_does_not_depend_on_how_many_layers_are_gdn():
    # attention, tree, and merge times all scale with the attention layers
    dense = replace(QWEN3_5_4B_BF16, layer_kinds=())
    for readers, rows, node in [(2, 16, 4000), (64, 8, 20_000), (500, 2, 3000),
                                (8, 512, 100_000), (1000, 1, 500),
                                (2, 16, 500), (4, 64, 2000)]:
        kwargs = dict(readers=readers, reader_rows=rows, node_tokens=node)
        assert budgets.choose_attention_path(
            QWEN3_5_4B_BF16, H100_SXM, **kwargs) == budgets.choose_attention_path(
            dense, H100_SXM, **kwargs)


def test_the_tree_merge_is_charged_for_the_attention_layers_only():
    # tree takes 0.73 of the unified time with the merge over 8 layers,
    # and 1.24 of it if the merge were charged over all 32
    assert budgets.choose_attention_path(
        QWEN3_5_4B_BF16, H100_SXM, readers=2, reader_rows=16,
        node_tokens=500) == "tree"
