"""Tests for model specs and budget calculations on Qwen3-4B / H100."""

from dataclasses import replace

import pytest

from quail.cost import budgets
from quail.specs import (
    H100_SXM,
    QWEN3_4B_FP8,
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


def test_gemma_layer_kinds_agree_with_its_period_and_window():
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    gemma = DIFFUSION_GEMMA_26B_FP8
    implicit = replace(gemma, layer_kinds=())
    assert gemma.layer_kinds == tuple(
        implicit.layer_kind(i) for i in range(gemma.layers))
    assert gemma.kv_shapes == implicit.kv_shapes
    assert gemma.sliding_layer_set == implicit.sliding_layer_set
