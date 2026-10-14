"""Tests for model specs and budget calculations on Qwen3-4B / H100."""

from dataclasses import replace

import pytest

from quail.cost import budgets
from quail.cost.dense_decoder_cost import (
    attention_pair_flops,
    dense_decoder_components,
)
from quail.cost.joins import PricedJoin, stage_work
from quail.cost.work import ask, scan, stream, triangle
from quail.planner.statistics.tables import summarize_alias
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
    fp8 = budgets.arena_tokens(replace(QWEN3_4B_FP8, kv_bytes=1.0), H100_SXM)
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


def test_measured_attention_rates_interpolate_and_fall_below_the_table():
    rates = H100_SXM.attention_rates_for(32, 8, 128)
    assert H100_SXM.attention_rates_for(64, 8, 128) is None
    assert RTX_PRO_6000_BLACKWELL_SERVER.attention_rates_for(32, 8, 128) is None
    # measured points, halfway in log length, and held past the longest
    assert rates.causal_rate(1024) == 411.2e12
    assert rates.causal_rate((1024 * 2048) ** 0.5) == pytest.approx(
        (411.2e12 + 506.1e12) / 2)
    assert rates.causal_rate(10**6) == 667.6e12
    # below the shortest document a call takes as long as at the
    # shortest, so the rate falls in proportion
    assert rates.causal_rate(64) == pytest.approx(104.1e12 / 2)
    assert rates.read_rate(64, 8192) == 496e12
    assert rates.read_rate(8, 8192) == pytest.approx(195e12 / 2)
    assert rates.read_rate(64, 100) == 412e12
    assert rates.read_rate(4096, 10**6) == 578e12


def test_attention_prices_at_measured_rates_for_the_measured_layout():
    work = scan(1024, 0) * 10 + ask(8192, 64) * 5
    full, _ = attention_pair_flops(QWEN3_4B_FP8)
    rates = H100_SXM.attention_rates_for(32, 8, 128)
    causal = 10 * triangle(1024) + 5 * triangle(64)
    length = 2 * causal / (10 * 1024 + 5 * 64) - 1
    attention = dense_decoder_components(work, QWEN3_4B_FP8, 1, H100_SXM)[2]
    assert attention.compute_seconds == pytest.approx(
        full * causal / rates.causal_rate(length)
        + full * 5 * 64 * 8192 / rates.read_rate(64, 8192))
    # without a device, or without a table for the layout, the flops
    # price at the bf16 peak
    assert dense_decoder_components(work, QWEN3_4B_FP8, 1)[2].compute_seconds is None
    assert dense_decoder_components(
        work, QWEN3_32B_FP8, 1, H100_SXM)[2].compute_seconds is None


def test_reads_count_one_per_prefix_read_however_many_suffixes_stack():
    branches = stream(100, [20, 0, 30])
    assert branches.reads == 1
    assert branches.read_rows == 50 and branches.read_pairs == 100 * 50
    assert stream(0, [20]).reads == 0 and stream(100, [0]).reads == 0
    spec = PricedJoin(
        written_pos=0, aliases=("a", "b"), anchor="a", anchor_free=False,
        semantics="full", selectivity=None, frame_tokens={"a": 3, "b": 3},
        label_tokens={"a": 1, "b": 1}, tail_tokens=4)
    lengths = {"a": summarize_alias([100, 200, 300]),
               "b": summarize_alias([50, 60])}
    # each resident anchor reads its prefix once for the frame and once
    # for the stacked partner suffixes
    work = stage_work(spec, "a", 3, 6, lengths, 10, resident=True)
    assert work.reads == 6
    assert stage_work(spec, "a", 3, 6, lengths, 10).reads == 3
    assert stage_work(spec, "a", 1.5, 3, lengths, 10).reads == 1.5
