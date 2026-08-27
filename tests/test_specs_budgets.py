"""Tests for model specs and budget calculations on Qwen3-4B / H100."""

from dataclasses import replace

import pytest

from quail.planner import budgets
from quail.specs import H100_SXM, QWEN3_4B_FP8


def test_model_widths_and_budget_limits():
    m = QWEN3_4B_FP8
    assert m.kv_elements_per_token == 73_728
    assert m.kappa == 147_456                      # bf16
    assert m.with_kv_bytes(1.0).kappa == 73_728    # fp8
    assert m.act_per_token == 81_920
    assert m.intermediate == 9_728
    assert m.W_mem == 4.5e9
    assert budgets.tensor_parallel(QWEN3_4B_FP8, H100_SXM) == 1
    big = replace(QWEN3_4B_FP8, w_mem_bytes=150e9)
    assert budgets.tensor_parallel(big, H100_SXM) == 4
    assert budgets.kernel_index_cap(QWEN3_4B_FP8) == 110_376
    assert budgets.chunk_memory_bound(QWEN3_4B_FP8, H100_SXM) == 421_752
    assert budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM) == 110_376


def test_arena_memory_tracks_kv_width():
    # the design table said ~400k with one chunk of activation
    # reservation; the milestone 1 filter run OOMed there, so the
    # reservation is two chunks - the arena lands near 346k, ~860
    # mean-length documents resident at once
    tokens = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM)
    assert 340_000 <= tokens <= 370_000
    assert tokens // 400 >= 750
    fp8 = budgets.arena_tokens(QWEN3_4B_FP8.with_kv_bytes(1.0), H100_SXM)
    assert fp8 == pytest.approx(2 * tokens, rel=0.01)


def test_compute_knee_and_attention_crossover():
    knee = budgets.compute_knee(QWEN3_4B_FP8, H100_SXM)
    assert 380 <= knee <= 450
    s = budgets.attention_crossover(QWEN3_4B_FP8, H100_SXM)
    assert 11_000 <= s <= 13_500


def test_derived_table():
    table = budgets.derived_table(QWEN3_4B_FP8, H100_SXM)
    assert table["tensor_parallel"] == 1
    assert table["chunk_budget"] == 110_376
    assert table["arena_tokens"] == budgets.arena_tokens(
        QWEN3_4B_FP8, H100_SXM)
