"""The section-8 table at 4B/H100, asserted against the numbers the
exploration repo measured and derived."""

from dataclasses import replace

import pytest

from quail.planner import budgets
from quail.planner.decide import channel_bandwidths, load_calibration
from quail.specs import H100_SXM, QWEN3_4B_FP8


def test_kappa_and_widths():
    m = QWEN3_4B_FP8
    assert m.kv_elements_per_token == 73_728
    assert m.kappa == 147_456                      # bf16
    assert m.with_kv_bytes(1.0).kappa == 73_728    # fp8
    assert m.act_per_token == 81_920
    assert m.intermediate == 9_728
    assert m.W_mem == 4.5e9


def test_tensor_parallel_is_one_at_4b():
    assert budgets.tensor_parallel(QWEN3_4B_FP8, H100_SXM) == 1


def test_tensor_parallel_grows_with_weights():
    big = replace(QWEN3_4B_FP8, w_mem_bytes=150e9)
    assert budgets.tensor_parallel(big, H100_SXM) == 4


def test_kernel_index_cap():
    # (2^31 - 1) // 19456; the join2way crash is the tuition paid
    assert budgets.kernel_index_cap(QWEN3_4B_FP8) == 110_376


def test_chunk_memory_bound_matches_exploration_b_star():
    # (80e9 * 0.92 - 4.5e9) // 81920 // 2 = the join plan's 421,752
    assert budgets.chunk_memory_bound(QWEN3_4B_FP8, H100_SXM) == 421_752


def test_chunk_budget_is_the_index_cap():
    assert budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM) == 110_376


def test_arena_tokens_at_bf16():
    # the design table said ~400k with one chunk of activation
    # reservation; the milestone 1 filter run OOMed there, so the
    # reservation is two chunks, and a further ~4.8 GiB is reserved
    # for the pinned store's device staging ring (store_staging_bytes)
    # so a later warm-pass store can't OOM against an arena that
    # already claimed the whole budget - the arena lands near 313k,
    # still ~780 mean-length documents resident at once
    tokens = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM)
    assert 300_000 <= tokens <= 330_000
    assert tokens // 400 >= 750


def test_arena_doubles_at_fp8_kv():
    bf16 = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM)
    fp8 = budgets.arena_tokens(QWEN3_4B_FP8.with_kv_bytes(1.0), H100_SXM)
    assert fp8 == pytest.approx(2 * bf16, rel=0.01)


def test_compute_knee_a_few_hundred_tokens():
    knee = budgets.compute_knee(QWEN3_4B_FP8, H100_SXM)
    assert 380 <= knee <= 450


def test_attention_crossover_near_12k():
    s = budgets.attention_crossover(QWEN3_4B_FP8, H100_SXM)
    assert 11_000 <= s <= 13_500


def test_calibration_anchor_file():
    cal = load_calibration(QWEN3_4B_FP8, H100_SXM)
    assert cal.source == "calibrated"
    assert cal.rate_tokens_per_s == pytest.approx(121_045, rel=1e-3)
    assert cal.a2_s_per_token2 == pytest.approx(4.9336e-10, rel=1e-3)


def test_store_break_even_under_pinned_bandwidth():
    cal = load_calibration(QWEN3_4B_FP8, H100_SXM)
    bw = channel_bandwidths()
    bf16 = budgets.store_break_even_bytes_per_s(
        QWEN3_4B_FP8, cal.a_s_per_token)
    # ~18 GB/s at bf16 KV and the packed 121k rate. Pinned host
    # memory clears it; disk and volumes do not.
    assert 17e9 <= bf16 <= 19e9
    assert bw["pinned_h2d"] > bf16
    assert bw["disk_read"] < bf16
    assert bw["volume_read"] < bf16


def test_spec_scaled_defaults_for_uncalibrated_pair():
    double = replace(QWEN3_4B_FP8, name="qwen3-8b-ish", params=7.2e9)
    cal = load_calibration(double, H100_SXM)
    anchor = load_calibration(QWEN3_4B_FP8, H100_SXM)
    assert cal.source.startswith("spec-scaled")
    assert cal.a_s_per_token == pytest.approx(
        2 * anchor.a_s_per_token, rel=1e-6)
    assert cal.a2_s_per_token2 == pytest.approx(
        2 * anchor.a2_s_per_token2, rel=1e-6)


def test_derived_table_complete():
    cal = load_calibration(QWEN3_4B_FP8, H100_SXM)
    table = budgets.derived_table(QWEN3_4B_FP8, H100_SXM,
                                  cal.a_s_per_token)
    assert table["tensor_parallel"] == 1
    assert table["chunk_budget"] == 110_376
    assert table["serving_rate_tokens_per_s"] == pytest.approx(121_045,
                                                               rel=1e-3)
