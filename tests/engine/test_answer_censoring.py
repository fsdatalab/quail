"""Censored answer lengths and eviction priorities."""

from dataclasses import replace

import numpy as np
import pytest

from quail.backends.quail.executor.answer_admission import (
    admission_worthwhile,
    overflow_probability,
    page_moments,
)
from quail.backends.quail.executor.answer_survival import (
    quantile_tokens,
    remaining_tokens,
)
from quail.cost.answers import prior_survival
from quail.cost.budgets import compute_knee
from quail.cost.decode import DecodeCost
from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.retention import RetentionPolicy, coefficients
from quail.cost.work import Work
from quail.specs import DEVICES, MODELS


def test_cold_start_uses_the_map_length_prior():
    prior = prior_survival(3, 8)
    remaining = remaining_tokens([], [], 3, 8)
    for written in range(8):
        assert np.isclose(remaining[written], prior[written + 1:].sum()
                          / prior[written])
    assert remaining[8] == 0
    np.testing.assert_allclose(remaining, remaining_tokens([], [], 3, 8,
                                                          survival=prior))


def test_censored_answers_leave_the_risk_set_after_their_written_length():
    # The censor at 3 leaves two answers at risk at 4.
    remaining = remaining_tokens([2, -1, 4, 6], [2, 3, 4, 6], 3, 6,
                                 prior_weight=0)
    survival = np.array([1, 1, .75, .75, .375, .375, 0])
    expected = np.r_[np.cumsum(survival[:0:-1])[::-1], 0]
    expected[:-1] /= survival[:-1]
    np.testing.assert_allclose(remaining, expected)
    assert remaining[2] > remaining[4] > remaining[5]


def test_censored_answers_cannot_survive_beyond_the_answer_limit():
    remaining = remaining_tokens([-1], [2], 1, 6)
    np.testing.assert_allclose(remaining[2:], [3, 2, 1, 0, 0])


def test_page_moments_condition_on_written_tokens_and_count_page_slack():
    curve = prior_survival(5, 17)
    lengths = np.arange(1, len(curve))
    probability = -np.diff(curve)
    written = np.array([0, 3, 8, 12, 16, 17])
    available = np.array([0, 6, 8, 16, 20, 20])
    for page_tokens in (1, 4, 8):
        mean, variance = page_moments(curve, written, available, page_tokens)
        for index, count in enumerate(written):
            mass = probability * (lengths > count)
            if mass.sum():
                mass /= mass.sum()
            pages = np.maximum(0, -(-(lengths - 1 - available[index]) // page_tokens))
            expected = mass @ pages
            assert np.isclose(mean[index], expected)
            assert np.isclose(variance[index], mass @ (pages * pages) - expected ** 2)


def test_deterministic_answers_have_no_page_variance():
    curve = np.r_[np.ones(9), 0.0]
    mean, variance = page_moments(curve, np.array([0, 7]), np.array([0, 8]), 4)
    np.testing.assert_array_equal(mean, [2, 0])
    np.testing.assert_array_equal(variance, 0)


def pooled_admissions(expected_tokens, limit, page_tokens, free_pages,
                      prompt_pages, recompute, gain_per_token=1.0):
    curve = prior_survival(expected_tokens, limit)
    mean, variance = page_moments(curve, np.array([0]), np.array([page_tokens]),
                                 page_tokens)
    mean, variance = float(mean[0]), float(variance[0])
    maximum = -(-(limit - 1 - page_tokens) // page_tokens)
    num_documents = 0
    while free_pages >= prompt_pages and admission_worthwhile(
            free_pages, prompt_pages, num_documents * mean, num_documents * variance,
            num_documents * maximum, mean, variance, maximum,
            curve.sum() * gain_per_token, recompute):
        num_documents += 1
        free_pages -= prompt_pages
    return num_documents


def test_expensive_recomputations_admit_fewer_documents_with_the_same_answer_curves():
    cost = RetentionPolicy(**coefficients(MODELS["qwen3-4b-fp8"], DEVICES["h100-sxm"],
                                         peak=True))
    short_work = cost.prefill_seconds(16)
    long_work = cost.prefill_seconds(1024)
    short = pooled_admissions(32, 128, 4, 100, 2, recompute=short_work,
                              gain_per_token=cost.linear_seconds)
    long = pooled_admissions(32, 128, 4, 100, 2, recompute=long_work,
                             gain_per_token=cost.linear_seconds)
    assert long_work / 1024 > short_work / 16 > cost.linear_seconds
    assert long < short


def test_pooling_small_answers_admits_more_than_individual_p90_reservations():
    num_documents = pooled_admissions(8, 64, 1, 1000, 3, recompute=10_000)
    quantile = quantile_tokens([], [], 8, 64, .9)
    # One answer token is already covered by the three prompt pages.
    reserved = 1000 // (3 + quantile - 2)
    assert num_documents > reserved


def test_decode_above_the_old_knee_requires_full_answer_capacity():
    model, device = MODELS["qwen3-32b-fp8"], DEVICES["h100-sxm"]
    cost = DecodeCost.from_specs(model, device)
    work = cost.row_work(np.full(600, 16)).total()
    added = cost.row_work([16]).total()
    assert work.tokens > compute_knee(model, device)
    gain = cost.gain(work, added, 4)
    assert gain == pytest.approx(0, abs=1e-16)
    recompute = cost.recompute_cost.prefill_seconds(16)
    assert not admission_worthwhile(4, 1, 0, 0, 0, 2, 1, 4, gain, recompute)
    assert admission_worthwhile(5, 1, 0, 0, 0, 2, 1, 4, gain, recompute)


def test_long_candidate_contexts_admit_less_than_the_old_free_pass_rule():
    cost = DecodeCost.from_specs(MODELS["qwen3-32b-fp8"], DEVICES["h100-sxm"])
    work = cost.row_work(np.full(200, 1500)).total()
    added = cost.row_work([16_000]).total()
    recompute = cost.recompute_cost.prefill_seconds(2000)
    gain_per_token = cost.gain(work, added, 1)
    assert gain_per_token == 0
    old = pooled_admissions(512, 1024, 16, 2000, 100, recompute,
                            gain_per_token=cost.recompute_cost.linear_seconds)
    new = pooled_admissions(512, 1024, 16, 2000, 100, recompute,
                            gain_per_token=gain_per_token)
    assert new < old


def test_sliding_layers_cap_reads_and_attention_without_capping_full_layers():
    model = MODELS["diffusion-gemma-26b-a4b-fp8"]
    cost = DecodeCost.from_specs(model, DEVICES["h100-sxm"])
    window = model.sliding_window
    work = cost.row_work([window, 2 * window, 4 * window])
    np.testing.assert_array_equal(work.sliding_pairs, window)
    np.testing.assert_array_equal(work.sliding_kv_read, window)
    np.testing.assert_array_equal(work.pairs, [window, 2 * window, 4 * window])
    np.testing.assert_array_equal(work.kv_read, work.pairs)
    assert cost.sliding_kv_seconds == model.kappa_sliding / DEVICES["h100-sxm"].hbm_bw


def test_tree_shares_kv_reads_but_not_attention_flops():
    model = MODELS["diffusion-gemma-26b-a4b-fp8"]
    cost = DecodeCost.from_specs(model, DEVICES["h100-sxm"])
    window = model.sliding_window
    contexts = np.array([window, window + 100, 2 * window])
    alone = cost.row_work(contexts).total()
    shared = cost.row_work(contexts, [0, 0, 1], [window, window, 0]).total()
    assert shared.pairs == alone.pairs
    assert shared.sliding_pairs == alone.sliding_pairs
    assert shared.kv_read == alone.kv_read - window
    assert shared.sliding_kv_read == alone.sliding_kv_read - window + 100


def test_decode_prices_ignore_measured_throughput_and_loaded_footprints():
    model, device = MODELS["qwen3-32b-fp8"], DEVICES["h100-sxm"]
    cost = DecodeCost.from_specs(model, device)
    assert cost == DecodeCost.from_specs(
        replace(model, w_mem_bytes=1),
        replace(device, mamf_fp8_flops=1, mamf_bf16_flops=1))


@pytest.mark.parametrize("model_name", [
    "qwen3.5-4b-bf16", "qwen3.5-27b-fp8", "qwen3.8-27b-fp8", "lfm2-2.6b-bf16",
])
@pytest.mark.parametrize("device_name", ["h100-sxm", "l40s", "b200"])
def test_hybrid_decode_prices_state_movement_and_compute_from_specs(
        model_name, device_name):
    model, device = MODELS[model_name], DEVICES[device_name]
    cost = DecodeCost.from_specs(model, device)
    work = cost.row_work([16, 16], [0, 0], [16, 16]).total()
    assert work.kv_read == 16
    assert work.state_read == work.state_written == work.tokens == 2
    assert cost.state_seconds == model.checkpoint_bytes / device.hbm_bw > 0
    components = dense_decoder_components(work, model, passes=1)
    memory = sum(component.bytes_moved for component in components) / device.hbm_bw
    compute = sum(component.flops / device.arithmetic_bandwidth(component.precision)
                  for component in components)
    assert cost.seconds(work) == pytest.approx(max(memory, compute))
    assert cost.seconds(work) > replace(cost, state_seconds=0).seconds(work)
    compute_bound = DecodeCost.from_specs(model, replace(device, hbm_bw=float("inf")))
    assert compute_bound.seconds(work) == pytest.approx(compute)
    state_compute = sum(
        component.flops / device.arithmetic_bandwidth(component.precision)
        for component in components if component.name in ("gdn_core", "conv"))
    assert state_compute > 0
    assert cost.recompute_cost.linear_seconds * work.tokens > state_compute
    assert cost.seconds(Work()) == 0
    assert cost.seconds(cost.row_work([]).total()) == 0
    assert cost == DecodeCost.from_specs(
        replace(model, w_mem_bytes=1),
        replace(device, mamf_fp8_flops=1, mamf_bf16_flops=1))


@pytest.mark.parametrize("model_name", ["qwen3-4b-fp8", "qwen3-32b-fp8"])
@pytest.mark.parametrize("device_name", ["h100-sxm", "l40s", "b200"])
@pytest.mark.parametrize("num_rows,context", [(1, 0), (200, 1500), (10000, 16)])
def test_qwen3_decode_prices_are_bit_identical_without_state_layers(
        model_name, device_name, num_rows, context):
    model, device = MODELS[model_name], DEVICES[device_name]
    cost = DecodeCost.from_specs(model, device)
    work = cost.row_work(np.full(num_rows, context)).total()
    policy = RetentionPolicy(**coefficients(model, device, peak=True))
    weight_bytes = sum(component.bytes_moved for component in
                       dense_decoder_components(Work(), model, passes=1))
    memory = (weight_bytes / device.hbm_bw
              + model.kappa_full / device.hbm_bw * work.kv_read
              + model.kappa_sliding / device.hbm_bw * work.sliding_kv_read)
    compute = (policy.linear_seconds * work.tokens
               + policy.pair_seconds * work.pairs
               + policy.sliding_pair_seconds * work.sliding_pairs)
    assert cost.state_seconds == work.state_read == work.state_written == 0
    assert cost.seconds(work).hex() == max(memory, compute).hex()


def test_qwen32_h100_decode_and_recompute_worked_example():
    # For 200 rows attending 1500 tokens, KV reads dominate the fp8 weight read.
    cost = DecodeCost.from_specs(MODELS["qwen3-32b-fp8"], DEVICES["h100-sxm"])
    work = cost.row_work(np.full(200, 1500)).total()
    added = cost.row_work([1500]).total()
    assert cost.seconds(work) == pytest.approx(0.0327906931)
    marginal = cost.seconds(work + added) - cost.seconds(work)
    assert marginal == pytest.approx(0.0001173779)
    assert cost.recompute_cost.prefill_seconds(2000) == pytest.approx(0.0673144464)


def test_overflow_respects_full_capacity_and_deterministic_lengths():
    assert overflow_probability(10, 9, 4, 10) == 0
    assert overflow_probability(9, 9, 0, 10) == 0
    assert overflow_probability(8, 9, 0, 10) == 1
    assert overflow_probability(-1, 9, 4, 10) == 1
    assert not admission_worthwhile(20, 1, 0, 0, 0, 1, 1, 30, 0, 10)
    assert admission_worthwhile(20, 1, 0, 0, 0, 1, 1, 19, 0, 10)
