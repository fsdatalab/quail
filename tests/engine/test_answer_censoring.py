"""Censored answer lengths and eviction priorities."""

from dataclasses import replace

import numpy as np
import pytest

from quail.backends.quail.executor.answer_admission import (
    admission_worthwhile,
    page_moments,
)
from quail.backends.quail.executor.answer_survival import (
    quantile_tokens,
    remaining_tokens,
)
from quail.cost.answers import prior_survival
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
