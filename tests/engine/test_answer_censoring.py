"""Censored answer lengths and eviction priorities."""

import numpy as np

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
from quail.cost.retention import RetentionPolicy, coefficients
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
                      prompt_pages, recompute):
    curve = prior_survival(expected_tokens, limit)
    mean, variance = page_moments(curve, np.array([0]), np.array([page_tokens]),
                                 page_tokens)
    mean, variance = float(mean[0]), float(variance[0])
    maximum = -(-(limit - 1 - page_tokens) // page_tokens)
    num_documents = 0
    while free_pages >= prompt_pages and admission_worthwhile(
            free_pages, prompt_pages, num_documents * mean, num_documents * variance,
            num_documents * maximum, mean, variance, maximum, curve.sum(), recompute):
        num_documents += 1
        free_pages -= prompt_pages
    return num_documents


def test_expensive_recomputations_admit_fewer_documents_with_the_same_answer_curves():
    cost = RetentionPolicy(**coefficients(MODELS["qwen3-4b-fp8"], DEVICES["h100-sxm"]))
    short_work = cost.prefill_seconds(16) / cost.linear_seconds
    long_work = cost.prefill_seconds(1024) / cost.linear_seconds
    short = pooled_admissions(32, 128, 4, 100, 2, recompute=short_work)
    long = pooled_admissions(32, 128, 4, 100, 2, recompute=long_work)
    assert long_work / 1024 > short_work / 16 > 1
    assert long < short


def test_pooling_small_answers_admits_more_than_individual_p90_reservations():
    num_documents = pooled_admissions(8, 64, 1, 1000, 3, recompute=10_000)
    quantile = quantile_tokens([], [], 8, 64, .9)
    # One answer token is already covered by the three prompt pages.
    reserved = 1000 // (3 + quantile - 2)
    assert num_documents > reserved


def test_overflow_respects_full_capacity_and_deterministic_lengths():
    assert overflow_probability(10, 9, 4, 10) == 0
    assert overflow_probability(9, 9, 0, 10) == 0
    assert overflow_probability(8, 9, 0, 10) == 1
    assert overflow_probability(-1, 9, 4, 10) == 1
    assert not admission_worthwhile(20, 1, 0, 0, 0, 1, 1, 30, 0, 10)
    assert admission_worthwhile(20, 1, 0, 0, 0, 1, 1, 19, 0, 10)
