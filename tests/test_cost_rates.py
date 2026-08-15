"""The rewired rate primitives against their measured anchors."""

import pytest

from quail.configs import DEVICES, MODELS
from quail.plan import cost

M = MODELS["Qwen3-4B-FP8"]
D = DEVICES["H100-SXM-80GB"]


def test_rate_is_derived_from_the_step_model():
    # on the calibrated configuration the rate is exactly the batch
    # sweep's per-token cost inverted, and lands within a percent of
    # the retired hand anchor 97,000/275,000 * R_D / 2P
    assert cost.read_rate(M, D) == pytest.approx(1.0 / cost.STEP_TOKEN_S)
    assert cost.read_rate(M, D) == pytest.approx(96_972, rel=0.01)


def test_rate_scales_with_the_device_ceiling():
    l40s = DEVICES["L40S-48GB"]
    ratio = cost.read_rate(M, D) / cost.read_rate(M, l40s)
    assert ratio == pytest.approx(D.R_D / l40s.R_D)


def test_quadratic_surcharge_prices_long_documents_higher():
    same_tokens = 32_768
    short = cost.t_in(M, D, same_tokens, doc_sq_tokens=64 * 512.0 ** 2)
    long = cost.t_in(M, D, same_tokens, doc_sq_tokens=2 * 16_384.0 ** 2)
    assert long > short


def test_surcharge_centers_on_the_calibration_corpus():
    # a corpus with exactly the calibration profile pays no surcharge:
    # its attention is already embedded in the sustained rate
    tokens = 1_000_000.0
    neutral = cost.t_in(M, D, tokens,
                        doc_sq_tokens=tokens * cost.CAL_SQ_PER_TOKEN)
    assert neutral == pytest.approx(tokens * cost.token_seconds(M, D))


def test_quest_read_term_prices_context_at_the_cached_rate():
    a = cost.t_quest(M, D, 1000, 1, 3, mean_doc_tokens=512)
    b = cost.t_quest(M, D, 1000, 1, 3, mean_doc_tokens=8192)
    assert b > a
    # the premium for longer documents is cached reading, far below
    # what recomputing the same tokens would cost
    read_tokens = (8192 - 512) * 1000 * 2   # two later stages, sel=1
    assert (b - a) == pytest.approx(
        read_tokens * cost.read_seconds_per_token(M, D))
    assert (b - a) < read_tokens * cost.token_seconds(M, D)
