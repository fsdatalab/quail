"""Survival curves over answer tokens, and the maps' attention path."""

import numpy as np
import pytest

from quail.cost.answers import expected_branches, prior_survival


def test_prior_survival_has_the_expected_mean_below_the_limit():
    curve = prior_survival(64, 10_000)
    assert curve[0] == 1.0 and curve[-1] == 0.0
    # the mean length is the sum of the survival curve
    assert curve.sum() == pytest.approx(64, rel=1e-3)
    assert np.all(np.diff(curve) <= 0)


def test_prior_survival_ends_at_the_limit():
    curve = prior_survival(64, 32)
    assert len(curve) == 33 and curve[32] == 0.0
    assert curve.sum() < 32


def test_expected_branches_counts_the_answers_still_written():
    # answers that always run to their limits: 3 for 32 passes, then 2,
    # then 1, over 256 passes in all
    curves = [np.r_[np.ones(limit), 0.0] for limit in (64, 256, 32)]
    assert expected_branches(curves) == pytest.approx((64 + 256 + 32) / 256)
    assert expected_branches([curves[0]] * 3) == pytest.approx(3.0)
