"""The prior survival curve over answer tokens."""

import numpy as np
import pytest

from quail.cost.answers import prior_survival


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
