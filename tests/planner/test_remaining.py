"""CPU checks for time remaining corrected by observed speed."""

import pytest

from quail.cost.remaining import RemainingTime


def test_before_any_work_finishes_the_prediction_stands():
    estimate = RemainingTime(100.0).update(0.0, 0.0)
    assert estimate.seconds == pytest.approx(100.0)
    assert estimate.basis == "predicted"
    assert estimate.upper_seconds > estimate.seconds


def test_a_query_running_twice_as_slow_as_predicted_is_corrected():
    remaining = RemainingTime(100.0)
    for step in range(1, 11):
        estimate = remaining.update(20.0 * step, 10.0 * step)
    assert estimate.basis == "observed"
    assert estimate.seconds == pytest.approx(0.0, abs=1e-6)
    midway = RemainingTime(100.0)
    for step in range(1, 6):
        estimate = midway.update(20.0 * step, 10.0 * step)
    assert estimate.seconds == pytest.approx(100.0, rel=0.1)


def test_the_upper_estimate_narrows_as_work_finishes():
    remaining = RemainingTime(100.0)
    early = remaining.update(2.0, 1.0)
    for step in range(2, 80):
        late = remaining.update(2.0 * step, 1.0 * step)
    assert late.upper_seconds / max(late.seconds, 1e-9) < (
        early.upper_seconds / early.seconds)


def test_a_new_total_changes_the_work_left():
    remaining = RemainingTime(100.0)
    remaining.update(10.0, 10.0)
    estimate = remaining.update(20.0, 20.0, total_seconds=200.0)
    assert estimate.seconds == pytest.approx(180.0, rel=0.05)


def test_a_total_that_is_not_positive_is_refused():
    with pytest.raises(ValueError, match="positive"):
        RemainingTime(0.0)
