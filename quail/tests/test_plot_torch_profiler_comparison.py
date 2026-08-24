"""Test for plot_torch_profiler_comparison.py's renormalization logic
- the one piece of real arithmetic in that script, and worth locking
down since it's what makes the measured-vs-formula comparison
apples-to-apples (see the function's own docstring for why raw
measured_pct can't be compared to formula_pct directly)."""

import sys
from pathlib import Path

import pytest

REPORTS = Path(__file__).resolve().parents[1] / "reports"
sys.path.insert(0, str(REPORTS))


def test_renormalized_measured_excludes_unmodeled_and_sums_to_100():
    pytest.importorskip("matplotlib")
    import plot_torch_profiler_comparison as m

    row = dict(measured_pct=dict(projection=40.0, elementwise=10.0,
                                 attention=10.0, unmodeled=30.0,
                                 other=10.0))
    out = m.renormalized_measured(row)
    assert set(out) == {"projection", "elementwise", "attention"}
    assert sum(out.values()) == pytest.approx(100.0, abs=1e-9)
    # ratios among the three modeled buckets must be preserved
    assert out["projection"] / out["elementwise"] == pytest.approx(4.0)


def test_renormalized_measured_all_unmodeled_is_zero_not_nan():
    pytest.importorskip("matplotlib")
    import plot_torch_profiler_comparison as m

    row = dict(measured_pct=dict(projection=0.0, elementwise=0.0,
                                 attention=0.0, unmodeled=100.0))
    out = m.renormalized_measured(row)
    assert out == {"projection": 0.0, "elementwise": 0.0, "attention": 0.0}
