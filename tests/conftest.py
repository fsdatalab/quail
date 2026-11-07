"""Put the project root on sys.path so tests run without an install.

Every test also checks, in shadow, that each model region it binds for
planning derives the same operators from its optimized DataFusion plan
as the binder builds, or the same refusal.
"""

import sys
from pathlib import Path

import _quail_native
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SHADOW = {"num_comparisons": 0, "num_refusals": 0, "regions": set()}


@pytest.fixture(autouse=True)
def _derived_operators_match_the_binders():
    """Fail a test whose regions derive operators the binder does not build."""
    _quail_native.set_shadow_check(True)
    yield
    num_comparisons, num_refusals, regions, differences = (
        _quail_native.take_shadow_log())
    _quail_native.set_shadow_check(False)
    _SHADOW["num_comparisons"] += num_comparisons
    _SHADOW["num_refusals"] += num_refusals
    _SHADOW["regions"].update(regions)
    assert not differences, "\n\n".join(differences)


def pytest_terminal_summary(terminalreporter):
    """Report how many regions the shadow check compared."""
    terminalreporter.write_line(
        f"derived operators: {len(_SHADOW['regions'])} distinct regions compared "
        f"{_SHADOW['num_comparisons']} times, "
        f"{_SHADOW['num_refusals']} of them refusals")
