"""Check elapsed-time accounting for nested profiler operations."""

import pytest

from experiments.profile_flamegraph import aggregate_intervals


def test_repeated_calls_keep_parent_context_and_self_time():
    result = aggregate_intervals([
        (0, 10, "parent"), (2, 5, "child"), (6, 8, "child"),
        (12, 17, "parent"), (13, 14, "child"), (18, 20, "other"),
        (18, 19, "child"),
    ], 25)
    parent = next(child for child in result["children"] if child["name"] == "parent")
    assert parent["calls"] == 2
    assert parent["seconds"] == pytest.approx(15e-6)
    assert parent["self_seconds"] == pytest.approx(9e-6)
    assert parent["children"][0]["calls"] == 3
    assert parent["children"][0]["seconds"] == pytest.approx(6e-6)
    assert sum(child["seconds"] for child in result["children"]) == pytest.approx(25e-6)


def test_parent_precedes_child_with_same_start():
    result = aggregate_intervals([(0, 1, "child"), (0, 2, "parent")], 3)
    parent = next(child for child in result["children"] if child["name"] == "parent")
    assert parent["children"][0]["name"] == "child"


def test_crossing_intervals_are_rejected():
    with pytest.raises(ValueError, match="cross rather than nest"):
        aggregate_intervals([(0, 3, "one"), (2, 4, "two")], 5)
