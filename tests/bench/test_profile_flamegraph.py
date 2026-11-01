"""Check elapsed-time accounting for nested profiler operations."""

import pytest

from experiments.profile_flamegraph import aggregate_intervals


def test_flamegraph_accounts_for_nested_calls_and_gpu_overlap():
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

    with pytest.raises(ValueError, match="cross rather than nest"):
        aggregate_intervals([(0, 3, "one"), (2, 4, "two")], 5)

    result = aggregate_intervals(
        [(0, 10, "parent"), (2, 5, "child"), (12, 17, "parent")],
        20, [(1, 4), (3, 6), (9, 13), (18, 22)],
    )
    parent = next(child for child in result["children"] if child["name"] == "parent")
    gap = next(child for child in result["children"] if child["calls"] == 0)
    assert result["gpu_seconds"] == pytest.approx(11e-6)
    assert result["gpu_idle_seconds"] == pytest.approx(9e-6)
    assert parent["gpu_seconds"] == pytest.approx(7e-6)
    assert parent["self_gpu_seconds"] == pytest.approx(4e-6)
    assert parent["children"][0]["gpu_seconds"] == pytest.approx(3e-6)
    assert gap["gpu_seconds"] == pytest.approx(4e-6)
    assert gap["gpu_idle_seconds"] == pytest.approx(1e-6)
    assert sum(child["gpu_seconds"]
               for child in result["children"]) == pytest.approx(11e-6)
