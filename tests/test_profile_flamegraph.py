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


def test_gpu_overlap_counts_concurrent_kernels_once_and_covers_unrecorded_time():
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
    assert sum(child["gpu_seconds"] for child in result["children"]) == pytest.approx(11e-6)


def test_reader_aligns_gpu_and_cpu_clocks_and_ignores_other_cpu_threads(tmp_path):
    import gzip
    import json

    pytest.importorskip("ijson")
    from experiments.profile_flamegraph import read_cpu_flamegraph

    path = tmp_path / "trace.json.gz"
    payload = {"baseTimeNanoseconds": 1_000_000, "traceEvents": [
        {"ph": "X", "cat": "kernel", "tid": 1, "name": "kernel", "ts": 0, "dur": 15},
        {"ph": "X", "cat": "gpu_memcpy", "tid": 2, "name": "copy", "ts": 14, "dur": 10},
        {"ph": "X", "cat": "cpu_op", "tid": 92, "name": "cpu", "ts": 12, "dur": 10},
        {"ph": "X", "cat": "cpu_op", "tid": 93, "name": "other", "ts": 10, "dur": 20},
    ]}
    with gzip.open(path, "wt") as stream:
        json.dump(payload, stream)
    result = read_cpu_flamegraph(path, (1_010_000, 1_030_000), 92)
    assert result["seconds"] == pytest.approx(20e-6)
    assert result["gpu_seconds"] == pytest.approx(14e-6)
    assert {child["name"] for child in result["children"]} == {"cpu", "[no recorded CPU operation]"}
    gap = next(child for child in result["children"] if child["calls"] == 0)
    assert gap["gpu_seconds"] == pytest.approx(4e-6)
