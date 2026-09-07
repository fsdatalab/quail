"""Check chronological CPU nesting and clock alignment."""

import gzip
import json

import pytest

from experiments.profile_cpu_timeline import nest_intervals, read_cpu_window


def test_nesting_preserves_call_order_and_fills_only_top_level_gaps():
    events = [(3, 5, "parent"), (3.5, 4, "child"), (1, 2, "parent"), (8, 12, "tail")]
    assert nest_intervals(events, 0, 10) == [
        [0, 1, 0, "[no recorded CPU operation]"], [1, 2, 0, "parent"],
        [2, 3, 0, "[no recorded CPU operation]"], [3, 5, 0, "parent"],
        [3.5, 4, 1, "child"], [5, 8, 0, "[no recorded CPU operation]"],
        [8, 10, 0, "tail"],
    ]


def test_crossing_intervals_are_rejected():
    with pytest.raises(ValueError, match="cross"):
        nest_intervals([(1, 3, "a"), (2, 4, "b")], 0, 5)


def test_reader_aligns_to_join_and_selects_recorded_worker_operations(tmp_path):
    pytest.importorskip("ijson")
    events = [
        {"cat": "user_annotation", "name": "vllm.scheduler.schedule", "ts": 1_000_010, "dur": 500_000},
        {"cat": "cpu_op", "name": "aten::item", "ts": 1_100_010, "dur": 100_000},
        {"cat": "kernel", "name": "gpu", "ts": 1_000_010, "dur": 500_000},
        {"cat": "user_annotation", "name": "broad_marker", "ts": 10, "dur": 3_000_000},
        {"cat": "cpu_op", "name": "other_thread", "ts": 10, "dur": 3_000_000, "tid": 9},
    ]
    for event in events:
        event.update(ph="X", tid=event.get("tid", 92))
    path = tmp_path / "trace.gz"
    with gzip.open(path, "wt") as stream:
        json.dump({"baseTimeNanoseconds": 1_000_000, "traceEvents": events}, stream)
    result = read_cpu_window(path, [1_010_000, 3_001_010_000], 92, 1, 2)
    assert result["cpu"] == [
        [1, 1.5, 0, "vllm.scheduler.schedule"], [1.1, 1.2, 1, "aten::item"],
        [1.5, 2, 0, "[no recorded CPU operation]"],
    ]
