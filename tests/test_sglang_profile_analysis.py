import gzip
import json

import pytest

from experiments.sglang_profile_analysis import (
    binned_activity,
    clip_interval,
    intersect_intervals,
    interval_duration,
    merge_intervals,
    read_trace,
)


def test_gpu_activity_counts_overlapping_streams_once():
    intervals = [(10, 20), (5, 15), (30, 40), (40, 45), (8, 9), (0, 0)]
    assert merge_intervals(intervals) == [(5, 20), (30, 45)]
    assert interval_duration(intervals) == 30


def test_activity_bins_preserve_union_duration_and_partial_final_bin():
    intervals = [(5, 15), (10, 20), (30, 43)]
    bins = binned_activity(intervals, 0, 45, 10)
    assert bins == [(0, 10, 5), (10, 20, 10), (20, 30, 0), (30, 40, 10), (40, 45, 3)]
    assert sum(occupied for _, _, occupied in bins) == pytest.approx(interval_duration(intervals))


def test_measurement_window_excludes_export_and_clips_crossing_operations():
    window = (10, 50)
    intervals = [(0, 5), (5, 15), (20, 30), (40, 60), (60, 80)]
    clipped = [span for item in intervals if (span := clip_interval(item, window))]
    assert clipped == [(10, 15), (20, 30), (40, 50)]
    assert interval_duration(clipped) == 25


def test_cpu_scope_overlap_with_gpu_gaps_uses_elapsed_time():
    assert intersect_intervals([(0, 10), (8, 20), (30, 50)], [(5, 15), (18, 35)]) == [
        (5, 15), (18, 20), (30, 35),
    ]


def test_trace_alignment_excludes_events_outside_the_driver_join(tmp_path):
    pytest.importorskip("ijson")
    base_ns = 1_788_000_000_000_000_000
    events = [
        {"cat": "Trace", "name": "PyTorch Profiler (0)", "ts": 0, "dur": 100},
        {"cat": "kernel", "name": "matmul", "ts": 5, "dur": 10},
        {"cat": "kernel", "name": "matmul", "ts": 40, "dur": 20},
        {"cat": "kernel", "name": "export_only", "ts": 60, "dur": 10},
        {"cat": "user_annotation", "name": "scheduler.run_batch", "ts": 0, "dur": 80},
    ]
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump({
            "baseTimeNanoseconds": base_ns,
            "traceEvents": [{"ph": "X", **event} for event in events],
        }, stream)
    summary = read_trace(path, (base_ns + 10_000, base_ns + 50_000))
    assert summary["raw_capture_us"] == 100
    assert summary["capture_us"] == 40
    assert summary["gpu_busy_us"] == 15
    assert summary["scope_us"]["scheduler.run_batch"] == 40
    assert summary["scope_gpu_idle_us"]["scheduler.run_batch"] == 25
    assert "export_only" not in summary["kernels"]
