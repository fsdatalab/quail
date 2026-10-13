import gzip
import json

import pytest

from experiments.sglang_profile_analysis import (
    binned_activity,
    input_preparation_breakdown,
    intersect_intervals,
    interval_duration,
    read_trace,
)


def test_profile_time_accounting():
    intervals = [(5, 15), (10, 20), (30, 43)]
    bins = binned_activity(intervals, 0, 45, 10)
    assert bins == [(0, 10, 5), (10, 20, 10), (20, 30, 0), (30, 40, 10), (40, 45, 3)]
    assert sum(occupied for _, _, occupied in bins) == pytest.approx(
        interval_duration(intervals))

    assert intersect_intervals([(0, 10), (8, 20), (30, 50)], [(5, 15), (18, 35)]) == [
        (5, 15), (18, 20), (30, 35),
    ]

    base = 1_788_000_000_000_000_000
    trace = {"base_ns": base, "start_us": 0, "gpu_busy_intervals": [(10e6, 12e6)]}
    events = [
        ("normalize_batch_and_arguments", 0, 1),
        ("_batch_tokenize_and_process", 2, 6),
        ("_send_batch_request", 6, 8),
        ("_dispatch_to_scheduler", 6, 8),
        ("normalize_batch_and_arguments", 11, 12),
    ]
    join = {"input_preparation": {"intervals": [
        {"name": f"sglang.input.{name}", "start_unix_ns": base + start * 10**9,
         "end_unix_ns": base + end * 10**9}
        for name, start, end in events
    ]}}
    assert input_preparation_breakdown(join, trace) == {
        "normalize_s": 1, "prepare_s": 4, "send_s": 2, "first_gpu_s": 10, "other_s": 3,
    }


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
