"""Check clock alignment and thread selection for the comparison windows."""

import gzip
import json

import pytest

from experiments.analyze_blog_profiles import analyze


def test_midpoint_window_uses_gpu_union_and_main_cpu_thread(tmp_path):
    pytest.importorskip("ijson")
    phase = {"started_unix_ns": 1_000_000_000, "finished_unix_ns": 21_000_000_000}
    (tmp_path / "phase.json").write_text(json.dumps(phase))
    events = [
        ("kernel", "kernel", 0, 8, 0),
        ("gpu_memcpy", "copy", 7, 2, 0),
        ("kernel", "kernel", 11, 11, 0),
        ("cuda_runtime", "cudaLaunchKernel", 7.5, 0.1, 7),
        ("cuda_runtime", "cudaLaunchKernel", 8, 0.1, 7),
        ("cuda_runtime", "other thread", 0, 20, 99),
        ("user_annotation", "quail.backends.quail.executor.loop.pack_chunk", 7, 5, 7),
    ]
    trace = {"baseTimeNanoseconds": 1_000_000_000, "traceEvents": [
        {"ph": "X", "cat": category, "name": name, "ts": start * 1e6,
         "dur": duration * 1e6, "tid": thread}
        for category, name, start, duration, thread in events
    ]}
    with gzip.open(tmp_path / "worker.trace.json.gz", "wt") as stream:
        json.dump(trace, stream)
    result = analyze(tmp_path, "/results/example/phase.json")
    assert result["duration_s"] == 20
    assert result["gpu_active_s"] == 18
    assert result["window_gpu_active_s"] == 3
    assert result["window"]["thread_id"] == 7
    assert result["window"]["gpu"] == [(7, 9), (11, 12)]
    assert result["window"]["cpu"] == [
        [7, 12, 0, "quail.backends.quail.executor.loop.pack_chunk"],
        [7.5, 7.6, 1, "cudaLaunchKernel"], [8, 8.1, 1, "cudaLaunchKernel"],
    ]
