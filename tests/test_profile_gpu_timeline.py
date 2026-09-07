"""Check the GPU timeline's timestamps and lossless encoding."""

import gzip
import json
import struct

import pytest

from experiments.profile_gpu_timeline import encode_intervals, read_gpu_intervals


def test_timeline_clips_to_join_and_merges_concurrent_operations(tmp_path):
    pytest.importorskip("ijson")
    path = tmp_path / "trace.json.gz"
    events = [
        {"cat": "kernel", "ts": 5, "dur": 10},
        {"cat": "gpu_memcpy", "ts": 14, "dur": 4},
        {"cat": "gpu_memset", "ts": 20, "dur": 15},
        {"cat": "cpu_op", "ts": 10, "dur": 20},
    ]
    for event in events:
        event["ph"] = "X"
    with gzip.open(path, "wt") as stream:
        json.dump({"baseTimeNanoseconds": 1_000_000, "traceEvents": events}, stream)
    intervals = read_gpu_intervals(path, (1_010_000, 1_030_000))
    assert intervals == [(0, 8), (10, 20)]
    assert list(struct.iter_unpack("<dd", gzip.decompress(encode_intervals(intervals)))) == intervals


def test_encoding_preserves_submicrosecond_boundaries():
    intervals = [(0.0009765625, 17.0283203125), (1_000_000.25, 1_000_000.5)]
    assert list(struct.iter_unpack("<dd", gzip.decompress(encode_intervals(intervals)))) == intervals
