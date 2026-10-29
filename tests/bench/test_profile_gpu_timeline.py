"""Check the GPU timeline's timestamps and lossless encoding."""

import gzip
import json
import struct
from types import SimpleNamespace

import pytest

from experiments.profile_gpu_timeline import encode_intervals, read_gpu_intervals


def test_query_timeline_records_forward_token_counts():
    from experiments.cells.profile_query import Timeline

    timeline = Timeline.__new__(Timeline)
    timeline.active = True
    timeline.entries = []
    owner = SimpleNamespace(forward=lambda pipeline, arena, chunk: "output")
    timeline._wrap(owner, "forward", "forward")
    assert owner.forward(None, None, SimpleNamespace(tokens=37)) == "output"
    assert timeline.entries[-1][-1] == 37


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
    assert list(struct.iter_unpack(
        "<dd", gzip.decompress(encode_intervals(intervals)))) == intervals
