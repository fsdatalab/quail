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


def test_kernel_trace_separates_idle_inside_forward_from_other_gpu_work():
    torch = pytest.importorskip("torch")
    from experiments.cells.profile_query import _forward_kernels, _gpu_gaps

    cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
    forward = SimpleNamespace(
        id=1, device_type=cpu, name="quail.forward", cpu_parent=None)
    attention = SimpleNamespace(
        id=2, device_type=cpu, name="quail.attention.shared", cpu_parent=forward)
    launch = SimpleNamespace(
        id=3, device_type=cpu, name="cuLaunchKernel", cpu_parent=attention)
    kernels = [SimpleNamespace(
        device_type=cuda, name="kernel", is_user_annotation=False,
        time_range=SimpleNamespace(start=start, end=end))
        for start, end in [(1, 3), (5, 8), (9, 20)]]
    annotation = SimpleNamespace(
        device_type=cuda, name="quail.forward", is_user_annotation=True,
        time_range=SimpleNamespace(start=1, end=8))
    got = _forward_kernels([forward, attention, launch, annotation, *kernels])
    assert got["passes"] == 1
    assert got["busy_s"] == 5e-6
    assert got["idle_s"] == 2e-6
    assert got["total_kernels"] == 3
    assert got["attention_dispatch"]["shared"]["commands_per_layer"] == 1
    for event in (forward, attention, launch):
        event.time_range = SimpleNamespace(start=0, end=10)
    prof = SimpleNamespace(events=lambda: [forward, attention, launch,
                                         annotation, *kernels])
    window = _gpu_gaps(prof, 10)
    assert window["busy_s"] == 16e-6
    assert [kernel["name"] for kernel in window["kernels"]] == ["kernel"]


def test_attention_work_counts_chunked_prefixes_and_shared_decode_reads():
    from experiments.cells.piecewise_probe import attention_work
    from quail.backends.quail.executor.chunk import DecodeRows

    decode = DecodeRows(positions=[0, 1], write_rows=[], page_table=[], slots=[],
                        shared_cu=[0, 2], shared_used=[100])
    chunk = SimpleNamespace(
        tokens=5, attention_mode="unified", row_spans=[("document", 0, 2, 3)],
        decode_rows=decode, meta=dict(
            unified=dict(cu_q_host=[0, 3, 5], used_host=[7, 6]), shared={}))
    got = attention_work(chunk)
    assert got["causal_pairs"] == 29
    assert got["prefix_pairs"] == 11
    assert got["suffix_pairs"] == 7
    assert got["decode_pairs"] == 11
    assert got["shared_pairs"] == 200

    def prefill(num_tokens, num_keys):
        chunk.tokens = num_tokens
        chunk.row_spans = [("document", 0, num_tokens, num_tokens)]
        chunk.decode_rows = None
        chunk.meta = dict(
            unified=dict(cu_q_host=[0, num_tokens], used_host=[num_keys]), shared=None)
        return attention_work(chunk)["prefix_pairs"]

    assert prefill(17, 17) + prefill(83, 100) == prefill(100, 100) == 5050
    assert prefill(4, 2) == 3


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
