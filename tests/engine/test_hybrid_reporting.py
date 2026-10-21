"""What a hybrid run reports: plan lines, planning refusals, and memory."""

from types import SimpleNamespace

from quail.backends.quail.backend import QuailBackend
from quail.backends.quail.executor.memory import memory_breakdown, merge_memory_reports
from quail.backends.quail.worker import pool_bytes
from quail.explain import architecture_lines
from quail.specs import DEVICES, MODELS


class _Tensor:
    def __init__(self, elements, element_size):
        self._elements, self._size = elements, element_size

    def numel(self):
        return self._elements

    def element_size(self):
        return self._size


def test_hybrid_model_lines_name_layers_precision_and_kernels(monkeypatch):
    monkeypatch.setenv("QUAIL_GDN_PREFILL", "flashinfer")
    architecture, kernels = architecture_lines("qwen3.5-4b-bf16")
    assert architecture == ("architecture=qwen3_5 (24 gated_delta_net, "
                            "8 full_attention), weights=bf16")
    assert "vLLM per-layer GEMM" in kernels
    assert "recurrence=flashinfer (QUAIL_GDN_PREFILL)" in kernels
    assert "FlashAttention over paged KV" in kernels
    architecture, kernels = architecture_lines("qwen3-4b-fp8")
    assert architecture.startswith("architecture=qwen3 (")
    assert architecture.endswith("weights=fp8")
    assert "fp8 GEMM (DeepGEMM)" in kernels and "recurrence" not in kernels
    assert architecture_lines("no-such-model") == []


def test_prefill_backend_support_matches_the_device(monkeypatch):
    monkeypatch.setenv("QUAIL_GDN_PREFILL", "flashinfer")
    backend = QuailBackend()
    model = MODELS["qwen3.5-4b-bf16"]
    blackwell = DEVICES["rtx-pro-6000-blackwell-server"]
    assert backend.supports(model, DEVICES["h100-sxm"], 1).supported
    result = backend.supports(model, blackwell, 1)
    assert not result.supported
    assert "flashinfer needs a Hopper GPU" in result.reason
    monkeypatch.delenv("QUAIL_GDN_PREFILL")
    assert backend.supports(model, blackwell, 1).supported


def test_pool_bytes_and_breakdown_add_up():
    arena = SimpleNamespace(
        k=[_Tensor(10, 2)] * 2, v=[_Tensor(10, 2)] * 2,
        gdn_state=_Tensor(5, 4), conv_state=_Tensor(7, 2))
    pools = pool_bytes(arena)
    assert pools == {"kv": 80, "state": 34}
    parts = memory_breakdown(1_000, pools, 1_500)
    assert parts == {"weights": 1_000, "kv": 80, "state": 34,
                     "other": 386, "peak": 1_500}
    arena.gdn_state = arena.conv_state = None
    assert pool_bytes(arena) == {"kv": 80, "state": 0}
    workers = [dict(
        memory_bytes=parts, memory_capacity={"pages": 16, "slots": 3},
        memory_trace_started=1.0, peak_gib=1.5,
        memory_trace=[[2.0 + i, 4, 1, 2, 3, 100, 90, 10, 2, i]],
        memory_events=[[3.0, "evict", {"tokens": 20}]]) for i in range(2)]
    merged = merge_memory_reports(workers)
    assert merged["memory_capacity"] == {"pages": 32, "slots": 6}
    assert merged["memory_bytes"] == {key: 2 * value for key, value in parts.items()}
    assert merged["memory_trace"][-1] == [3.0, 8, 2, 4, 6, 200, 180, 10, 2, 1]
    assert [event[2]["worker"] for event in merged["memory_events"]] == [0, 1]
    assert merged["memory_workers"] == workers
    assert merge_memory_reports(workers) == merged
