"""Booting a GPU for a query, and reusing it for the next one."""

import contextlib
import sys
import types
from types import SimpleNamespace

import pytest

from quail.backends.quail import worker
from quail.builtins import built_in_registry


def install_fake_torch(monkeypatch):
    """Put a GPU-free torch in sys.modules for the duration of a test."""
    functional = types.ModuleType("torch.nn.functional")
    functional.linear = lambda a, b: a
    nn = types.ModuleType("torch.nn")
    nn.functional = functional
    torch = types.ModuleType("torch")
    torch.nn = nn
    torch.bfloat16 = "bfloat16"
    torch.ones = lambda *args, **kwargs: SimpleNamespace()
    torch.inference_mode = contextlib.nullcontext
    torch.cuda = SimpleNamespace(
        mem_get_info=lambda: (40 * 2**30, 80 * 2**30),
        current_blas_handle=lambda: 1,
        synchronize=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional)


class FakeExecution:
    def __init__(self, calls):
        self.calls = calls

    def bind_loaded_model(self, *, model, arena, pipeline):
        self.calls.append("bind_loaded_model")

    def bind_query(self, *, torch, async_answers, answer_rows, chunk_tokens):
        self.calls.append(f"bind_query:{chunk_tokens}")


class FakeBackend:
    name = "quail"

    def __init__(self, calls):
        self.calls = calls

    def start(self, context):
        self.calls.append(f"start:gpu{context.gpu_index}")
        return FakeExecution(self.calls)


@pytest.fixture
def booted(monkeypatch):
    install_fake_torch(monkeypatch)
    calls = []
    monkeypatch.setattr(worker, "set_gpu_index", lambda index: None)
    monkeypatch.setattr(worker, "say", lambda message: None)
    monkeypatch.setattr(worker, "load_model",
                        lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(worker, "KVArena", lambda **k: SimpleNamespace())
    monkeypatch.setattr(worker, "build_pipeline",
                        lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(worker, "AsyncAnswers",
                        lambda torch, rows: SimpleNamespace())
    monkeypatch.setattr(worker, "AnswerRows",
                        lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(worker, "warm_kernels",
                        lambda *a, **k: {"tier": "compiled"})
    return calls


def test_gpu_context_model_reuse_and_graph_state(booted):
    registry = built_in_registry()
    envelope = {"workers": 4, "model": "qwen3-4b-fp8", "device": "h100-sxm"}

    context = worker._single_gpu_context(registry, envelope)

    assert context.gpu_index == 0
    assert context.gpu_count == 4
    assert context.model.name == "qwen3-4b-fp8"
    assert context.device.name == "h100-sxm"
    assert dict(context.query_settings) == {}

    calls = booted
    registry = built_in_registry()
    backend = FakeBackend(calls)
    context = worker._single_gpu_context(registry, {
        "workers": 1, "model": "qwen3-4b-fp8", "device": "h100-sxm",
    })
    runtime_state = {}

    gpu, boot = worker._boot_for_query(
        runtime_state, backend, context, 8192, [1, 2], [3, 4])

    assert boot["kind"] == "cold"
    assert boot["warm_tier"] == "compiled"
    assert gpu.chunk_tokens == 8192
    assert calls == ["start:gpu0", "bind_loaded_model", "bind_query:8192"]

    calls.clear()
    again, second = worker._boot_for_query(
        runtime_state, backend, context, 4096, [1, 2], [3, 4])

    assert again is gpu
    assert second["kind"] == "warm"
    assert second["load_model_s"] == 0.0
    assert second["warm_kernels_s"] == 0.0
    assert again.chunk_tokens == 4096
    assert calls == ["bind_query:4096"]

    registry = built_in_registry()
    context = worker._single_gpu_context(registry, {
        "workers": 1, "model": "qwen3-4b-fp8", "device": "h100-sxm",
    })
    gpu, _ = worker._boot_for_query(
        {}, FakeBackend(booted), context, 8192, [1], [2])

    state = worker._gpu_state(gpu)

    assert state["model_execution"] is gpu.execution
    assert state["arena"] is gpu.arena
    assert state["pipeline"] is gpu.pipeline
    assert state["spec"].name == "qwen3-4b-fp8"


ENVELOPE = {
    "backend": "quail", "model": "qwen3-4b-fp8", "device": "h100-sxm",
    "workers": 1,
    "settings": {"chunk_tokens": 8192, "true_ids": [1], "false_ids": [2]},
}


def payload_for(envelope):
    """Build the flattened payload execute_quail_payload expects."""
    return {
        "physical_plan": envelope, "model": envelope["model"],
        "workers": envelope["workers"], "docs": {},
        **dict(envelope["settings"]),
    }


def test_prepared_boot_is_handed_to_the_query_and_used_once(
        booted, monkeypatch):
    registry = built_in_registry()
    backend = registry.backend("quail")
    runtime_state = {}
    boots = []
    real_boot_for_query = worker._boot_for_query

    def counting_boot(*args, **kwargs):
        boots.append(1)
        return real_boot_for_query(*args, **kwargs)

    monkeypatch.setattr(worker, "_boot_for_query", counting_boot)
    monkeypatch.setattr(worker, "execute_single",
                        lambda *a, **k: {"_outputs": {}, "wall_s": 1.0})

    worker.prepare_quail_request(SimpleNamespace(
        gpu_count=1, registry=registry, runtime_state=runtime_state,
        request=SimpleNamespace(plan=ENVELOPE),
    ))

    gpu = runtime_state[("quail", "qwen3-4b-fp8")]
    assert gpu.prepared_boot["kind"] == "cold"
    assert len(boots) == 1

    response = worker.execute_quail_payload(
        payload_for(ENVELOPE), registry, object(), backend, runtime_state)

    assert response.metrics["boot_kind"] == "cold"
    assert gpu.prepared_boot is None
    assert len(boots) == 1

    second = worker.execute_quail_payload(
        payload_for(ENVELOPE), registry, object(), backend, runtime_state)

    assert second.metrics["boot_kind"] == "warm"
    assert len(boots) == 2
    assert runtime_state[("quail", "qwen3-4b-fp8")] is gpu


def test_bind_query_resizes_the_arena_to_the_plan_split(booted):
    calls = booted
    backend = FakeBackend(calls)
    registry = built_in_registry()
    context = worker._single_gpu_context(registry, {
        "workers": 1, "model": "qwen3-4b-fp8", "device": "h100-sxm",
    })
    gpu, _ = worker._boot_for_query({}, backend, context, 8192, [1], [2])
    resized = []
    gpu.arena = SimpleNamespace(
        resize=lambda *pages, **kw: resized.append((pages, kw)))
    gpu.bind_query([1], [2], 4096, arena_pages=(8, 2))
    assert resized == [((8, 2), {"free_resident": True})]
