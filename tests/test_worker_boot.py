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

    def bind_query(self, *, torch, async_answers, chunk_tokens):
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
    monkeypatch.setattr(worker, "Pipeline", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(worker, "AsyncAnswers",
                        lambda torch, answerer: SimpleNamespace())
    monkeypatch.setattr(worker, "_PayloadAnswerer",
                        lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(worker, "warm_kernels",
                        lambda *a, **k: {"tier": "compiled"})
    monkeypatch.setattr("quail.runtime.volumes.commit_kernel_cache",
                        lambda: None)
    return calls


def test_single_gpu_context_reads_the_plan_envelope():
    registry = built_in_registry()
    envelope = {"workers": 4, "model": "qwen3-4b-fp8", "device": "h100-sxm"}

    context = worker._single_gpu_context(registry, envelope)

    assert context.gpu_index == 0
    assert context.gpu_count == 4
    assert context.model.name == "qwen3-4b-fp8"
    assert context.device.name == "h100-sxm"
    assert dict(context.query_settings) == {}


def test_second_query_reuses_the_loaded_model(booted):
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


def test_gpu_state_carries_what_the_graph_reads(booted):
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
