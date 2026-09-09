"""Checks for releasing Quail before another GPU engine starts."""

import sys
from types import SimpleNamespace

from quail.backends.quail import worker
from quail.backends.quail.worker import LoadedGpu


class _MockGpu(LoadedGpu):
    """Minimal stand-in that skips real GPU init."""

    def __init__(self, close_fn):
        self._close_fn = close_fn

    def close(self):
        self._close_fn()


def test_release_booted_models_clears_state_and_cuda_cache(monkeypatch):
    calls = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
        memory_allocated=lambda: 123,
        memory_reserved=lambda: 456,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(worker.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(
        worker,
        "_release_vllm_parallel_state",
        lambda: calls.append("release_parallel_state"),
    )
    mock_gpu = _MockGpu(close_fn=lambda: calls.append("close"))
    booted = {("quail", "test"): mock_gpu}

    result = worker.release_booted_models(booted)

    assert booted == {}
    assert calls == [
        "synchronize",
        "close",
        "release_parallel_state",
        "gc",
        "empty_cache",
        "ipc_collect",
    ]
    assert result == {
        "models_released": 1,
        "vllm_parallel_state_released": True,
        "cuda_allocated_bytes": 123,
        "cuda_reserved_bytes": 456,
    }
