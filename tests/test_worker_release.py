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


def test_booting_a_different_model_releases_the_loaded_one(monkeypatch):
    """A second model cannot load beside the first one's weights and arena."""
    calls = []
    monkeypatch.setattr(
        worker, "release_booted_models",
        lambda state: (calls.append(sorted(state)), state.clear()))

    class _FakeLoaded:
        def __init__(self, backend, context, answer_ids):
            calls.append(("load", context.model.name))

        def bind_query(self, *args):
            calls.append("bind")

        def warm(self):
            return 0.0, None

        load_model_s = arena_s = pipeline_s = 0.0

    monkeypatch.setattr(worker, "LoadedGpu", _FakeLoaded)
    monkeypatch.setattr(worker, "_boot_record",
                        lambda gpu, cold, warm_s, tier, t: {
                            "boot_s": 0.0, "kind": "cold" if cold else "warm"})
    backend = SimpleNamespace(name="quail")
    state = {("quail", "qwen3-4b-fp8"): object()}
    context = SimpleNamespace(model=SimpleNamespace(name="qwen3-reranker"))

    gpu, boot = worker._boot_for_query(state, backend, context, 100, [1], [2])

    assert calls == [[("quail", "qwen3-4b-fp8")], ("load", "qwen3-reranker"),
                     "bind"]
    assert list(state) == [("quail", "qwen3-reranker")] and boot["kind"] == "cold"

    # the same model again is reused, nothing released
    gpu2, boot2 = worker._boot_for_query(state, backend, context, 100, [1], [2])
    assert gpu2 is gpu and boot2["kind"] == "warm" and calls[-1] == "bind"
