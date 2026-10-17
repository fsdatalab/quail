"""Booting a GPU for a query, reusing it, and releasing it for another model."""

import contextlib
import gc
import sys
import types
import weakref
from types import SimpleNamespace

import pytest

from quail.backends import resources
from quail.backends.quail import worker
from quail.backends.quail.executor.chunk import InputStaging
from quail.backends.quail.worker import LoadedGpu
from quail.builtins import built_in_registry


def install_fake_torch(monkeypatch):
    functional = types.ModuleType("torch.nn.functional")
    functional.linear = lambda a, b: a
    nn = types.ModuleType("torch.nn")
    nn.functional = functional
    torch = types.ModuleType("torch")
    torch.nn = nn
    torch.bfloat16 = "bfloat16"
    torch.ones = lambda *args, **kwargs: SimpleNamespace()
    torch.inference_mode = contextlib.nullcontext
    torch.cuda = SimpleNamespace(mem_get_info=lambda: (40 * 2**30, 80 * 2**30),
                                 is_initialized=lambda: False,
                                 current_blas_handle=lambda: 1,
                                 synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional)


@pytest.fixture
def booted(monkeypatch):
    install_fake_torch(monkeypatch)
    monkeypatch.setattr(resources, "_release_vllm_parallel_state", lambda: None)
    monkeypatch.setattr(worker, "set_gpu_index", lambda index: None)
    monkeypatch.setattr(worker, "say", lambda message: None)
    for name in ("load_model", "build_pipeline", "AnswerRows", "KVArena",
                 "AsyncAnswers"):
        monkeypatch.setattr(worker, name, lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(worker, "warm_kernels",
                        lambda *a, **k: {"tier": "compiled"})


ENVELOPE = {
    "backend": "quail", "model": "qwen3-4b-fp8", "device": "h100-sxm",
    "workers": 1,
    "settings": {"chunk_tokens": 8192, "true_ids": [1], "false_ids": [2]},
}
PAYLOAD = {"physical_plan": ENVELOPE, "model": "qwen3-4b-fp8", "workers": 1,
           "docs": {}, **ENVELOPE["settings"]}


def test_arena_resize_releases_graphs_before_reallocating(booted, monkeypatch):
    calls = []
    gpu = LoadedGpu.__new__(LoadedGpu)
    gpu.torch = sys.modules["torch"]
    monkeypatch.setattr(gpu.torch.cuda, "synchronize",
                        lambda: calls.append("synchronize"))
    gpu.pipeline = SimpleNamespace(decode_graphs=SimpleNamespace(
        invalidate=lambda: calls.append("invalidate")))
    gpu.arena = SimpleNamespace(
        n_pages=8, n_sliding_pages=0,
        resize=lambda *a, **kw: calls.append("resize"))
    gpu.model = gpu.F = None
    gpu.decision_head = None
    gpu.execution = SimpleNamespace(bind_query=lambda **kw: None)
    gpu.bind_query([1], [2], 32, arena_pages=(8, 0))
    assert calls == ["resize"]
    calls.clear()
    gpu.bind_query([1], [2], 32, arena_pages=(16, 0))
    assert calls == ["synchronize", "invalidate", "resize"]


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

    gpu = runtime_state["gpu"]
    assert gpu.prepared_boot["kind"] == "cold"
    assert len(boots) == 1

    response = worker.execute_quail_payload(
        PAYLOAD, registry, object(), backend, runtime_state)

    assert response.metrics["boot_kind"] == "cold"
    assert gpu.prepared_boot is None
    assert len(boots) == 1

    execution = gpu.execution
    loaded = execution.loaded_model
    loaded.label_readout = object()
    readout = loaded.label_readout
    staging = loaded.input_staging = InputStaging(gpu.torch)
    buffers = staging.buffers
    buffers["tokens"] = object()
    staging.fixed_tokens[1] = object()
    query = execution.query
    query.async_scores = SimpleNamespace(rows=query.answer_rows)
    query.gpu_timing = True
    old_query = weakref.ref(query)
    reranker = execution._reranker_execution({"d": [[1]]})
    old_reranker = weakref.ref(reranker)
    del query, reranker

    second = worker.execute_quail_payload(
        PAYLOAD, registry, object(), backend, runtime_state)

    assert second.metrics["boot_kind"] == "warm"
    assert old_query() is None and old_reranker() is None
    assert execution.query.loaded_model is loaded
    assert execution.query.async_answers is gpu.async_ans
    assert execution.query.async_scores is None
    assert execution.query.gpu_timing is False
    assert loaded.label_readout is readout
    assert loaded.input_staging is staging
    assert staging.buffers is buffers and "tokens" in buffers
    assert not staging.fixed_tokens
    assert len(boots) == 2
    assert runtime_state["gpu"] is gpu
    resized = []
    gpu.arena = SimpleNamespace(
        resize=lambda *pages, **kw: resized.append((pages, kw)))
    gpu.bind_query([1], [2], 4096, arena_pages=(8, 2))
    assert resized == [((8, 2), {"free_resident": True})]
    execution.bind_loaded_model(model=object(), arena=gpu.arena,
                                pipeline=gpu.pipeline)
    assert execution.loaded_model is not loaded
    assert execution.loaded_model.label_readout is None
    assert execution.loaded_model.input_staging is None
    with pytest.raises(RuntimeError, match="no bound query"):
        execution.query
    replacement = weakref.ref(execution.loaded_model)
    execution.close()
    assert replacement() is None


def test_release_clears_cuda_state(monkeypatch):
    calls = []
    cuda = SimpleNamespace(
        is_initialized=lambda: True,
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(resources.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(resources, "_release_vllm_parallel_state",
                        lambda: calls.append("release_parallel_state"))
    gpu = LoadedGpu.__new__(LoadedGpu)
    gpu.close = lambda: calls.append("close")
    booted = {("quail", "test"): gpu}
    resources.release_models(booted)
    assert booted == {}
    assert calls == ["synchronize", "close", "release_parallel_state", "gc",
                     "empty_cache", "ipc_collect"]


def test_release_drops_gpu_references_before_emptying_cuda_cache(monkeypatch):
    gpu = LoadedGpu.__new__(LoadedGpu)
    gpu.execution = SimpleNamespace(close=lambda: None)
    gpu_ref = weakref.ref(gpu)
    state = {("quail", "qwen3-4b-fp8"): gpu}
    del gpu
    live_at_cache_release = []
    cuda = SimpleNamespace(
        is_initialized=lambda: True,
        synchronize=lambda: None,
        empty_cache=lambda: live_at_cache_release.append(gpu_ref() is not None),
        ipc_collect=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(resources, "_release_vllm_parallel_state", lambda: None)

    resources.release_models(state)

    assert live_at_cache_release == [False]
    assert gpu_ref() is None


@pytest.mark.parametrize("failure", ["cublas", "bind", "warm", "cancel"])
def test_failed_boot_releases_allocations_and_can_retry(booted, monkeypatch, failure):
    class Allocation:
        pass

    allocations = []

    def allocate(*args, **kwargs):
        allocation = Allocation()
        allocations.append(weakref.ref(allocation))
        return allocation

    def fail(*args, **kwargs):
        if failure == "cancel":
            raise KeyboardInterrupt("startup cancelled")
        raise RuntimeError("warmup failed")

    for name in ("load_model", "KVArena", "build_pipeline", "AnswerRows",
                 "AsyncAnswers"):
        monkeypatch.setattr(worker, name, allocate)
    cuda = sys.modules["torch"].cuda
    monkeypatch.setattr(cuda, "is_initialized", lambda: True, raising=False)
    live_at_release = []
    monkeypatch.setattr(cuda, "empty_cache",
                        lambda: live_at_release.append(
                            any(reference() is not None for reference in allocations)),
                        raising=False)
    monkeypatch.setattr(cuda, "ipc_collect", lambda: None, raising=False)
    monkeypatch.setattr(resources, "_release_vllm_parallel_state", lambda: None)
    registry = built_in_registry()
    state = {}
    with monkeypatch.context() as startup:
        if failure == "cublas":
            startup.setattr(cuda, "current_blas_handle", fail)
        else:
            startup.setattr(LoadedGpu, "bind_query" if failure == "bind" else "warm",
                            fail)
        with pytest.raises(KeyboardInterrupt if failure == "cancel" else RuntimeError
                           ) as saved:
            worker._boot_for_query(
                state, registry.backend("quail"),
                worker._single_gpu_context(registry, ENVELOPE), 8192, [1], [2])
    gc.collect()
    assert saved.value.__traceback__ is not None
    assert not state
    assert allocations
    assert all(reference() is None for reference in allocations)
    assert live_at_release == [False]

    gpu, cold = worker._boot_for_query(
        state, registry.backend("quail"),
        worker._single_gpu_context(registry, ENVELOPE), 8192, [1], [2])
    reused, warm = worker._boot_for_query(
        state, registry.backend("quail"),
        worker._single_gpu_context(registry, ENVELOPE), 8192, [1], [2])
    assert reused is gpu
    assert cold["kind"] == "cold"
    assert warm["kind"] == "warm"
    resources.release_models(state)
    assert all(reference() is None for reference in allocations)
    assert live_at_release == [False, False]


def test_worker_boot_reuses_model_and_releases_failed_preparation(booted, monkeypatch):
    registry = built_in_registry()
    child = worker.ChildState(gpu_index=1, registry=registry,
                              model_path="/snapshots/qwen")
    paths = []

    def load(path, **kwargs):
        paths.append((path, kwargs["revision"]))
        return SimpleNamespace()

    monkeypatch.setattr(worker, "load_model", load)
    sub = {**PAYLOAD, "workers": 2}
    assert worker._child_boot(child, sub)["kind"] == "cold"
    first = child.gpu
    assert worker._child_boot(child, sub)["kind"] == "warm"
    assert child.gpu is first
    assert first.execution.context.gpu_index == 1
    assert first.execution.context.gpu_count == 2
    assert paths == [("/snapshots/qwen", None)]

    def fail(self):
        raise RuntimeError("warmup failed")

    with monkeypatch.context() as startup:
        startup.setattr(LoadedGpu, "warm", fail)
        with pytest.raises(RuntimeError, match="warmup failed"):
            worker._child_boot(child, sub)
    assert child.gpu is None
    assert child.runtime_context is None
    assert first.model is first.arena is first.pipeline is first.execution is None
    assert worker._child_boot(child, sub)["kind"] == "cold"
    assert child.gpu is not first
    assert paths == [("/snapshots/qwen", None)] * 2
    resources.release_models({"gpu": child.gpu})
