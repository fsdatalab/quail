"""Booting a GPU for a query, reusing it, and releasing it for another model."""

import contextlib
import sys
import types
import weakref
from types import SimpleNamespace

import pytest

from quail.backends import resources
from quail.backends.quail import worker
from quail.backends.quail.executor.chunk import InputStaging
from quail.backends.quail.executor.memory import memory_report
from quail.backends.quail.worker import LoadedGpu
from quail.builtins import built_in_registry
from quail.specs import H100_SXM, MODELS


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
                                 current_blas_handle=lambda: 1,
                                 memory_allocated=lambda: 8 * 2**30,
                                 max_memory_allocated=lambda: 20 * 2**30,
                                 reset_peak_memory_stats=lambda: None,
                                 synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional)


@pytest.fixture
def booted(monkeypatch):
    install_fake_torch(monkeypatch)
    monkeypatch.setattr(worker, "set_gpu_index", lambda index: None)
    monkeypatch.setattr(worker, "say", lambda message: None)
    for name in ("load_model", "AnswerRows", "KVArena", "AsyncAnswers"):
        monkeypatch.setattr(worker, name, lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(worker, "build_pipeline",
                        lambda *a, **k: SimpleNamespace(decode_graphs=None))
    monkeypatch.setattr(worker, "KVArena", lambda *a, **k: SimpleNamespace(
        k=[], v=[], gdn_state=None, conv_state=None, num_pages=0))
    monkeypatch.setattr(worker, "warm_kernels",
                        lambda *a, **k: {"tier": "compiled"})


ENVELOPE = {
    "backend": "quail", "model": "qwen3-4b-fp8", "device": "h100-sxm",
    "gpus": 1,
    "settings": {"chunk_tokens": 8192, "true_ids": [1], "false_ids": [2]},
}
PAYLOAD = {"physical_plan": ENVELOPE, "model": "qwen3-4b-fp8", "gpus": 1,
           "documents": {}, **ENVELOPE["settings"]}


def test_arena_resize_releases_graphs_before_reallocating(booted, monkeypatch):
    calls = []
    gpu = LoadedGpu.__new__(LoadedGpu)
    gpu.torch = sys.modules["torch"]
    monkeypatch.setattr(gpu.torch.cuda, "synchronize",
                        lambda: calls.append("synchronize"))
    gpu.pipeline = SimpleNamespace(decode_graphs=SimpleNamespace(
        invalidate=lambda: calls.append("invalidate")))
    gpu.arena = SimpleNamespace(
        num_pages=8, num_sliding_pages=0, k=[], v=[], gdn_state=None,
        conv_state=None, host=None, resize=lambda *a, **kw: calls.append("resize"))
    gpu.model = gpu.F = None
    gpu.decision_head = None
    gpu.host_kv_bytes = 0
    # a dense model keeps no checkpoint slots, so resize gets no state pool
    gpu.spec = MODELS["qwen3-4b-fp8"]
    gpu.device = H100_SXM
    gpu.execution = SimpleNamespace(bind_query=lambda **kw: None)
    gpu.bind_query([1], [2], 32, arena_pages=(8, 0))
    assert calls == ["resize"]
    calls.clear()
    gpu.bind_query([1], [2], 32, arena_pages=(16, 0))
    assert calls == ["synchronize", "invalidate", "resize"]


def test_bind_query_starts_the_host_store_at_the_configured_size(
        booted, monkeypatch):
    stores = []

    class FakeStore:
        def __init__(self, arena, capacity_bytes, torch):
            self.num_rows = capacity_bytes // 10
            self.bytes = capacity_bytes
            self.flushed = False
            stores.append(self)

        def flush(self):
            self.flushed = True

    monkeypatch.setattr(worker, "HostKV", FakeStore)
    gpu = LoadedGpu.__new__(LoadedGpu)
    gpu.torch = sys.modules["torch"]
    gpu.pipeline = SimpleNamespace(decode_graphs=None)
    gpu.arena = SimpleNamespace(
        num_pages=8, num_sliding_pages=0, k=[], v=[], gdn_state=None,
        conv_state=None, host=None, resize=lambda *a, **kw: None)
    gpu.model = gpu.F = None
    gpu.decision_head = None
    gpu.spec = MODELS["qwen3-4b-fp8"]
    gpu.device = H100_SXM
    gpu.host_kv_bytes = 0
    gpu.execution = SimpleNamespace(bind_query=lambda **kw: None)
    gpu.bind_query([1], [2], 32, arena_pages=(8, 0))
    assert gpu.arena.host is None and not stores
    gpu.bind_query([1], [2], 32, arena_pages=(8, 0), host_kv_bytes=1000)
    assert gpu.arena.host is stores[0] and stores[0].bytes == 1000
    # the same size keeps the store and its entries for the next query
    gpu.bind_query([1], [2], 32, arena_pages=(8, 0), host_kv_bytes=1000)
    assert gpu.arena.host is stores[0] and len(stores) == 1
    gpu.bind_query([1], [2], 32, arena_pages=(8, 0))
    assert gpu.arena.host is None and stores[0].flushed


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
                        lambda gpu, *a, **k: {
                            "_outputs": {}, "wall_s": 1.0,
                            **memory_report(gpu)})

    worker.prepare_quail_request(SimpleNamespace(
        gpus=1, registry=registry, runtime_state=runtime_state,
        request=SimpleNamespace(plan=ENVELOPE),
    ))

    gpu = runtime_state["gpu"]
    assert gpu.prepared_boot["kind"] == "cold"
    assert len(boots) == 1

    response = worker.execute_quail_payload(
        PAYLOAD, registry, object(), backend, runtime_state)

    assert response.metrics["boot_kind"] == "cold"
    assert response.metrics["memory_bytes"] == {
        "weights": 8 * 2**30, "kv": 0, "state": 0,
        "other": 12 * 2**30, "peak": 20 * 2**30}
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
        k=[], v=[], gdn_state=None, conv_state=None,
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


def test_a_failed_boot_releases_the_gpu_and_the_next_boot_loads_again(
        booted, monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "release_models",
                        lambda state: calls.append(("release", sorted(state))))
    registry = built_in_registry()
    backend = registry.backend("quail")
    context = worker._single_gpu_context(registry, ENVELOPE)
    state = {}

    def fail(*args, **kwargs):
        raise RuntimeError("warm failed")

    monkeypatch.setattr(worker.LoadedGpu, "warm", fail)
    with pytest.raises(RuntimeError, match="warm failed"):
        worker._boot_for_query(state, backend, context, 100, [1], [2])
    assert calls == [("release", ["gpu"])]
    assert state == {}

    monkeypatch.setattr(worker.LoadedGpu, "warm", lambda self: (0.0, None))
    gpu, boot = worker._boot_for_query(state, backend, context, 100, [1], [2])
    assert state == {"gpu": gpu} and boot["kind"] == "cold"
    again, boot = worker._boot_for_query(state, backend, context, 100, [1], [2])
    assert again is gpu and boot["kind"] == "warm"
    assert calls == [("release", ["gpu"])]


def test_releasing_a_quail_request_stops_its_children_and_frees_cuda(monkeypatch):
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
    monkeypatch.setattr(worker, "_WORKERS",
                        SimpleNamespace(close=lambda: calls.append("children")))
    state = {"gpu": SimpleNamespace(close=lambda: calls.append("close"))}
    worker.release_quail_request(state)
    assert state == {}
    assert worker._WORKERS is None
    assert calls == ["children", "synchronize", "close", "release_parallel_state",
                     "gc", "empty_cache", "ipc_collect"]
