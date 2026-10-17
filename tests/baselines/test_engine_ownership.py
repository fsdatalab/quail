"""A request backend owns its engine throughout startup and warm reuse."""

import gc
import os
import sys
import weakref
from types import SimpleNamespace

import pytest

from quail.backends import resources
from quail.baselines import sglang, vllm, vllm_decision
from quail.baselines.request import RequestBackend
from quail.builtins import built_in_registry


@pytest.mark.parametrize("kind", ["vllm", "decision", "sglang"])
@pytest.mark.parametrize("failure", ["capacity", "warmup", "cancel"])
def test_failed_startup_closes_engine_before_retry(monkeypatch, kind, failure):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    live = []
    references = []
    closed = []
    cache_releases = []
    fail = [True]
    registry = built_in_registry()
    model = next(spec for spec in registry.models.values()
                 if spec.role == "decision") if kind == "decision" else registry.model(
                     "qwen3-4b-fp8")

    class Engine:
        def get_server_info(self):
            return {}

    def construct(**kwargs):
        if kind != "sglang":
            assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
        engine = Engine()
        references.append(weakref.ref(engine))
        live.append(engine)

        def shutdown():
            live.remove(engine)
            closed.append(True)

        engine.shutdown = shutdown
        engine.llm_engine = SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=shutdown),
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(enable_prefix_caching=True),
                scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
                model_config=SimpleNamespace(tokenizer_mode="gigatoken")))
        return engine

    def capacity(config):
        if fail[0] and failure == "capacity":
            raise RuntimeError("capacity failed")
        return {"kv_cache_size_tokens": 100, "block_size": 1, "max_num_seqs": 8}

    def generate(self, *args, **kwargs):
        if fail[0] and failure != "capacity":
            if failure == "cancel":
                raise KeyboardInterrupt
            raise RuntimeError("warmup failed")
        return [SimpleNamespace(outputs=SimpleNamespace(data=[0.1, 0.9]))]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=construct, SamplingParams=lambda **kw: SimpleNamespace(
            **({"logprobs": None} | kw))))
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=construct))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: True,
            synchronize=lambda: None,
            empty_cache=lambda: cache_releases.append(
                any(reference() is not None for reference in references)),
            ipc_collect=lambda: None)))
    monkeypatch.setattr(resources, "_release_vllm_parallel_state", lambda: None)
    monkeypatch.setattr(vllm, "resolve_model_path", lambda *args: "/model")
    monkeypatch.setattr(sglang, "resolve_model_path", lambda *args: "/model")
    monkeypatch.setattr("quail.specs.files.resolve_model_path", lambda *args: "/model")
    monkeypatch.setattr("quail.backends.quail.executor.model_loading.engine_args",
                        lambda *args: {"model": "/model"})
    monkeypatch.setattr(vllm, "_capacity", capacity)
    monkeypatch.setattr(sglang, "_capacity", capacity)
    monkeypatch.setattr(vllm_decision, "pooling_params", lambda *args: object())
    for client in (vllm.VLLMClient, vllm_decision.VLLMDecisionClient,
                   sglang.SGLangClient):
        monkeypatch.setattr(client, "generate", generate)

    backend = RequestBackend("test", sglang.SGLangEngine() if kind == "sglang"
                             else vllm.VLLMEngine(), "pipelined")
    context = SimpleNamespace(
        registry=registry, runtime_state={}, request=SimpleNamespace(plan={
            "model": model.name, "settings": {"true_ids": [1], "false_ids": [2]}}))
    error = KeyboardInterrupt if failure == "cancel" else RuntimeError
    with pytest.raises(error) as saved:
        backend.prepare_request(context)
    assert not live
    assert len(closed) == 1
    assert context.runtime_state == {}
    gc.collect()
    assert saved.value.__traceback__ is not None
    assert references[0]() is None
    assert cache_releases == [False]

    fail[0] = False
    backend.prepare_request(context)
    assert len(live) == 1
    assert context.runtime_state["engine"].prepared_boot["kind"] == "cold"
    backend.prepare_request(context)
    assert len(live) == 1
    assert context.runtime_state["engine"].prepared_boot["kind"] == "warm"
    backend.release_request(context.runtime_state)
    assert not live
    assert len(closed) == 2
    gc.collect()
    assert all(reference() is None for reference in references)
    assert cache_releases == [False, False]
