"""Local requests release the previous engine before loading another one."""

import sys
from contextlib import ExitStack, closing
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pytest

from quail.api import execute
from quail.backends import resources
from quail.backends.quail import worker
from quail.baselines import request as request_backend
from quail.baselines.vllm import VLLMClient, VLLMEngine
from quail.baselines.vllm_decision import VLLMDecisionClient
from quail.builtins import built_in_registry
from quail.execution.types import PhysicalRequest, PhysicalResponse, document_input
from quail.physical import PhysicalGraph, PortRef, Scan, plan_envelope


@pytest.mark.parametrize("changes", [
    {"backend": "test.second"},
    {"model": "qwen3-32b-fp8"},
    {"workers": 2},
    {"device": "rtx-pro-6000-blackwell-server"},
])
def test_local_requests_release_previous_engine_before_loading(monkeypatch, changes):
    monkeypatch.setattr(execute, "_BACKEND_STATE", execute._BackendState())
    live = []
    events = []

    class Backend:
        def __init__(self, name):
            self.name = name

        def prepare_request(self, context):
            state = context.runtime_state
            if "engine" not in state:
                assert not live
                config = tuple(context.request.plan[name] for name in (
                    "backend", "model", "device", "workers"))
                live.append(config)
                state["engine"] = config
                events.append(("load", config))

        def execute_request(self, context):
            self.prepare_request(context)
            return PhysicalResponse(
                {}, {"configuration": context.runtime_state["engine"]})

        def release_request(self, state):
            events.append(("release", live.pop()))
            state.clear()

    registry = built_in_registry()
    for name in ("test.first", "test.second"):
        registry.register_backend(Backend(name))
    graph = PhysicalGraph(
        (Scan(node_id="input:d", alias="d", input_id="d", n_docs=1),),
        PortRef("input:d", "ids:d"))
    config = {"backend": "test.first", "model": "qwen3-4b-fp8",
              "device": "h100-sxm", "workers": 1}

    def run(configuration):
        envelope = plan_envelope(**configuration, graph=graph, codecs=registry.codecs)
        plan = SimpleNamespace(
            **configuration, graph=graph, to_envelope=lambda codecs: envelope)
        execute._prepare_backend(plan, registry)
        response = execute._execute_physical(PhysicalRequest(
            envelope, {"d": document_input(pa.array([[1]]))}), registry)
        assert response.metrics["configuration"] == tuple(configuration.values())

    run(config)
    run(config)
    run({**config, **changes})
    run(config)
    assert [kind for kind, _ in events] == [
        "load", "release", "load", "release", "load"]
    execute._BACKEND_STATE.close()
    assert not live


def test_switching_quail_and_vllm_releases_each_engine(monkeypatch):
    state = execute._BackendState()
    monkeypatch.setattr(execute, "_BACKEND_STATE", state)
    monkeypatch.setattr(worker, "_WORKERS", None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: False)))
    monkeypatch.setattr(resources, "_release_vllm_parallel_state", lambda: None)
    live = []
    events = []

    def load(name):
        assert not live
        live.append(name)
        events.append(("load", name))

    def close(name):
        assert live.pop() == name
        events.append(("close", name))

    class Gpu:
        def load(self, *args, **kwargs):
            load("quail")

        def bind_query(self, *args):
            pass

        def warm(self):
            return 0.0, None

        def close(self):
            close("quail")

    def boot(self, model, allowed_ids, *, resources):
        load("vllm")
        engine = SimpleNamespace(engine_core=SimpleNamespace(
            shutdown=lambda: close("vllm")))
        client = resources.enter_context(closing(
            VLLMClient(SimpleNamespace(llm_engine=engine), {})))
        return {"client": client}, {"kind": "cold", "boot_s": 0.0}

    monkeypatch.setattr(worker, "LoadedGpu", Gpu)
    monkeypatch.setattr(worker, "_boot_record", lambda *args: {
        "kind": "cold", "boot_s": 0.0})
    monkeypatch.setattr(VLLMEngine, "boot", boot)
    monkeypatch.setattr(request_backend, "execute_request_graph", lambda *args:
                        PhysicalResponse({}, {"boot": args[-1]["kind"]}))
    registry = built_in_registry()
    graph = PhysicalGraph(
        (Scan(node_id="input:d", alias="d", input_id="d", n_docs=1),),
        PortRef("input:d", "ids:d"))

    def prepare(backend):
        envelope = plan_envelope(
            backend=backend, model="qwen3-4b-fp8", device="h100-sxm", workers=1,
            graph=graph, codecs=registry.codecs,
            settings={"true_ids": [1], "false_ids": [2], "chunk_tokens": 100})
        execute._prepare_backend(SimpleNamespace(
            backend=backend, to_envelope=lambda codecs: envelope,
            graph=graph, workers=1), registry)
        return PhysicalRequest(envelope, {"d": document_input(pa.array([[1]]))})

    try:
        prepare("quail")
        prepare("quail")
        baseline = prepare("stock_vllm")
        assert execute._execute_physical(baseline, registry).metrics["boot"] == "cold"
        assert execute._execute_physical(baseline, registry).metrics["boot"] == "warm"
        pipelined = prepare("pipelined_vllm")
        assert execute._execute_physical(pipelined, registry).metrics["boot"] == "warm"
        prepare("quail")
    finally:
        state.close()
    assert events == [(kind, name) for name in ("quail", "vllm", "quail")
                      for kind in ("load", "close")]


def test_baseline_cleanup_does_not_initialize_cuda_in_parent(monkeypatch):
    from quail.baselines.request import LoadedRequestEngine
    from quail.baselines.sglang import SGLangClient, pipelined_sglang_backend

    cuda = Mock()
    cuda.is_initialized.return_value = False
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(resources, "_release_vllm_parallel_state", lambda: None)
    engine = Mock()
    cleanup = ExitStack()
    client = cleanup.enter_context(closing(SGLangClient(engine, {})))
    state = {"engine": LoadedRequestEngine({"client": client}, cleanup)}

    pipelined_sglang_backend().release_request(state)

    assert not state
    engine.shutdown.assert_called_once()
    cuda.synchronize.assert_not_called()
    cuda.empty_cache.assert_not_called()
    cuda.ipc_collect.assert_not_called()


@pytest.mark.parametrize("client_type", [VLLMClient, VLLMDecisionClient])
def test_vllm_cleanup_stops_generative_and_decision_engines(monkeypatch, client_type):
    from quail.baselines.request import LoadedRequestEngine

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: False)))
    monkeypatch.setattr(resources, "_release_vllm_parallel_state", lambda: None)
    core = Mock()
    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=core))
    cleanup = ExitStack()
    client = cleanup.enter_context(closing(client_type(llm, {})))
    state = {"engine": LoadedRequestEngine({"client": client}, cleanup)}

    built_in_registry().backend("stock_vllm").release_request(state)

    assert not state
    core.shutdown.assert_called_once()
