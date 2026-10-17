"""Concurrent local queries share the process's GPU engine in sequence."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from types import SimpleNamespace

import pytest
from helpers.baselines import CAPACITY, _Client, _session

import quail
from quail.api import execute
from quail.backends import resources
from quail.baselines import request
from quail.baselines.vllm import VLLMEngine


@pytest.mark.parametrize("phase", ["boot", "execution"])
@pytest.mark.parametrize("second_backend", ["stock_vllm", "dumb_vllm"])
def test_local_queries_keep_engine_until_execution_finishes(
        monkeypatch, phase, second_backend):
    state = execute._BackendState()
    monkeypatch.setattr(execute, "_BACKEND_STATE", state)
    monkeypatch.setattr(execute, "gpu_problem", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: False, is_available=lambda: False)))
    monkeypatch.setattr(resources, "_release_vllm_parallel_state", lambda: None)
    entered = threading.Event()
    resume = threading.Event()
    second_started = threading.Event()
    live = []
    boots = []
    executions = []

    def wait():
        entered.set()
        assert resume.wait(5)

    class Client(_Client):
        def close(self):
            live.remove(self)

    def boot(self, model, allowed_ids, *, resources):
        client = Client()
        live.append(client)
        resources.callback(client.close)
        boots.append(self.kind)
        if phase == "boot" and len(boots) == 1:
            wait()
        return ({"client": client, "sampling_params": object(),
                 "capacity": CAPACITY}, {"kind": "cold"})

    run_graph = request.execute_request_graph

    def execute_graph(context, backend, settings, boot):
        executions.append(backend.engine.kind)
        if phase == "execution" and len(executions) == 1:
            wait()
        assert live == [settings["client"]]
        return run_graph(context, backend, settings, boot)

    monkeypatch.setattr(VLLMEngine, "boot", boot)
    monkeypatch.setattr(request, "execute_request_graph", execute_graph)
    with (_session("stock_vllm", docs={"id": ["a"], "body": ["one"]}) as first,
          _session(second_backend, docs={"id": ["b"], "body": ["two"]}) as second):
        queries = [(session.docs("docs").alias("d")
                    .ai_filter(quail.prompt("useful {0}", quail.col("d.body")))
                    .select("d.id")) for session in (first, second)]
        for query in queries:
            query.plan()
            query.wait_for_tokens()

        def second_run():
            second_started.set()
            return queries[1].run()

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first_result = pool.submit(queries[0].run)
                try:
                    assert entered.wait(5)
                    second_result = pool.submit(second_run)
                    assert second_started.wait(5)
                    with pytest.raises(TimeoutError):
                        second_result.result(timeout=0.25)
                finally:
                    resume.set()
                first_result.result(timeout=5).collect()
                second_result.result(timeout=5).collect()
            assert len(boots) == (1 if second_backend == "stock_vllm" else 2)
            assert len(executions) == 2
        finally:
            state.close()
    assert not live
