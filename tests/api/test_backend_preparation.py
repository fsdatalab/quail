"""Baseline startup finishes before readiness callbacks and query timing."""

import sys
from types import SimpleNamespace

import pytest
from helpers.baselines import CAPACITY, _Client, _session

import quail
from quail.api import execute
from quail.backends import resources
from quail.baselines.vllm import VLLMEngine


@pytest.mark.parametrize("fails", [False, True])
def test_baseline_preparation_precedes_readiness_and_query_time(monkeypatch, fails):
    events = []
    clock = [0.0]
    state = execute._BackendState()
    monkeypatch.setattr(execute, "_BACKEND_STATE", state)
    monkeypatch.setattr(execute, "gpu_problem", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: False, is_initialized=lambda: False)))
    monkeypatch.setattr(resources, "_release_vllm_parallel_state", lambda: None)

    class Client(_Client):
        def generate(self, prompts, sampling_params, use_tqdm=False):
            events.append("query")
            clock[0] += 2.0
            return super().generate(prompts, sampling_params, use_tqdm)

        def close(self):
            pass

    def boot(self, model, allowed_ids, *, resources):
        events.append("boot")
        clock[0] += 5.0
        if fails:
            raise RuntimeError("cannot load model")
        clock[0] += 6.0
        return ({"client": Client(), "sampling_params": object(),
                 "capacity": CAPACITY},
                {"kind": "cold", "llm_init_s": 5.0, "boot_s": 5.0})

    monkeypatch.setattr(VLLMEngine, "boot", boot)
    with _session("stock_vllm", docs={"id": ["a"], "body": ["one"]}) as session:
        query = (session.docs("docs").alias("d")
                 .ai_filter(quail.prompt("useful {0}", quail.col("d.body")))
                 .select("d.id"))
        query.plan()
        query.wait_for_tokens()
        monkeypatch.setattr(execute.time, "perf_counter", lambda: clock[0])

        def run():
            return execute.execute_query(
                query, on_backend_ready=lambda: events.append("ready"),
                on_execution_ready=lambda: events.append("executing"))

        try:
            if fails:
                with pytest.raises(RuntimeError, match="cannot load model"):
                    run()
                assert events == ["boot"]
            else:
                first = run()
                assert events == ["boot", "ready", "executing", "query"]
                assert first.report["boot_s"] == 11.0
                assert first.report["boot"]["llm_init_s"] == 5.0
                assert first.report["boot_kind"] == "cold"
                assert first.report["coordinator_wall_s"] == 2.0
                events.clear()
                second = run()
                assert events == ["ready", "executing", "query"]
                assert second.report["boot_s"] == 0.0
                assert second.report["boot_kind"] == "warm"
                assert second.report["coordinator_wall_s"] == 2.0
        finally:
            state.close()
