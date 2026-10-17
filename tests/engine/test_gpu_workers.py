"""GPU worker ownership across local queries, exercised with CPU processes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from helpers.gpu_workers import run

from quail.backends.quail import worker
from quail.builtins import built_in_registry


@pytest.fixture
def execute(monkeypatch):
    monkeypatch.setattr(worker, "_WORKERS", None)
    monkeypatch.setattr(worker, "_child_main", run)
    monkeypatch.setattr(worker, "resolve_model_path", lambda name, revision: name)
    registry = built_in_registry()

    def query(model="qwen3-4b-fp8", value="first", kind="read", gpus=2,
              prepare=False):
        def graph(payload, graph, count, round_fn, *args):
            outputs = round_fn(kind, [value] * count)
            return {"_outputs": {}, "workers": outputs}

        monkeypatch.setattr(worker, "execute_distributed_graph", graph)
        envelope = {
            "model": model, "device": "h100-sxm", "backend": "quail",
            "workers": gpus,
            "settings": {"chunk_tokens": 100, "true_ids": [1], "false_ids": [2]},
        }
        context = SimpleNamespace(
            request=SimpleNamespace(plan=envelope, inputs={}, column_tables=dict),
            graph=SimpleNamespace(nodes=()), registry=registry,
            gpu_count=gpus, runtime_state={})
        backend = registry.backend("quail")
        if prepare:
            backend.prepare_request(context)
            assert worker._WORKERS is not None
            assert worker._WORKERS.round("read", ["ready"] * gpus) == [
                {"model": model, "value": "ready"}] * gpus
        return backend.execute_request(context)

    yield query
    worker._close_children()


def test_model_switch_replaces_gpu_workers(execute):
    first = execute()
    assert first.metrics["workers"] == [
        {"model": "qwen3-4b-fp8", "value": "first"}] * 2
    second = execute(model="qwen3-32b-fp8", value="second")
    assert second.metrics["workers"] == [
        {"model": "qwen3-32b-fp8", "value": "second"}] * 2
    assert second.metrics["boot_kind"] == "cold"


@pytest.mark.parametrize("kind, error", [("fail", RuntimeError), ("exit", EOFError)])
def test_failed_query_cannot_leave_responses_for_the_next_query(execute, kind, error):
    with pytest.raises(error):
        execute(kind=kind)
    result = execute(value="second")
    assert result.metrics["workers"] == [
        {"model": "qwen3-4b-fp8", "value": "second"}] * 2
    assert result.metrics["boot_kind"] == "cold"


def test_repeated_queries_keep_gpu_workers_warm(execute):
    assert execute().metrics["boot_kind"] == "cold"
    result = execute(value="second")
    assert result.metrics["workers"] == [
        {"model": "qwen3-4b-fp8", "value": "second"}] * 2
    assert result.metrics["boot_kind"] == "warm"


def test_prepared_workers_load_before_execution_and_boot_once_per_query(execute):
    first = execute(prepare=True)
    assert first.metrics["boot_kind"] == "cold"
    assert first.metrics["boot"]["boot_count"] == 1

    second = execute(prepare=True)
    assert second.metrics["boot_kind"] == "warm"
    assert second.metrics["boot"]["boot_count"] == 2


def test_dead_worker_is_replaced_before_the_next_query(execute):
    execute()
    process = worker._WORKERS.children[1][0]
    process.kill()
    process.join(5)

    result = execute(value="second")

    assert result.metrics["workers"] == [
        {"model": "qwen3-4b-fp8", "value": "second"}] * 2
    assert result.metrics["boot_kind"] == "cold"


def test_reducing_gpu_count_releases_unused_workers(execute):
    execute(gpus=4)
    previous = [process for process, _ in worker._WORKERS.children]

    result = execute(gpus=2)

    assert result.metrics["workers"] == [
        {"model": "qwen3-4b-fp8", "value": "first"}] * 2
    assert all(not process.is_alive() and process.exitcode == 0
               for process in previous)
    assert len(worker._WORKERS.children) == 2


def test_releasing_quail_stops_its_gpu_workers(execute):
    execute()
    processes = [process for process, _ in worker._WORKERS.children]

    built_in_registry().backend("quail").release_request({})

    assert worker._WORKERS is None
    assert all(not process.is_alive() and process.exitcode == 0
               for process in processes)


@pytest.mark.parametrize("failed_index", [0, 1])
def test_spawn_failure_releases_started_workers_and_pipes(monkeypatch, failed_index):
    children = []

    def pipe():
        process, parent, child = Mock(), Mock(), Mock()
        process.is_alive.return_value = True
        if len(children) == failed_index:
            process.start.side_effect = OSError("cannot spawn worker")
        children.append((process, parent, child))
        return parent, child

    context = SimpleNamespace(Pipe=pipe, Process=lambda **_: children[-1][0])
    monkeypatch.setattr("multiprocessing.get_context", lambda _: context)

    with pytest.raises(OSError, match="cannot spawn worker"):
        worker.GpuWorkers(2, ())

    for index, (process, parent, child) in enumerate(children):
        parent.close.assert_called_once()
        child.close.assert_called_once()
        if index != failed_index:
            process.kill.assert_called_once()
            process.join.assert_called_once()
