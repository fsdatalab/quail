"""An execution retains its own worker when a replacement starts."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from quail.server.executor import ChildProcessExecutor, Job


@pytest.fixture()
def children(monkeypatch):
    made = []

    def pipe():
        parent, child = Mock(), Mock()
        process = Mock()
        process.is_alive.return_value = False
        process.start.side_effect = lambda: setattr(
            process.is_alive, "return_value", True)
        process.kill.side_effect = lambda: setattr(
            process.is_alive, "return_value", False)
        parent.send.side_effect = lambda job: (
            setattr(process.is_alive, "return_value", False) if job is None else None)
        made.append(SimpleNamespace(connection=parent, child_connection=child,
                                    process=process))
        return parent, child

    context = SimpleNamespace(Pipe=pipe, Process=lambda **_: made[-1].process)
    monkeypatch.setattr("multiprocessing.get_context", lambda _: context)
    monkeypatch.setattr("quail.server.executor.threading.Thread", Mock())
    return made


@pytest.mark.parametrize("action", ["disconnect", "stop", "model_ready"])
def test_old_execution_cannot_change_its_replacement(children, action):
    executor = ChildProcessExecutor()
    first = Job("first", {}, {"model": "first-model"}, {}, {}, "/unused")
    second = Job("second", {}, {"model": "second-model"}, {}, {}, "/unused")
    events = []
    old = executor.start(first, lambda *event: events.append(event))
    old.stop()
    current = executor.start(second, lambda *event: events.append(event))
    original, replacement = children
    replacement.connection.recv.side_effect = AssertionError("read replacement pipe")
    original.connection.recv.side_effect = (
        [("model_ready", {"model": "first-model"}), ("done", {})]
        if action == "model_ready" else EOFError)
    try:
        if action == "stop":
            old.stop()
        else:
            old.run()
        assert replacement.process.is_alive()
        replacement.connection.close.assert_not_called()
        assert events == []
        assert not current.wait(0)
        again = executor.start(second, lambda *_: None)
        assert again.job.model_phase["name"] == "loading_model"
        assert len(children) == 2
    finally:
        executor.close()


def test_finished_execution_releases_its_worker(children):
    executor = ChildProcessExecutor()
    job = Job("first", {}, {"model": "model"}, {}, {}, "/unused")
    old = executor.start(job, lambda *_: None)
    children[0].connection.recv.return_value = ("done", {})
    old.run()
    assert old.wait(0)
    executor.start(job, lambda *_: None)
    try:
        old.stop()
        assert len(children) == 1
        assert children[0].process.is_alive()
        children[0].connection.close.assert_not_called()
    finally:
        executor.close()


@pytest.mark.parametrize("changes", [
    {"backend": "stock_vllm"},
    {"gpus": 2},
    {"device": "rtx-pro-6000-blackwell-server"},
])
def test_changed_engine_configuration_replaces_worker(children, changes):
    executor = ChildProcessExecutor()
    config = {"model": "qwen3-4b-fp8", "device": "h100-sxm",
              "backend": "quail", "gpus": 1}
    first = Job("first", {}, config, {}, {}, "/unused")
    second = replace(first, id="second", config={**config, **changes})
    execution = executor.start(first, lambda *_: None)
    children[0].connection.recv.side_effect = [
        ("model_ready", {"model": config["model"]}), ("done", {})]
    execution.run()
    try:
        current = executor.start(second, lambda *_: None)
        assert len(children) == 2
        assert not children[0].process.is_alive()
        children[0].connection.close.assert_called_once()
        assert children[1].process.is_alive()
        assert current.job.model_phase["name"] == "loading_model"
        assert children[1].connection.send.call_args.args == (current.job,)
    finally:
        executor.close()


def test_query_settings_reuse_worker(children):
    executor = ChildProcessExecutor()
    config = {"model": "qwen3-4b-fp8", "device": "h100-sxm"}
    first = Job("first", {}, config, {}, {}, "/unused")
    second = replace(first, id="second", config={
        **config, "backend": "quail", "gpus": 1,
        "attention": "unified", "canvas_draws": 8, "gpu_timing": True,
    })
    execution = executor.start(first, lambda *_: None)
    children[0].connection.recv.side_effect = [
        ("model_ready", {"model": config["model"]}), ("done", {})]
    execution.run()
    try:
        current = executor.start(second, lambda *_: None)
        assert len(children) == 1
        assert children[0].process.is_alive()
        assert current.job.model_phase is None
    finally:
        executor.close()


def test_spawn_failure_closes_both_pipe_ends(monkeypatch):
    parent, child, process = Mock(), Mock(), Mock()
    process.start.side_effect = OSError("cannot spawn worker")
    context = SimpleNamespace(Pipe=lambda: (parent, child), Process=lambda **_: process)
    monkeypatch.setattr("multiprocessing.get_context", lambda _: context)
    executor = ChildProcessExecutor()
    with pytest.raises(OSError, match="cannot spawn worker"):
        executor.start(Job("first", {}, {"model": "model"}, {}, {}, "/unused"), Mock())
    parent.close.assert_called_once()
    child.close.assert_called_once()
    assert executor._child is None


def test_event_delivery_failure_stops_its_worker(children):
    executor = ChildProcessExecutor()
    emit = Mock(side_effect=OSError("cannot save event"))
    execution = executor.start(
        Job("first", {}, {"model": "model"}, {}, {}, "/unused"), emit)
    children[0].connection.recv.return_value = ("progress", {})
    try:
        with pytest.raises(OSError, match="cannot save event"):
            execution.run()
        assert execution.wait(0)
        assert not children[0].process.is_alive()
        children[0].connection.close.assert_called_once()
    finally:
        executor.close()
