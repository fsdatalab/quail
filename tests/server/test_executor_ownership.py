"""A child execution owns its worker; a replacement worker is untouched."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from quail.server.executor import ChildProcessExecutor, Job


@pytest.fixture
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
        made.append(SimpleNamespace(connection=parent, process=process))
        return parent, child

    context = SimpleNamespace(Pipe=pipe, Process=lambda **_: made[-1].process)
    monkeypatch.setattr("multiprocessing.get_context", lambda _: context)
    monkeypatch.setattr("quail.server.executor.threading.Thread", Mock())
    return made


def test_a_stopped_execution_leaves_its_replacement_worker_alone(children):
    executor = ChildProcessExecutor()
    first = Job("first", {}, {"model": "first-model"}, {}, {}, "/unused")
    second = Job("second", {}, {"model": "second-model"}, {}, {}, "/unused")
    events = []
    old = executor.start(first, lambda *event: events.append(event))
    old.stop()
    current = executor.start(second, lambda *event: events.append(event))
    original, replacement = children
    replacement.connection.recv.side_effect = AssertionError("read replacement pipe")
    original.connection.recv.side_effect = EOFError
    old.run()
    assert replacement.process.is_alive()
    replacement.connection.close.assert_not_called()
    assert events == [] and not current.wait(0)
    executor.close()
    assert not replacement.process.is_alive()


def test_a_failed_spawn_or_event_save_leaves_no_worker_behind(children):
    executor = ChildProcessExecutor()
    job = Job("first", {}, {"model": "model"}, {}, {}, "/unused")
    execution = executor.start(job, Mock(side_effect=OSError("cannot save event")))
    children[0].connection.recv.return_value = ("progress", {})
    with pytest.raises(OSError, match="cannot save event"):
        execution.run()
    assert execution.wait(0) and not children[0].process.is_alive()
    children[0].connection.close.assert_called_once()

    children[0].process.start.side_effect = OSError("cannot spawn worker")
    pipe_ends = []
    context = SimpleNamespace(
        Pipe=lambda: pipe_ends.append((Mock(), Mock())) or pipe_ends[-1],
        Process=lambda **_: children[0].process)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("multiprocessing.get_context", lambda _: context)
        with pytest.raises(OSError, match="cannot spawn worker"):
            executor.start(job, Mock())
    for end in pipe_ends[0]:
        end.close.assert_called_once()
    assert executor._child is None
