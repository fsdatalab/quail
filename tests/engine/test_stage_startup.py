"""CPU regressions for failure before all GPU stage servers connect."""

import json
import multiprocessing as mp
import os
import queue
import signal
import socket
import threading
import time
from types import SimpleNamespace

import _quail_native
import pytest

from quail.backends.quail import graph as staged
from quail.backends.quail.worker import GpuWorkers, RegionServers
from quail.physical import PhysicalGraph, PortRef, Scan


def graph():
    return PhysicalGraph(
        (Scan(node_id="scan:r", alias="r", input_id="r", num_documents=1),),
        PortRef("scan:r", "rows:r"),
    )


@pytest.fixture
def setup(monkeypatch):
    # These tests exercise startup, not compute selection or GPU execution.
    monkeypatch.setattr(staged, "compute_subgraph", lambda value: value)
    monkeypatch.setattr(staged, "finished_rows", lambda: None)
    return {"stages": []}, graph()


def forbidden_serve(*_args):
    pytest.fail("invalid coordinator inputs must not start listeners")


def test_grouping_failure_precedes_listeners(setup, monkeypatch):
    payload, plan = setup
    sentinel = RuntimeError("grouping preparation sentinel")

    def fail(*_args):
        raise sentinel

    monkeypatch.setattr(staged, "stage_groupings", fail)
    with pytest.raises(RuntimeError) as caught:
        staged.execute_staged_graph(payload, plan, 2, forbidden_serve, None)
    assert caught.value is sentinel


@pytest.mark.parametrize("stages", [None, "not stages", {}, [{"key": "r"}]])
def test_invalid_stages_precede_listeners(setup, stages):
    payload, plan = setup
    payload["stages"] = stages
    with pytest.raises(ValueError):
        staged.execute_staged_graph(payload, plan, 2, forbidden_serve, None)


def test_invalid_group_ids_precede_listeners(setup, monkeypatch):
    payload, plan = setup
    monkeypatch.setattr(staged, "stage_groupings", lambda *_: {"r": [["not an id"]]})
    with pytest.raises(TypeError):
        staged.execute_staged_graph(payload, plan, 2, forbidden_serve, None)


def _wait_in_accept(path, connection, ignore_terminate):
    if ignore_terminate:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(path)
        listener.listen()
        connection.send(("ok", None))
        listener.accept()  # No coordinator connects in this regression.


@pytest.mark.parametrize("ignore_terminate", [False, True])
def test_coordinator_failure_aborts_unconnected_children(setup, monkeypatch,
                                                        ignore_terminate):
    payload, plan = setup
    context = mp.get_context("spawn")
    workers = GpuWorkers.__new__(GpuWorkers)
    workers.children = []
    pids = []
    failed_at = []
    sentinel = RuntimeError("coordinator startup sentinel")

    def serve(paths, _wanted):
        for path in paths:
            parent, child = context.Pipe()
            process = context.Process(target=_wait_in_accept,
                                      args=(path, child, ignore_terminate), daemon=True)
            process.start()
            child.close()
            workers.children.append((process, parent))
            pids.append(process.pid)
        for _, connection in workers.children:
            assert connection.poll(10), "CPU listener did not start"
            assert connection.recv() == ("ok", None)
        handle = RegionServers(workers)
        # Cleanup must never try to collect a result from accept().
        handle.finish = lambda: pytest.fail("receive called during abort")
        return handle

    def fail(*_args):
        failed_at.append(time.monotonic())
        raise sentinel

    monkeypatch.setattr(_quail_native, "run_stages", fail)
    try:
        with pytest.raises(RuntimeError) as caught:
            staged.execute_staged_graph(payload, plan, 2, serve, None)
        assert caught.value is sentinel
        assert time.monotonic() - failed_at[0] < 8, "abort did not finish promptly"
        assert not workers.children
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        workers.close(terminate=True)


def test_abort_failure_does_not_replace_coordinator_error(setup, monkeypatch):
    payload, plan = setup
    sentinel = RuntimeError("original coordinator error")

    def fail(*_args):
        raise sentinel

    def abort():
        raise RuntimeError("secondary abort error")

    monkeypatch.setattr(_quail_native, "run_stages", fail)
    with pytest.raises(RuntimeError) as caught:
        staged.execute_staged_graph(
            payload, plan, 2,
            lambda *_: SimpleNamespace(abort=abort, finish=forbidden_serve), None)
    assert caught.value is sentinel


def test_result_failure_also_aborts_owned_servers(setup, monkeypatch):
    payload, plan = setup
    sentinel = RuntimeError("server result sentinel")
    aborted = []

    def finish():
        raise sentinel

    monkeypatch.setattr(_quail_native, "run_stages", lambda *_: None)
    handle = SimpleNamespace(finish=finish, abort=lambda: aborted.append(True))
    with pytest.raises(RuntimeError) as caught:
        staged.execute_staged_graph(payload, plan, 2, lambda *_: handle, None)
    assert caught.value is sentinel
    assert aborted == [True]


def test_partial_socket_failure_ends_peers_before_and_after_bad_path(tmp_path):
    """A failed middle connection must not strand the later listener."""
    paths = [str(tmp_path / name)
             for name in ("first.sock", "missing.sock", "last.sock")]
    received = queue.Queue()
    listeners = []
    threads = []

    def receive(listener):
        try:
            with listener.accept()[0] as stream:
                stream.settimeout(3)
                # The end frame is tiny; make no assumptions about recv boundaries.
                with stream.makefile("rb") as reader:
                    count = reader.read(8)
                    assert len(count) == 8, "coordinator dropped a partial connection"
                    body = reader.read(int.from_bytes(count, "little"))
                    received.put(json.loads(body))
        except BaseException as error:
            received.put(error)

    try:
        for path in (paths[0], paths[2]):
            listener = socket.socket(socket.AF_UNIX)
            listener.settimeout(3)
            listener.bind(path)
            listener.listen()
            listeners.append(listener)
            thread = threading.Thread(target=receive, args=(listener,), daemon=True)
            thread.start()
            threads.append(thread)
        with pytest.raises(RuntimeError, match="missing.sock"):
            _quail_native.run_stages(graph().to_json(), "[]", {}, paths, lambda _: None)
        replies = [received.get(timeout=5) for _ in threads]
        assert replies == [{"end": True}, {"end": True}]
    finally:
        for listener in listeners:
            listener.close()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
