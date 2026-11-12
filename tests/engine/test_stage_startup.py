"""CPU tests of how a region starts and reaches its GPU stage servers."""

import json
import multiprocessing as mp
import os
import queue
import signal
import socket
import sys
import threading
import time
from types import SimpleNamespace

import _quail_native
import pytest
from test_kv_reads import named

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


def test_invalid_inputs_fail_before_any_listener_starts(setup, monkeypatch):
    payload, plan = setup
    for stages in [None, "not stages", {}, [{"key": "r"}]]:
        with pytest.raises(ValueError):
            staged.execute_staged_graph(dict(payload, stages=stages), plan, 2,
                                        forbidden_serve, None)
            pytest.fail(f"stages {stages!r} were accepted")

    # a failure while grouping is raised as it is
    sentinel = RuntimeError("grouping preparation sentinel")

    def fail(*_args):
        raise sentinel

    monkeypatch.setattr(staged, "stage_groupings", fail)
    with pytest.raises(RuntimeError) as caught:
        staged.execute_staged_graph(payload, plan, 2, forbidden_serve, None)
    assert caught.value is sentinel

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


def check_coordinator_failure_aborts_unconnected_children(setup, monkeypatch,
                                                         ignore_terminate):
    payload, plan = setup
    context = mp.get_context("spawn")
    workers = GpuWorkers.__new__(GpuWorkers)
    workers.children = []
    pids = []
    failed_at = []
    sentinel = RuntimeError("coordinator startup sentinel")

    def serve(paths, _wanted, _hinted):
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


def test_coordinator_failure_aborts_unconnected_children(setup, monkeypatch):
    for ignore_terminate in [False, True]:
        named(
            f"ignore_terminate={ignore_terminate}",
            check_coordinator_failure_aborts_unconnected_children,
            setup=setup, monkeypatch=monkeypatch, ignore_terminate=ignore_terminate)


def test_a_coordinator_or_result_failure_aborts_the_servers_and_keeps_its_error(
        setup, monkeypatch):
    # a failing abort does not replace the coordinator's error
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

    # a failing result also aborts the servers
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
            _quail_native.run_stages(graph().to_json(), "[]", {}, paths,
                                         lambda _: None, lambda _: "null")
        replies = [received.get(timeout=5) for _ in threads]
        assert replies == [{"end": True}, {"end": True}]
    finally:
        for listener in listeners:
            listener.close()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()


def _read_shared(message, connection):
    import numpy as np

    from quail.backends.quail.shared import opened

    found = opened(message)
    table = found["table"]
    with open("/proc/self/maps") as maps:
        mapped = [line.split()[0].split("-") for line in maps
                  if line.rstrip().endswith("shared (deleted)")
                  or line.rstrip().endswith("shared")]
    address = table.column("id").chunk(0).buffers()[1].address
    connection.send((
        table.column("id").to_pylist(), found["count"],
        int(np.asarray(found["array"]).sum()),
        any(int(start, 16) <= address < int(end, 16) for start, end in mapped)))


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc/self/maps")
def test_another_process_reads_a_shared_value_over_the_files_mapped_pages(tmp_path):
    import numpy as np
    import pyarrow as pa

    from quail.backends.quail.shared import share

    value = {"table": pa.table({"id": pa.array(range(1000), pa.int32())}),
             "array": np.arange(100), "count": 3}
    message = share(value, str(tmp_path / "shared"))
    # the message names the file; the buffers are not in it
    assert len(message) < 1000
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_read_shared, args=(message, child))
    process.start()
    try:
        ids, count, total, in_file = parent.recv()
    finally:
        process.join(timeout=30)
    assert (ids, count, total) == (list(range(1000)), 3, 4950)
    assert in_file
