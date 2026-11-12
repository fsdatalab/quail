"""Quail Server in Rust: a client uploads tables and runs SQL over Arrow Flight."""

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.flight as flight
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SERVER = REPOSITORY / "target" / "debug" / "quail-server"
WORKER = REPOSITORY / "tests" / "helpers" / "oracle_worker.py"

REVIEWS = pa.table({
    "id": [f"r{index}" for index in range(6)],
    "body": [f"review {index} " + "pad " * index for index in range(6)],
})
QUERY = ("SELECT r.id FROM reviews r "
         "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) ORDER BY r.id")

pytestmark = pytest.mark.skipif(
    not SERVER.exists(),
    reason="quail-server is not built; run cargo build --workspace")


@contextlib.contextmanager
def serving(data, delay=0.0):
    """A Flight client of a server on `data` whose workers answer from an oracle.

    Args:
        data: The server's data directory.
        delay: Seconds each filter answer takes.
    """
    with socket.socket() as found:
        found.bind(("127.0.0.1", 0))
        port = found.getsockname()[1]
    server = subprocess.Popen(
        [str(SERVER), "--model", "qwen3-4b-fp8", "--device", "h100-sxm",
         "--listen", f"127.0.0.1:{port}", "--data", str(data)],
        env={**os.environ, "QUAIL_WORKER": f"{sys.executable} {WORKER}",
             "QUAIL_ORACLE_DELAY": str(delay)})
    try:
        client = flight.FlightClient(f"grpc://127.0.0.1:{port}")
        deadline = time.monotonic() + 60
        while True:
            try:
                list(client.list_actions())
                break
            except flight.FlightUnavailableError:
                assert server.poll() is None, "quail-server exited"
                assert time.monotonic() < deadline, "quail-server did not start"
                time.sleep(0.1)
        descriptor = flight.FlightDescriptor.for_path("reviews", "id")
        writer, _ = client.do_put(descriptor, REVIEWS.schema)
        writer.write_table(REVIEWS)
        writer.close()
        yield client
    finally:
        server.terminate()
        server.wait(timeout=10)


def _act(client, name, body) -> dict | list:
    text = body if isinstance(body, str) else json.dumps(body)
    answer, = client.do_action(flight.Action(name, text.encode()))
    return json.loads(answer.body.to_pybytes())


def _finished(client, submitted, seconds=30) -> dict:
    """Wait for a submission to finish; return its status."""
    deadline = time.monotonic() + seconds
    while True:
        status = _act(client, "status", submitted)
        if status["state"] not in ("queued", "running"):
            return status
        assert time.monotonic() < deadline, status
        time.sleep(0.05)


def _rows(client, ticket) -> dict:
    return client.do_get(flight.Ticket(ticket.encode())).read_all().to_pydict()


def test_a_client_uploads_a_table_and_reads_the_rows_of_an_ai_query(tmp_path):
    with serving(tmp_path) as client:
        # the oracle keeps the documents at even positions
        assert _rows(client, QUERY) == {"r.id": ["r0", "r2", "r4"]}
        plan, = client.do_action(flight.Action("explain", QUERY.encode()))
        assert "ModelRegionExec" in plan.body.to_pybytes().decode()
        # a query that does not plan is the client's error, and the next runs
        with pytest.raises(pa.ArrowInvalid, match="missing"):
            _rows(client, "SELECT * FROM missing")
        assert _rows(client, "SELECT count(*) AS n FROM reviews") == {"n": [6]}


def test_a_submission_keeps_its_status_and_rows_when_the_server_restarts(tmp_path):
    with serving(tmp_path) as client:
        submitted = _act(client, "submit", {"sql": QUERY, "id": "first"})
        assert submitted["state"] in ("queued", "running", "succeeded")
        status = _finished(client, "first")
        assert (status["state"], status["num_rows"]) == ("succeeded", 3)
        assert _rows(client, '{"query_id": "first"}') == {"r.id": ["r0", "r2", "r4"]}
        # the same id runs the same query once, and names no other query
        assert _act(client, "submit", {"sql": QUERY, "id": "first"}) == status
        with pytest.raises(pa.ArrowInvalid, match="another query"):
            _act(client, "submit", {"sql": "SELECT 1", "id": "first"})
        failed = _act(client, "submit", {"sql": "SELECT * FROM missing"})
        assert _finished(client, failed["id"])["state"] == "failed"
    with serving(tmp_path) as client:
        found = _act(client, "list", "")
        assert [(one["id"], one["state"]) for one in found] == [
            ("first", "succeeded"), (failed["id"], "failed")]
        assert _rows(client, '{"query_id": "first"}') == {"r.id": ["r0", "r2", "r4"]}


def test_a_timeout_or_a_cancel_stops_a_query_and_the_next_one_runs(tmp_path):
    # six answers of 0.5 s each: the query takes about 3 s
    with serving(tmp_path, delay=0.5) as client:
        slow = _act(client, "submit", {"sql": QUERY, "timeout_s": 0.5})
        waiting = _act(client, "submit", {"sql": QUERY, "id": "waiting"})
        running = _act(client, "submit", {"sql": QUERY, "id": "running"})
        # a waiting submission that is cancelled never runs
        assert _act(client, "cancel", "waiting")["state"] == "cancelled"
        status = _finished(client, slow["id"])
        assert status["state"] == "failed" and "timed out" in status["error"]
        # the next query runs on workers Quail started again
        deadline = time.monotonic() + 30
        while _act(client, "status", "running")["state"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.05)
        _act(client, "cancel", "running")
        assert _finished(client, "running")["state"] == "cancelled"
        assert waiting["state"] == "queued" and running["state"] == "queued"
        last = _act(client, "submit", {"sql": "SELECT count(*) AS n FROM reviews"})
        assert _finished(client, last["id"])["num_rows"] == 1
