"""Quail Server in Rust: a client uploads tables and runs SQL over Arrow Flight."""

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


@pytest.fixture
def client():
    """A Flight client of a server whose workers answer from an oracle."""
    if not SERVER.exists():
        pytest.skip("quail-server is not built; run cargo build --workspace")
    with socket.socket() as found:
        found.bind(("127.0.0.1", 0))
        port = found.getsockname()[1]
    server = subprocess.Popen(
        [str(SERVER), "--model", "qwen3-4b-fp8", "--device", "h100-sxm",
         "--listen", f"127.0.0.1:{port}"],
        env={**os.environ, "QUAIL_WORKER": f"{sys.executable} {WORKER}"})
    try:
        connected = flight.FlightClient(f"grpc://127.0.0.1:{port}")
        deadline = time.monotonic() + 60
        while True:
            try:
                list(connected.list_actions())
                break
            except flight.FlightUnavailableError:
                assert server.poll() is None, "quail-server exited"
                assert time.monotonic() < deadline, "quail-server did not start"
                time.sleep(0.1)
        yield connected
    finally:
        server.terminate()
        server.wait(timeout=10)


def _upload(client, name, table, id_column):
    descriptor = flight.FlightDescriptor.for_path(name, id_column)
    writer, _ = client.do_put(descriptor, table.schema)
    writer.write_table(table)
    writer.close()


def test_a_client_uploads_a_table_and_reads_the_rows_of_an_ai_query(client):
    _upload(client, "reviews", REVIEWS, "id")
    query = ("SELECT r.id FROM reviews r "
             "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) ORDER BY r.id")
    rows = client.do_get(flight.Ticket(query.encode())).read_all()
    # the oracle keeps the documents at even positions
    assert rows.to_pydict() == {"r.id": ["r0", "r2", "r4"]}
    plan, = client.do_action(flight.Action("explain", query.encode()))
    assert "ModelRegionExec" in plan.body.to_pybytes().decode()
    # a query that does not plan is the client's error, and the next runs
    with pytest.raises(pa.ArrowInvalid, match="missing"):
        client.do_get(flight.Ticket(b"SELECT * FROM missing")).read_all()
    count = client.do_get(flight.Ticket(b"SELECT count(*) AS n FROM reviews"))
    assert count.read_all().to_pydict() == {"n": [6]}
