"""HTTP routes of Quail Server, driven with Starlette's test client."""

import hashlib
import io
import json
import threading
import time

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from helpers import server_fakes

pytest.importorskip("starlette")

from quail.server.app import ServerSettings, create_app  # noqa: E402

SETTINGS = dict(models=("qwen3-4b-fp8",), device="h100-sxm",
                hooks=server_fakes.hooks, in_process=True)


def ipc_bytes(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def upload(client) -> str:
    payload = ipc_bytes(server_fakes.reviews_table())
    content_id = hashlib.sha256(payload).hexdigest()
    response = client.put(f"/v1/inputs/{content_id}", content=payload)
    assert response.status_code in (200, 201), response.text
    return content_id


def submission(content_id, **overrides) -> dict:
    reviews = {"kind": "snapshot", "content_id": content_id, "id_col": "id"}
    return {"sql": server_fakes.FILTER_SQL, "dialect": "snowflake",
            "config": server_fakes.CONFIG, "inputs": {"reviews": reviews},
            **overrides}


def wait_done(client, query_id, timeout=30.0) -> dict:
    deadline = time.monotonic() + timeout
    status = client.get(f"/v1/queries/{query_id}").json()
    while status["state"] not in ("succeeded", "failed", "interrupted",
                                  "cancelled"):
        assert time.monotonic() < deadline, status
        status = client.get(
            f"/v1/queries/{query_id}",
            params={"after": status["revision"], "wait": 1}).json()
    return status


def assert_error(response, code, message):
    assert response.status_code == code, response.text
    assert message in response.json()["error"]["message"]


@pytest.fixture()
def make_client(tmp_path):
    from starlette.testclient import TestClient

    clients = []

    def make(**overrides):
        settings = ServerSettings(data_dir=tmp_path / "data",
                                  **{**SETTINGS, **overrides})
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def test_http_lifecycle_events_files_and_durable_submissions(make_client):
    client = make_client()
    server = client.app.state.server
    server.scheduler.stop()
    content_id = upload(client)
    status = client.post("/v1/queries", json=submission(content_id)).json()
    query_id = status["id"]
    assert client.get(f"/v1/queries/{query_id}/result").status_code == 409

    def read_events():
        with client.stream("GET", f"/v1/queries/{query_id}/events") as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            return [json.loads(line[len("data: "):])
                    for line in response.iter_lines() if line.startswith("data: ")]

    seen = {}

    def read_into(index):
        seen[index] = read_events()

    readers = [threading.Thread(target=read_into, args=(i,)) for i in range(2)]
    for thread in readers:
        thread.start()
    time.sleep(0.5)
    server.scheduler.start()
    for thread in readers:
        thread.join(30)
        assert not thread.is_alive()
    final = wait_done(client, query_id)
    for snapshots in seen.values():
        revisions = [s["revision"] for s in snapshots]
        assert revisions[0] == status["revision"]
        assert revisions == sorted(set(revisions)) and len(revisions) >= 3
        assert snapshots[-1]["state"] == "succeeded"
    assert final["state"] == final["phase"]["name"] == "succeeded"
    assert final["result"]["rows"] == 2
    replayed = read_events()
    assert [event["revision"] for event in replayed] == [final["revision"]]

    result = client.get(f"/v1/queries/{query_id}/result")
    assert result.headers["content-type"] == "application/vnd.apache.arrow.stream"
    assert result.headers["x-quail-rows"] == "2"
    with ipc.open_stream(pa.BufferReader(result.content)) as reader:
        table = reader.read_all()
    assert sorted(table.column("r.id").to_pylist()) == ["r0", "r3"]
    answers = client.get(
        f"/v1/queries/{query_id}/files/answers/filters/0.arrow")
    assert answers.status_code == 200
    forbidden = client.get(f"/v1/queries/{query_id}/files/../quail.sqlite3")
    assert forbidden.status_code in (400, 404)
    rows = client.get(f"/v1/queries/{query_id}/rows", params={"limit": 1}).json()
    assert rows["columns"] == ["r.id"] and rows["total_rows"] == 2
    assert len(rows["rows"]) == 1

    fresh = client.post("/v1/queries", json=submission(content_id)).json()
    started = time.monotonic()
    polled = client.get(f"/v1/queries/{fresh['id']}",
                        params={"after": fresh["revision"], "wait": 20}).json()
    assert polled["revision"] > fresh["revision"]
    assert time.monotonic() - started < 15, "a long poll returns on a change"
    wait_done(client, fresh["id"])

    # a submission is made durable before it is acknowledged
    server.scheduler.stop()
    synced = []
    server.add_sync(lambda: synced.append(server.store.list_recent()[0].id))
    accepted = client.post("/v1/queries", json=submission(content_id, query_id="d"))
    assert accepted.status_code == 201
    assert synced == ["d"], "the sync ran after the record was saved"
    client.post("/v1/queries", json=submission(content_id, query_id="d"))
    assert synced == ["d"], "a repeated submission does not sync again"

    def failing_sync():
        raise OSError("volume commit failed")

    server.add_sync(failing_sync)
    refused = client.post("/v1/queries", json=submission(content_id, query_id="e"))
    assert_error(refused, 500, "volume commit failed")
    assert client.get("/v1/queries/e").status_code == 404, "no record was kept"
    assert client.get("/v1/queries/d").status_code == 200
