"""HTTP routes of the query service, driven with Starlette's test client."""

import hashlib
import io
import json
import threading
import time

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import service_fakes

from quail.service import inputs

pytest.importorskip("starlette")

from quail.service.app import ServiceSettings, create_app  # noqa: E402

SETTINGS = dict(models=("qwen3-4b-fp8",), device="h100-sxm",
                hooks=service_fakes.hooks, in_process=True)


def ipc_bytes(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def upload(client, table=None) -> str:
    payload = ipc_bytes(table if table is not None
                        else service_fakes.reviews_table())
    content_id = hashlib.sha256(payload).hexdigest()
    response = client.put(f"/v1/inputs/{content_id}", content=payload)
    assert response.status_code in (200, 201), response.text
    return content_id


def submission(content_id, **overrides) -> dict:
    body = {
        "sql": service_fakes.FILTER_SQL,
        "dialect": "snowflake",
        "config": service_fakes.CONFIG,
        "inputs": {"reviews": {"kind": "snapshot", "content_id": content_id,
                               "id_col": "id"}},
    }
    body.update(overrides)
    return body


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


@pytest.fixture()
def make_client(tmp_path):
    from starlette.testclient import TestClient

    clients = []

    def make(**overrides):
        settings = ServiceSettings(data_dir=tmp_path / "data",
                                   **{**SETTINGS, **overrides})
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def test_capabilities_and_config_checks(make_client):
    client = make_client()
    capabilities = client.get("/v1/capabilities").json()
    assert capabilities["models"] == ["qwen3-4b-fp8"]
    assert capabilities["default_timeout_s"] == 1000.0
    content_id = upload(client)
    for config, message in [
        ({**service_fakes.CONFIG, "model": "qwen3-32b-fp8"}, "does not run model"),
        ({**service_fakes.CONFIG, "device": "rtx-pro-6000-blackwell-server"},
         "runs on device"),
        ({**service_fakes.CONFIG, "gpus": 2}, "GPUs"),
        ({**service_fakes.CONFIG, "backend": "stock_vllm"}, "backends"),
        ({"model": "qwen3-4b-fp8"}, "bad engine config"),
    ]:
        response = client.post("/v1/queries",
                               json=submission(content_id, config=config))
        assert response.status_code == 400, response.text
        assert message in response.json()["error"]["message"]


def test_uploads_are_checked_and_saved_once(make_client):
    client = make_client(max_upload_bytes=4096)
    payload = ipc_bytes(service_fakes.reviews_table())
    content_id = hashlib.sha256(payload).hexdigest()
    assert client.head(f"/v1/inputs/{content_id}").status_code == 404
    bad = client.put("/v1/inputs/notahash", content=payload)
    assert bad.status_code == 400
    mismatch = client.put(f"/v1/inputs/{'0' * 64}", content=payload)
    assert mismatch.status_code == 400
    assert "hash" in mismatch.json()["error"]["message"]
    text = b"not arrow"
    not_arrow = client.put(
        f"/v1/inputs/{hashlib.sha256(text).hexdigest()}", content=text)
    assert not_arrow.status_code == 400
    assert "Arrow IPC" in not_arrow.json()["error"]["message"]
    big = ipc_bytes(pa.table({"id": ["x" * 100] * 100}))
    too_big = client.put(f"/v1/inputs/{hashlib.sha256(big).hexdigest()}",
                         content=big)
    assert too_big.status_code == 413
    assert client.put(f"/v1/inputs/{content_id}",
                      content=payload).status_code == 201
    assert client.put(f"/v1/inputs/{content_id}",
                      content=payload).status_code == 200
    assert client.head(f"/v1/inputs/{content_id}").status_code == 200
    service = client.app.state.service
    assert not list(service.inputs_dir.glob("*.tmp"))
    assert service.store.get_input(content_id).byte_count == len(payload)


def test_submission_validation_and_request_keys(make_client):
    client = make_client()
    content_id = upload(client)
    no_input = client.post("/v1/queries", json=submission("f" * 64))
    assert no_input.status_code == 400
    assert "not uploaded" in no_input.json()["error"]["message"]
    bad_sql = client.post("/v1/queries", json=submission(
        content_id, sql="SELECT r.nothing FROM reviews r"))
    assert bad_sql.status_code == 400
    assert bad_sql.json()["error"]["message"].startswith("CompileError")
    too_long = client.post("/v1/queries",
                           json=submission(content_id, timeout_s=10 ** 9))
    assert too_long.status_code == 400
    assert "maximum" in too_long.json()["error"]["message"]
    assert client.post("/v1/queries", content=b"{").status_code == 400
    assert client.post("/v1/queries", json=submission(
        content_id, inputs={})).status_code == 400

    first = client.post("/v1/queries",
                        json=submission(content_id, request_key="k"))
    assert first.status_code == 201, first.text
    status = first.json()
    assert status["state"] == "queued"
    assert status["timeout_s"] == 1000.0
    again = client.post("/v1/queries",
                        json=submission(content_id, request_key="k"))
    assert again.json()["id"] == status["id"]
    conflict = client.post("/v1/queries", json=submission(
        content_id, request_key="k", timeout_s=5))
    assert conflict.status_code == 409
    assert client.get("/v1/queries/nope").status_code == 404
    listed = client.get("/v1/queries").json()["queries"]
    assert [item["id"] for item in listed] == [status["id"]]


def test_lifecycle_over_http_with_long_poll_events_and_files(make_client):
    client = make_client()
    content_id = upload(client)
    status = client.post("/v1/queries", json=submission(content_id)).json()
    query_id = status["id"]
    not_ready = client.get(f"/v1/queries/{query_id}/result")
    assert not_ready.status_code == 409

    final = wait_done(client, query_id)
    assert final["state"] == "succeeded", final
    assert final["result"]["rows"] == 2
    assert final["plan"]["backend"] == "quail"
    assert final["progress"]["done"] == 6

    # events replay the current snapshot and stop at the terminal state
    with client.stream("GET", f"/v1/queries/{query_id}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    events = [json.loads(line[len("data: "):])
              for line in body.splitlines() if line.startswith("data: ")]
    assert [event["revision"] for event in events] == [final["revision"]]

    result = client.get(f"/v1/queries/{query_id}/result")
    assert result.status_code == 200
    with ipc.open_file(pa.BufferReader(result.content)) as reader:
        table = reader.read_all()
    assert sorted(table.column("r.id").to_pylist()) == ["r0", "r3"]
    report = client.get(f"/v1/queries/{query_id}/files/report.json").json()
    assert report["backend"] == "quail"
    answers = client.get(
        f"/v1/queries/{query_id}/files/answers/filters/r--0.arrow")
    assert answers.status_code == 200
    forbidden = client.get(f"/v1/queries/{query_id}/files/../quail.sqlite3")
    assert forbidden.status_code in (400, 404)
    rows = client.get(f"/v1/queries/{query_id}/rows", params={"limit": 1}).json()
    assert rows["columns"] == ["r.id"] and rows["total_rows"] == 2
    assert len(rows["rows"]) == 1

    # a long poll returns as soon as the revision changes
    fresh = client.post("/v1/queries", json=submission(content_id)).json()
    started = time.monotonic()
    polled = client.get(f"/v1/queries/{fresh['id']}",
                        params={"after": fresh["revision"], "wait": 20}).json()
    assert polled["revision"] > fresh["revision"]
    assert time.monotonic() - started < 15
    wait_done(client, fresh["id"])


def test_events_stream_newer_revisions_to_several_readers(tmp_path):
    import urllib.request

    settings = ServiceSettings(data_dir=tmp_path / "data", **SETTINGS)
    app = create_app(settings)
    url, stop = service_fakes.start_server(app)
    try:
        service = app.state.service
        # hold the scheduler so the record stays queued while readers attach
        service.scheduler.stop()
        payload = ipc_bytes(service_fakes.reviews_table())
        content_id = hashlib.sha256(payload).hexdigest()
        request = urllib.request.Request(
            f"{url}/v1/inputs/{content_id}", data=payload, method="PUT")
        urllib.request.urlopen(request).read()
        request = urllib.request.Request(
            f"{url}/v1/queries", method="POST",
            data=json.dumps(submission(content_id)).encode(),
            headers={"content-type": "application/json"})
        status = json.loads(urllib.request.urlopen(request).read())
        seen = {0: [], 1: []}

        def read(index):
            with urllib.request.urlopen(
                    f"{url}/v1/queries/{status['id']}/events") as response:
                for raw in response:
                    line = raw.decode("utf-8").rstrip("\n")
                    if line.startswith("data: "):
                        seen[index].append(json.loads(line[len("data: "):]))

        threads = [threading.Thread(target=read, args=(i,)) for i in seen]
        for thread in threads:
            thread.start()
        time.sleep(0.5)
        service.scheduler.start()
        for thread in threads:
            thread.join(30)
            assert not thread.is_alive()
    finally:
        stop()
    for revisions in ([s["revision"] for s in seen[0]],
                      [s["revision"] for s in seen[1]]):
        assert revisions[0] == status["revision"]
        assert revisions == sorted(revisions)
        assert len(set(revisions)) == len(revisions) >= 3
    assert seen[0][-1]["state"] == "succeeded"
    assert seen[1][-1]["state"] == "succeeded"


def test_cancel_and_timeout_over_http(make_client):
    client = make_client(hooks=service_fakes.sleeping_hooks)
    content_id = upload(client)
    queued = client.post("/v1/queries", json=submission(content_id)).json()
    slow = client.post("/v1/queries",
                       json=submission(content_id, timeout_s=0.5)).json()
    cancelled = client.post(f"/v1/queries/{slow['id']}/cancel").json()
    assert cancelled["state"] == "cancelled"
    running = wait_running(client, queued["id"])
    flagged = client.post(f"/v1/queries/{queued['id']}/cancel").json()
    assert flagged["cancel_requested"] and flagged["revision"] > running["revision"]
    final = wait_done(client, queued["id"])
    assert final["state"] == "cancelled"
    assert final["error"]["type"] == "Cancelled"

    timed = client.post("/v1/queries",
                        json=submission(content_id, timeout_s=0.5)).json()
    final = wait_done(client, timed["id"])
    assert final["state"] == "failed"
    assert final["error"]["type"] == "TimeoutError"
    assert client.post("/v1/queries/none/cancel").status_code == 404


def wait_running(client, query_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    status = client.get(f"/v1/queries/{query_id}").json()
    while status["state"] != "running":
        assert time.monotonic() < deadline, status
        status = client.get(f"/v1/queries/{query_id}",
                            params={"after": status["revision"],
                                    "wait": 1}).json()
    return status


def test_bearer_token_guards_the_api_but_not_the_page(make_client):
    client = make_client(token="secret")
    assert client.get("/v1/capabilities").status_code == 401
    assert client.get("/v1/capabilities",
                      headers={"authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/v1/capabilities",
                      headers={"authorization": "Bearer secret"}).status_code == 200
    page = client.get("/queries/abc")
    assert page.status_code == 200 and "<title>Quail query" in page.text
    assert client.get("/").status_code == 200


def test_restart_recovers_records_and_keeps_inputs(tmp_path):
    from starlette.testclient import TestClient

    settings = ServiceSettings(data_dir=tmp_path / "data", **SETTINGS)
    with TestClient(create_app(settings)) as client:
        content_id = upload(client)
        finished = client.post("/v1/queries", json=submission(content_id)).json()
        wait_done(client, finished["id"])
        service = client.app.state.service
        service.scheduler.stop()
        queued = client.post("/v1/queries", json=submission(content_id)).json()
        epoch = service.store.begin(queued["id"])
        service.store.update(queued["id"], epoch, state="running")
        active = client.post("/v1/queries", json=submission(content_id)).json()
    # the process "restarts": a new app over the same data directory
    with TestClient(create_app(settings)) as client:
        assert client.app.state.service.recovered == [queued["id"]]
        assert client.get(
            f"/v1/queries/{queued['id']}").json()["state"] == "interrupted"
        again = client.get(f"/v1/queries/{finished['id']}").json()
        assert again["state"] == "succeeded"
        assert client.get(
            f"/v1/queries/{finished['id']}/result").status_code == 200
        assert client.head(f"/v1/inputs/{content_id}").status_code == 200
        assert wait_done(client, active["id"])["state"] == "succeeded"


def test_hf_inputs_are_resolved_without_download(make_client, monkeypatch):
    client = make_client()
    calls = []

    def fake_from_hf(dataset, id_col, split="train", config="",
                     revision=""):
        calls.append((dataset, revision))
        from quail.catalog import HuggingFaceProvider

        return HuggingFaceProvider(
            dataset, id_col=id_col, split=split, config=config,
            revision=revision,
            arrow_schema=pa.schema([("id", pa.string()),
                                    ("review", pa.string())]))

    monkeypatch.setattr(inputs.DocumentProvider, "from_hf",
                        staticmethod(fake_from_hf))
    body = submission("unused", inputs={"reviews": {
        "kind": "hf", "dataset": "org/reviews", "config": "", "split": "train",
        "revision": "abc123", "id_col": "id"}})
    response = client.post("/v1/queries", json=body)
    assert response.status_code == 201, response.text
    assert calls == [("org/reviews", "abc123")]
    client.app.state.service.scheduler.stop()
