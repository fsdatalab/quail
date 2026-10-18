"""HTTP routes of Quail Server, driven with Starlette's test client."""

import asyncio
import hashlib
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from helpers import server_fakes

from quail.remote import inputs
from quail.remote.records import ServerError

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


def test_uploads_are_checked_and_saved_once(make_client):
    client = make_client(max_upload_bytes=4096)
    payload = ipc_bytes(server_fakes.reviews_table())
    content_id = hashlib.sha256(payload).hexdigest()
    assert client.head(f"/v1/inputs/{content_id}").status_code == 404
    assert client.put("/v1/inputs/notahash", content=payload).status_code == 400
    assert_error(client.put(f"/v1/inputs/{'0' * 64}", content=payload),
                 400, "hash")
    text = b"not arrow"
    assert_error(client.put(f"/v1/inputs/{hashlib.sha256(text).hexdigest()}",
                            content=text), 400, "Arrow IPC")
    big = ipc_bytes(pa.table({"id": ["x" * 100] * 100}))
    too_big = client.put(f"/v1/inputs/{hashlib.sha256(big).hexdigest()}",
                         content=big)
    assert too_big.status_code == 413
    for expected in (201, 200):
        assert client.put(f"/v1/inputs/{content_id}",
                          content=payload).status_code == expected
    assert client.head(f"/v1/inputs/{content_id}").status_code == 200
    server = client.app.state.server
    assert not list(server.inputs_dir.glob("*.tmp"))
    assert server.store.get_input(content_id).num_bytes == len(payload)


def test_capabilities_submission_validation_client_ids_and_hf_inputs(
        make_client, monkeypatch):
    client = make_client()
    capabilities = client.get("/v1/capabilities").json()
    assert capabilities["models"] == ["qwen3-4b-fp8"]
    assert capabilities["default_timeout_s"] == 1000.0
    content_id = upload(client)
    config = server_fakes.CONFIG
    for body, message in [
        (submission(content_id, config={**config, "model": "qwen3-32b-fp8"}),
         "does not run model"),
        (submission(content_id, config={
            **config, "device": "rtx-pro-6000-blackwell-server"}), "runs on device"),
        (submission(content_id, config={**config, "gpus": 2}), "GPUs"),
        (submission(content_id, config={**config, "backend": "stock_vllm"}),
         "backends"),
        (submission(content_id, config={"model": "qwen3-4b-fp8"}),
         "bad engine config"),
        (submission("f" * 64), "not uploaded"),
        (submission(content_id, sql="SELECT r.nothing FROM reviews r"),
         "CompileError"),
        (submission(content_id, timeout_s=10 ** 9), "maximum"),
        (submission(content_id, query_id="a/b"), "query id"),
    ]:
        assert_error(client.post("/v1/queries", json=body), 400, message)
    assert client.post("/v1/queries", content=b"{").status_code == 400
    assert client.post("/v1/queries", json=submission(
        content_id, inputs={})).status_code == 400

    first = client.post("/v1/queries", json=submission(
        content_id, query_id="k", session_id="s1"))
    assert first.status_code == 201, first.text
    status = first.json()
    assert status["id"] == "k" and status["session_id"] == "s1"
    assert status["state"] == status["phase"]["name"] == "queued"
    assert "done" not in status
    assert status["timeout_s"] == 1000.0
    again = client.post("/v1/queries",
                        json=submission(content_id, query_id="k"))
    assert again.json()["id"] == "k"
    conflict = client.post("/v1/queries", json=submission(
        content_id, query_id="k", timeout_s=5))
    assert conflict.status_code == 409
    assert client.get("/v1/queries/nope").status_code == 404
    listed = client.get("/v1/queries").json()["queries"]
    assert [item["id"] for item in listed] == ["k"]
    other = client.post("/v1/queries", json=submission(content_id)).json()
    assert len(other["id"]) == 32 and other["session_id"] is None
    mine = client.get("/v1/queries", params={"session_id": "s1"}).json()
    assert [item["id"] for item in mine["queries"]] == ["k"]
    all_queries = client.get("/v1/queries", params={"limit": "all"}).json()
    assert {item["id"] for item in all_queries["queries"]} == {"k", other["id"]}
    # every EngineConfig field reaches the record, not only the four
    # the server checks
    timed = client.post("/v1/queries", json=submission(
        content_id, query_id="timed",
        config={**config, "gpu_timing": True, "attention": "tree"})).json()
    stored = client.app.state.server.store.find(timed["id"]).config
    assert stored["gpu_timing"] is True and stored["attention"] == "tree"
    assert stored["canvas_draws"] == 4

    # a Hugging Face input is resolved without a download
    from quail.catalog import HuggingFaceProvider

    calls = []

    def fake_from_hf(dataset, id_col, split="train", config="", revision=""):
        calls.append((dataset, revision))
        return HuggingFaceProvider(
            dataset, id_col=id_col, split=split, config=config,
            revision=revision,
            arrow_schema=pa.schema([("id", pa.string()), ("review", pa.string())]))

    monkeypatch.setattr(inputs.DocumentProvider, "from_hf",
                        staticmethod(fake_from_hf))
    body = submission("unused", inputs={"reviews": {
        "kind": "hf", "dataset": "org/reviews", "config": "", "split": "train",
        "revision": "abc123", "id_col": "id"}})
    response = client.post("/v1/queries", json=body)
    assert response.status_code == 201, response.text
    assert calls == [("org/reviews", "abc123")]
    assert wait_done(client, response.json()["id"])["state"] == "failed"


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
        f"/v1/queries/{query_id}/files/answers/filters/r--0.arrow")
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


def test_bearer_token_guards_the_api_but_not_the_page(make_client):
    client = make_client(token="secret")
    for headers, code in [({}, 401), ({"authorization": "Bearer wrong"}, 401),
                          ({"authorization": "Bearer secret"}, 200)]:
        assert client.get("/v1/capabilities", headers=headers).status_code == code
    page = client.get("/queries/abc")
    assert page.status_code == 200 and "<title>Quail query" in page.text
    assert client.get("/").status_code == 200
    assert client.get("/queries").status_code == 200


@pytest.mark.parametrize("concurrent", [False, True])
def test_a_cancelled_upload_leaves_no_partial_file(make_client, concurrent):
    client = make_client()
    server = client.app.state.server
    payload = ipc_bytes(server_fakes.reviews_table())
    content_id = hashlib.sha256(payload).hexdigest()

    async def scenario():
        waiting = asyncio.Event()

        async def paused_body():
            yield payload[:100]
            waiting.set()
            await asyncio.Event().wait()

        async def complete_body():
            yield payload

        complete = SimpleNamespace(stream=complete_body)
        task = asyncio.create_task(server.save_upload(
            SimpleNamespace(stream=paused_body), content_id))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=2)
            if concurrent:
                assert await server.save_upload(complete, content_id) == 201
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not list(server.inputs_dir.glob("*.tmp"))
        if not concurrent:
            assert not list(server.inputs_dir.iterdir())
            assert await server.save_upload(complete, content_id) == 201
        assert (server.inputs_dir / f"{content_id}.arrow").read_bytes() == payload

    asyncio.run(scenario())


def test_a_retry_waits_for_the_first_submission_to_be_durable(make_client, monkeypatch):
    client = make_client()
    content_id = upload(client)
    server = client.app.state.server
    server.scheduler.stop()
    syncing, release = threading.Event(), threading.Event()

    def sync():
        if not syncing.is_set():
            syncing.set()
            assert release.wait(5)
            raise OSError("volume commit failed")

    server.add_sync(sync)
    body = submission(content_id, query_id="retry")
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(server.accept, body)
        assert syncing.wait(5)
        retry = pool.submit(server.accept, body)
        queued = pool.submit(server.store.next_queued)
        try:
            with pytest.raises(TimeoutError):
                retry.result(timeout=0.2)
            assert queued.result(timeout=1) is None
        finally:
            release.set()
        with pytest.raises(ServerError, match="volume commit failed"):
            first.result(timeout=5)
        # the retry saw no record of the discarded first attempt
        assert retry.result(timeout=5).state == "queued"


@pytest.mark.parametrize("finish", ["all", "disconnect", "send_error", "cancel"])
def test_the_result_stream_closes_its_file_however_the_response_ends(
        make_client, monkeypatch, finish):
    from contextlib import contextmanager

    from starlette.requests import ClientDisconnect

    client = make_client()
    query_id = client.post("/v1/queries", json=submission(upload(client))).json()["id"]
    assert wait_done(client, query_id)["state"] == "succeeded"
    endpoint = next(route.endpoint for route in client.app.routes
                    if route.name == "result_stream")
    open_file = ipc.open_file
    closed = []

    @contextmanager
    def tracked_file(path):
        with open_file(path) as reader:
            try:
                yield reader
            finally:
                closed.append(path)

    monkeypatch.setattr(ipc, "open_file", tracked_file)

    async def scenario():
        sent = asyncio.Event()
        response = await endpoint(SimpleNamespace(
            path_params={"query_id": query_id}, query_params={"limit": "1"}))
        assert response.headers["x-quail-rows"] == "1"

        async def send(message):
            if message["type"] == "http.response.body":
                sent.set()
                if finish == "send_error":
                    raise OSError("client disconnected")
                if finish in {"disconnect", "cancel"}:
                    await asyncio.Event().wait()

        async def receive():
            if finish == "disconnect":
                await sent.wait()
                return {"type": "http.disconnect"}
            await asyncio.Event().wait()

        # an ASGI 2.3 server reports a disconnect on receive; 2.4 on send
        spec = "2.3" if finish == "disconnect" else "2.4"
        task = asyncio.create_task(response(
            {"type": "http", "asgi": {"spec_version": spec}}, receive, send))
        if finish == "cancel":
            await asyncio.wait_for(sent.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif finish == "send_error":
            with pytest.raises(ClientDisconnect):
                await task
        else:
            await asyncio.wait_for(task, timeout=5)
        assert len(closed) == 1

    asyncio.run(scenario())
    bad = client.get(f"/v1/queries/{query_id}/result", params={"limit": -1})
    assert bad.status_code == 400
