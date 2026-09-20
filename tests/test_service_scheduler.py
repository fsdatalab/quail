"""Executor and scheduler: lifecycle, failures, cancel, timeout, restart."""

import json
import time

import pyarrow as pa
import pytest
import service_fakes
from test_session import fake_tok

import quail
from quail.builtins import built_in_registry
from quail.service import artifacts, inputs
from quail.service.executor import (
    ChildProcessExecutor,
    InProcessExecutor,
    load_hooks,
)
from quail.service.scheduler import Scheduler
from quail.service.store import Store


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "quail.sqlite3")
    yield store
    store.close()


def submit(store, tmp_path, sql=service_fakes.FILTER_SQL, timeout_s=1000.0,
           table=None):
    provider = quail.DocumentProvider.from_table(
        table if table is not None else service_fakes.reviews_table(),
        id_col="id")
    prepared = inputs.describe(provider, tmp_path / "uploads")
    store.put_input(prepared.content_id, str(prepared.upload_path),
                    prepared.upload_path.stat().st_size)
    return store.create(
        spec={"sql": sql, "dialect": "snowflake", "order": None},
        config=service_fakes.CONFIG,
        inputs={"reviews": prepared.spec}, timeout_s=timeout_s)


def wait_done(store, query_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    status = store.get(query_id)
    while not status.done and time.monotonic() < deadline:
        status = store.wait(query_id, status.revision, 1.0)
    assert status.done, status
    return status


def test_in_process_run_saves_plan_progress_and_a_readable_result(
        store, tmp_path):
    status = submit(store, tmp_path)
    scheduler = Scheduler(store, InProcessExecutor(service_fakes.hooks),
                          tmp_path, poll_s=0.02)
    final = scheduler.run_one(status)

    assert final.state == "succeeded"
    assert final.plan["text"].startswith("Project") or "Scan" in final.plan["text"]
    assert final.plan["backend"] == "quail"
    assert final.progress["label"] == "filter (2 stages)"
    assert final.progress["done"] == 6
    assert final.result["rows"] == 2
    assert final.result["columns"] == ["r.id"]
    directory = tmp_path / final.result["directory"]
    result = artifacts.load_result(directory, final.result,
                                   built_in_registry().codecs)
    assert sorted(result.to_rows()) == [("r0",), ("r3",)]
    assert set(result.answer_tables["filters"]) == {("r", 0), ("r", 1)}
    assert result.report["fresh_tokens"] > 0
    assert "Scan" in result.explain()
    report = json.loads((directory / "report.json").read_text())
    assert report["backend"] == "quail"


def test_compile_and_executor_failures_are_saved(store, tmp_path):
    bad = submit(store, tmp_path, sql="SELECT r.nothing FROM reviews r")
    scheduler = Scheduler(store, InProcessExecutor(service_fakes.hooks),
                          tmp_path, poll_s=0.02)
    final = scheduler.run_one(bad)
    assert final.state == "failed"
    assert final.error["type"] == "CompileError"

    failing = submit(store, tmp_path)
    scheduler = Scheduler(store, InProcessExecutor(service_fakes.failing_hooks),
                          tmp_path, poll_s=0.02)
    final = scheduler.run_one(failing)
    assert final.state == "failed"
    assert final.error["type"] == "RuntimeError"
    assert "refused to start" in final.error["message"]
    assert "Traceback" in final.error["traceback"]


def test_publication_failure_is_a_saved_failure_not_success(
        store, tmp_path, monkeypatch):
    status = submit(store, tmp_path)

    def broken_verify(directory, manifest):
        raise OSError("disk full")

    monkeypatch.setattr("quail.service.scheduler.verify_result", broken_verify)
    scheduler = Scheduler(store, InProcessExecutor(service_fakes.hooks),
                          tmp_path, poll_s=0.02)
    final = scheduler.run_one(status)
    assert final.state == "failed"
    assert final.error["type"] == "PublishError"
    assert "disk full" in final.error["message"]


def test_verify_rejects_a_missing_or_short_result_file(tmp_path):
    table = pa.table({"r.id": ["r0", "r3"]})
    result = quail.QueryResult.from_table(table, {"backend": "quail"})
    manifest = artifacts.write_result(result, tmp_path)
    artifacts.verify_result(tmp_path, manifest)
    assert artifacts.load_result(tmp_path, manifest).to_rows() == [
        ("r0",), ("r3",)]
    with pytest.raises(ValueError, match="holds 2 rows"):
        artifacts.verify_result(tmp_path, {**manifest, "rows": 3})
    (tmp_path / "result.arrow").unlink()
    with pytest.raises(FileNotFoundError):
        artifacts.verify_result(tmp_path, manifest)
    assert not list(tmp_path.glob("*.tmp"))


def test_scheduler_thread_runs_queued_records_in_order(store, tmp_path):
    first = submit(store, tmp_path)
    second = submit(store, tmp_path)
    scheduler = Scheduler(store, InProcessExecutor(service_fakes.hooks),
                          tmp_path, poll_s=0.02)
    scheduler.start()
    try:
        done_first = wait_done(store, first.id)
        done_second = wait_done(store, second.id)
    finally:
        scheduler.stop()
    assert done_first.state == done_second.state == "succeeded"
    assert done_first.started_at <= done_second.started_at


def test_cancel_before_execution_never_starts_it(store, tmp_path):
    status = submit(store, tmp_path)
    store.request_cancel(status.id)
    scheduler = Scheduler(store, InProcessExecutor(service_fakes.hooks),
                          tmp_path, poll_s=0.02)
    final = scheduler.run_one(status)
    assert final.state == "cancelled"
    assert final.started_at is None


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_child_process_execution_stops_on_cancel_or_timeout(
        store, tmp_path, stop):
    timeout_s = 1.0 if stop == "timeout" else 1000.0
    status = submit(store, tmp_path, timeout_s=timeout_s)
    executor = ChildProcessExecutor("service_fakes:sleeping_hooks")
    scheduler = Scheduler(store, executor, tmp_path, poll_s=0.05)
    scheduler.start()
    try:
        deadline = time.monotonic() + 60
        while store.get(status.id).state != "running":
            assert time.monotonic() < deadline, store.get(status.id)
            time.sleep(0.05)
        if stop == "cancel":
            store.request_cancel(status.id)
        final = wait_done(store, status.id, timeout=60)
        if stop == "cancel":
            assert final.state == "cancelled"
            assert final.error["type"] == "Cancelled"
        else:
            assert final.state == "failed"
            assert final.error["type"] == "TimeoutError"
            assert "1 s timeout" in final.error["message"]
        # the record closes first, then the child is killed; the next
        # query starts a fresh child
        deadline = time.monotonic() + 30
        while executor._process is not None and executor._process.is_alive():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        again = submit(store, tmp_path)
        executor.hooks_reference = "service_fakes:hooks"
        assert wait_done(store, again.id, timeout=60).state == "succeeded"
    finally:
        scheduler.stop()


def test_load_hooks_checks_the_type():
    assert load_hooks(None) is None
    assert load_hooks("service_fakes:hooks") is service_fakes.hooks
    with pytest.raises(TypeError):
        load_hooks("service_fakes:TRUTH")


def test_describe_and_resolve_supported_providers(tmp_path):
    table = service_fakes.reviews_table()
    memory = inputs.describe(
        quail.DocumentProvider.from_table(table, id_col="id"), tmp_path)
    assert memory.spec["kind"] == "snapshot"
    assert memory.spec["id_col"] == "id"
    assert memory.upload_path.name == f"{memory.content_id}.arrow"
    # the same table gives the same content id
    same = inputs.describe(
        quail.DocumentProvider.from_table(table, id_col="id"), tmp_path)
    assert same.content_id == memory.content_id

    import pyarrow.parquet as pq

    pq.write_table(table, tmp_path / "r.parquet")
    parquet = inputs.describe(quail.DocumentProvider.from_parquet(
        str(tmp_path / "r.parquet"), id_col="id"), tmp_path)
    assert parquet.spec["kind"] == "snapshot"
    provider = inputs.resolve(parquet.spec,
                              {parquet.content_id: parquet.upload_path})
    scanned = provider.scan(quail.ScanRequest(columns=("id",))).read_all()
    assert scanned.column("id").to_pylist() == table.column("id").to_pylist()

    hf = quail.catalog.HuggingFaceProvider(
        "org/data", id_col="id", arrow_schema=pa.schema([("id", pa.string())]))
    pinned = inputs.describe(hf, tmp_path,
                             resolve_revision=lambda name: f"sha-of-{name}")
    assert pinned.spec == {"kind": "hf", "dataset": "org/data", "config": "",
                           "split": "train", "revision": "sha-of-org/data",
                           "id_col": "id"}
    assert pinned.upload_path is None
    with pytest.raises(inputs.InvalidRequestError, match="revision"):
        inputs.resolve({**pinned.spec, "revision": ""}, {})
    with pytest.raises(inputs.InvalidRequestError, match="not uploaded"):
        inputs.resolve(memory.spec, {})
    with pytest.raises(inputs.InvalidRequestError, match="unknown input kind"):
        inputs.resolve({"kind": "s3"}, {})

    class Custom:
        id_col = "id"

    with pytest.raises(TypeError, match="Custom cannot be sent"):
        inputs.describe(Custom(), tmp_path)


def test_hf_provider_carries_its_revision():
    provider = quail.catalog.HuggingFaceProvider(
        "org/data", id_col="id", revision="abc",
        arrow_schema=pa.schema([("id", pa.string())]))
    head = quail.catalog.HuggingFaceProvider(
        "org/data", id_col="id", arrow_schema=pa.schema([("id", pa.string())]))
    assert provider.content_identity() != head.content_identity()
    assert fake_tok("a b") == ["a", "b"]
