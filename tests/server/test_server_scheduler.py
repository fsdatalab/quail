"""Executor and scheduler: lifecycle, failures, cancel, timeout, restart."""

import json

import pyarrow as pa
import pytest
from helpers import server_fakes

import quail
from quail.builtins import built_in_registry
from quail.remote import artifacts, inputs
from quail.server.executor import InProcessExecutor
from quail.server.scheduler import Scheduler
from quail.server.store import Store


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "quail.sqlite3")
    yield store
    store.close()


def submit(store, tmp_path, sql=server_fakes.FILTER_SQL, timeout_s=1000.0,
           config=None):
    prepared = inputs.describe(quail.DocumentProvider.from_table(
        server_fakes.reviews_table(), id_col="id"), tmp_path / "uploads")
    store.put_input(prepared.content_id, str(prepared.upload_path),
                    prepared.upload_path.stat().st_size)
    return store.create(
        spec={"sql": sql, "dialect": "snowflake", "order": None},
        config=config or server_fakes.CONFIG, inputs={"reviews": prepared.spec},
        timeout_s=timeout_s)


def scheduler_for(store, tmp_path, executor=None, poll_s=0.02):
    executor = executor or InProcessExecutor(server_fakes.hooks)
    return Scheduler(store, executor, tmp_path, poll_s=poll_s)


def test_in_process_run_saves_a_readable_and_verified_result(store, tmp_path):
    final = scheduler_for(store, tmp_path).run_one(submit(store, tmp_path))
    assert final.state == "succeeded"
    assert final.progress["label"] == "filter (2 stages)"
    assert final.result["rows"] == 2 and final.result["columns"] == ["r.id"]
    directory = tmp_path / final.result["directory"]
    result = artifacts.load_result(directory, final.result,
                                   built_in_registry().codecs)
    assert sorted(result.to_rows()) == [("r0",), ("r3",)]
    assert result.report["fresh_tokens"] > 0
    assert "Scan" in result.explain()
    assert json.loads((directory / "report.json").read_text())["backend"] == "quail"

    # each entry is (row index, last stage asked, passed)
    saved = artifacts.read_answers(directory / artifacts.ANSWERS_FILE)
    assert [entry["kind"] for entry in saved] == ["filter", "filter"]
    assert all(entry["alias"] == "r" and entry["stages"] == 1 for entry in saved)
    first, last = [entry["documents"] for entry in saved]
    assert len(first) == 6 and len(last) == 4
    assert {row for row, _, passed in last if passed} == {0, 3}
    assert final.progress["answers_saved"] == 2

    # verification rejects a missing or short result file
    verified = tmp_path / "verified"
    verified.mkdir()
    table = pa.table({"r.id": ["r0", "r3"]})
    result = quail.QueryResult.from_table(table, {"backend": "quail"})
    # every answer kind is saved under the key the result gives it
    result.answer_tables = {
        "filters": {("r", 0): pa.table({"r": [0, 3], "answer": [True, True]})},
        "joins": {0: pa.table({"r": [0], "p": [1], "answer": [True]})},
        "classifies": {"kind": pa.table({"r": [0], "kind": ["a"]})},
        "scores": {"score": pa.table({"r": [0], "score": [0.5]})},
        "extracts": {"lease": pa.table({"r": [0], "lease": ["x"]})},
    }
    manifest = artifacts.write_result(result, verified)
    artifacts.verify_result(verified, manifest)
    loaded = artifacts.load_result(verified, manifest)
    assert loaded.to_rows() == [("r0",), ("r3",)]
    for kind, tables in result.answer_tables.items():
        assert loaded.answer_tables[kind].keys() == tables.keys()
        for key, table in tables.items():
            assert loaded.answer_tables[kind][key].equals(table)
    with pytest.raises(ValueError, match="holds 2 rows"):
        artifacts.verify_result(verified, {**manifest, "rows": 3})
    (verified / "result.arrow").unlink()
    with pytest.raises(FileNotFoundError):
        artifacts.verify_result(verified, manifest)
    assert not list(verified.glob("*.tmp"))


def test_failed_and_cancelled_runs_are_saved_and_not_killed(
        store, tmp_path, monkeypatch):
    executor = InProcessExecutor(server_fakes.hooks)
    scheduler = scheduler_for(store, tmp_path, executor)
    bad = scheduler.run_one(
        submit(store, tmp_path, sql="SELECT r.nothing FROM reviews r"))
    assert (bad.state, bad.error["type"]) == ("failed", "CompileError")
    assert executor.loaded_model is None

    queued = submit(store, tmp_path)
    store.request_cancel(queued.id)
    cancelled = scheduler.run_one(queued)
    assert cancelled.state == "cancelled" and cancelled.started_at is None

    failing = scheduler_for(
        store, tmp_path, InProcessExecutor(server_fakes.failing_hooks),
    ).run_one(submit(store, tmp_path))
    assert (failing.state, failing.error["type"]) == ("failed", "RuntimeError")
    assert "refused to start" in failing.error["message"]
    assert "Traceback" in failing.error["traceback"]

    def broken_verify(directory, manifest):
        raise OSError("disk full")

    monkeypatch.setattr("quail.server.scheduler.verify_result", broken_verify)
    unpublished = scheduler.run_one(submit(store, tmp_path))
    assert (unpublished.state, unpublished.error["type"]) == (
        "failed", "PublishError")
    assert "disk full" in unpublished.error["message"]

    stops = []

    class Execution:
        def __init__(self, emit):
            self.emit = emit

        def wait(self, timeout):
            # the first poll sees the record closed before done is reported
            if self.emit is None:
                return True
            self.emit("failed", {"type": "RuntimeError", "message": "GPU said no"})
            self.emit = None
            return False

        def stop(self):
            stops.append(1)

    class Executor:
        def start(self, job, emit):
            return Execution(emit)

    final = scheduler_for(store, tmp_path, Executor()).run_one(
        submit(store, tmp_path))
    assert final.state == "failed" and final.error["message"] == "GPU said no"
    assert stops == [], "killing it would throw away the loaded model"
