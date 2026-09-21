"""Copying the live SQLite file to another disk and restoring from it."""

import sqlite3
import threading
import time

import pytest
import service_fakes

from quail.service.checkpoint import Checkpoint, restore
from quail.service.store import Store

SPEC = {"sql": "SELECT r.id FROM reviews r", "dialect": "snowflake",
        "order": None}
INPUTS = {"reviews": {"kind": "snapshot", "content_id": "abc", "id_col": "id"}}


def create(store):
    return store.create(spec=SPEC, config=service_fakes.CONFIG, inputs=INPUTS,
                        timeout_s=1000.0)


def test_backup_is_a_complete_copy_including_the_wal(tmp_path):
    store = Store(tmp_path / "live" / "quail.sqlite3")
    first = create(store)
    epoch = store.begin(first.id)
    store.update(first.id, epoch, state="running", progress={"done": 1})
    # the WAL holds these writes; a plain file copy would miss them
    assert (tmp_path / "live" / "quail.sqlite3-wal").stat().st_size > 0
    copy = tmp_path / "volume" / "quail.sqlite3"
    store.backup(copy)
    assert not copy.with_name("quail.sqlite3.tmp").exists()
    with sqlite3.connect(str(copy)) as conn:
        rows = conn.execute("SELECT id, state FROM queries").fetchall()
    assert rows == [(first.id, "running")]
    store.close()

    # a restart on another machine starts from the copy
    live = tmp_path / "fresh" / "quail.sqlite3"
    assert restore(copy, live)
    assert not restore(copy, live)
    reopened = Store(live)
    assert reopened.recover() == [first.id]
    assert reopened.get(first.id).progress == {"done": 1}
    reopened.close()


def test_checkpoint_copies_after_writes_and_commits(tmp_path):
    store = Store(tmp_path / "live" / "quail.sqlite3")
    commits = []
    copy = tmp_path / "volume" / "quail.sqlite3"
    checkpoint = Checkpoint(store, copy, commit=lambda: commits.append(1),
                            min_interval_s=0.05)
    assert checkpoint.run_once()
    assert not checkpoint.run_once()
    assert commits == [1] and copy.exists()

    checkpoint.start()
    status = create(store)
    deadline = time.monotonic() + 10
    while checkpoint.copies < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert checkpoint.copies == 2
    with sqlite3.connect(str(copy)) as conn:
        assert conn.execute("SELECT count(*) FROM queries").fetchone() == (1,)
    time.sleep(0.2)
    assert checkpoint.copies == 2, "no writes, no copies"

    store.request_cancel(status.id)
    checkpoint.stop()
    checkpoint.stop()
    with sqlite3.connect(str(copy)) as conn:
        assert conn.execute("SELECT state FROM queries").fetchone() == (
            "cancelled",)
    assert len(commits) == checkpoint.copies == 3
    store.close()
    assert not checkpoint.run_once()


def test_progress_only_writes_are_copied_on_the_slow_timer(tmp_path):
    store = Store(tmp_path / "live" / "quail.sqlite3")
    copy = tmp_path / "volume" / "quail.sqlite3"
    checkpoint = Checkpoint(store, copy, progress_interval_s=0.3)
    status = create(store)
    epoch = store.begin(status.id)
    assert checkpoint.run_once(), "a state change is copied at once"

    for done in range(1, 4):
        store.update(status.id, epoch, progress={"done": done})
        assert not checkpoint.run_once(), "progress alone waits for the timer"
    assert store.write_count == store.durable_write_count + 3
    time.sleep(0.35)
    assert checkpoint.run_once(), "the timer ran out"
    with sqlite3.connect(str(copy)) as conn:
        assert conn.execute("SELECT progress_json FROM queries").fetchone() == (
            '{"done": 3}',)

    store.update(status.id, epoch, progress={"done": 4})
    assert not checkpoint.run_once()
    assert checkpoint.run_once(force=True), "stop() forces the last progress"
    store.update(status.id, epoch, progress={"done": 5})
    store.update(status.id, epoch, state="running")
    assert checkpoint.run_once(), "a state change is not held back"
    store.close()


def test_checkpoint_survives_a_failing_commit(tmp_path, caplog):
    store = Store(tmp_path / "live" / "quail.sqlite3")
    calls = []

    def flaky_commit():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("volume busy")

    checkpoint = Checkpoint(store, tmp_path / "volume" / "quail.sqlite3",
                            commit=flaky_commit, min_interval_s=0.05)
    checkpoint.start()
    deadline = time.monotonic() + 10
    while checkpoint.copies < 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    checkpoint.stop()
    assert checkpoint.copies == 1 and len(calls) == 2
    assert "checkpoint" in caplog.text and "volume busy" in caplog.text
    store.close()


def test_service_runs_closers_before_closing_the_store(tmp_path):
    pytest.importorskip("starlette")
    from quail.service.app import ServiceSettings, create_app

    settings = ServiceSettings(
        data_dir=tmp_path / "data", db_path=tmp_path / "local" / "live.sqlite3",
        models=("qwen3-4b-fp8",), device="h100-sxm", in_process=True,
        hooks=service_fakes.hooks)
    app = create_app(settings)
    service = app.state.service
    assert service.store.path == tmp_path / "local" / "live.sqlite3"
    checkpoint = Checkpoint(service.store, tmp_path / "data" / "quail.sqlite3")
    service.add_closer(checkpoint.stop)
    seen = []
    service.add_closer(lambda: seen.append(service.store.closed))
    thread = threading.Thread(target=service.stop)
    thread.start()
    thread.join(30)
    assert seen == [False]
    assert service.store.closed
    assert (tmp_path / "data" / "quail.sqlite3").exists()
