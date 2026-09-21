"""Saved query records: revisions, query ids, epochs, cancel, recovery."""

import threading
import time

import pytest

from quail.server.records import (
    InvalidRequestError,
    QueryIdConflictError,
    QueryStatus,
    UnknownQueryError,
)
from quail.server.store import Store

SPEC = {"sql": "SELECT r.id FROM reviews r", "dialect": "snowflake",
        "order": None}
CONFIG = {"model": "qwen3-4b-fp8", "device": "h100-sxm", "gpus": 1,
          "backend": "quail"}
INPUTS = {"reviews": {"kind": "snapshot", "content_id": "abc", "id_col": "id"}}


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "quail.sqlite3")
    yield store
    store.close()


def create(store, **overrides):
    arguments = dict(spec=SPEC, config=CONFIG, inputs=INPUTS, timeout_s=1000.0)
    arguments.update(overrides)
    return store.create(**arguments)


def test_create_saves_a_queued_record_with_a_complete_snapshot(store):
    status = create(store)
    assert status.state == "queued"
    assert status.revision == 1
    assert status.spec == SPEC and status.config == CONFIG
    assert status.inputs == INPUTS
    assert status.timeout_s == 1000.0
    assert not status.done
    assert store.get(status.id) == status
    # the snapshot round trips through JSON without Quail objects
    assert QueryStatus.from_dict(status.to_dict()) == status
    with pytest.raises(UnknownQueryError):
        store.get("missing")
    with pytest.raises(InvalidRequestError):
        create(store, timeout_s=0)


def test_a_client_query_id_returns_the_same_record_or_conflicts(store):
    first = create(store, query_id="demo-1")
    assert first.id == "demo-1"
    again = create(store, query_id="demo-1")
    assert again == first
    assert len(store.list_recent()) == 1
    with pytest.raises(QueryIdConflictError):
        store.create(spec={**SPEC, "sql": "SELECT 1"}, config=CONFIG,
                     inputs=INPUTS, timeout_s=1000.0, query_id="demo-1")
    other = create(store, query_id="demo-2")
    assert other.id != first.id
    # the store picks an id when the client sends none
    assert len(create(store).id) == 32
    for bad in ("", "-x", "a/b", "a" * 129, 7):
        with pytest.raises(InvalidRequestError, match="query id"):
            create(store, query_id=bad)
    assert store.find("demo-1") == first
    assert store.find("nope") is None


def test_records_carry_their_session_id(store):
    mine = create(store, session_id="s1")
    other = create(store, session_id="s2")
    create(store)
    assert mine.session_id == "s1"
    assert [item.id for item in store.list_recent(session_id="s1")] == [mine.id]
    assert [item.id for item in store.list_recent(session_id="s2")] == [other.id]
    assert len(store.list_recent()) == 3


def test_discard_deletes_a_queued_record_or_cancels_a_started_one(store):
    queued = create(store, query_id="q")
    assert store.discard(queued.id)
    assert store.find("q") is None
    # the same id can be used again, for a different query too
    again = store.create(spec={**SPEC, "sql": "SELECT 1"}, config=CONFIG,
                         inputs=INPUTS, timeout_s=1000.0, query_id="q")
    assert again.spec["sql"] == "SELECT 1"
    store.begin(again.id)
    assert not store.discard(again.id)
    assert store.get(again.id).cancel_requested


def test_listeners_are_called_after_every_committed_write(store):
    status = create(store)
    calls = []

    def listener():
        calls.append(store.get(status.id).revision)

    store.add_listener(listener)
    epoch = store.begin(status.id)
    store.update(status.id, epoch, progress={"done": 1})
    assert calls == [2, 3], "each call sees the committed write"
    store.remove_listener(listener)
    store.update(status.id, epoch, progress={"done": 2})
    assert calls == [2, 3]


def test_updates_bump_revisions_and_only_the_current_epoch_writes(store):
    status = create(store)
    epoch = store.begin(status.id)
    planning = store.get(status.id)
    assert (planning.state, planning.revision, epoch) == ("planning", 2, 1)
    assert planning.started_at is not None
    with pytest.raises(InvalidRequestError):
        store.begin(status.id)

    assert store.update(status.id, epoch, plan={"text": "Scan"})
    assert store.update(status.id, epoch, state="running",
                        progress={"done": 3, "total": 6})
    running = store.get(status.id)
    assert running.state == "running"
    assert running.plan == {"text": "Scan"}
    assert running.progress == {"done": 3, "total": 6}
    assert running.revision == 4

    # a stale execution cannot change the record
    assert not store.update(status.id, epoch + 1, progress={"done": 6})
    assert store.get(status.id).progress == {"done": 3, "total": 6}
    with pytest.raises(ValueError):
        store.update(status.id, epoch, state="succeeded")

    assert store.finish(status.id, epoch, "succeeded", result={"rows": 2})
    final = store.get(status.id)
    assert final.done and final.result == {"rows": 2}
    assert final.revision == 5
    # a closed record never changes again
    assert not store.finish(status.id, epoch, "failed",
                            error={"type": "Late", "message": "late"})
    assert not store.update(status.id, epoch, progress={"done": 0})
    assert store.get(status.id).revision == 5
    with pytest.raises(ValueError):
        store.finish(status.id, epoch, "succeeded")


def test_wait_returns_when_the_revision_passes(store):
    status = create(store)
    seen = []

    def reader():
        seen.append(store.wait(status.id, after=1, timeout=5.0))

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for thread in threads:
        thread.start()
    time.sleep(0.1)
    assert not seen
    store.begin(status.id)
    for thread in threads:
        thread.join(5.0)
    assert [item.revision for item in seen] == [2, 2, 2]
    # a timeout returns the current snapshot unchanged
    started = time.monotonic()
    same = store.wait(status.id, after=2, timeout=0.05)
    assert same.revision == 2 and time.monotonic() - started < 1.0


def test_cancel_closes_queued_records_and_flags_running_ones(store):
    queued = create(store)
    cancelled = store.request_cancel(queued.id)
    assert cancelled.state == "cancelled" and cancelled.done
    assert cancelled.error["type"] == "Cancelled"
    assert store.next_queued() is None

    running = create(store)
    epoch = store.begin(running.id)
    flagged = store.request_cancel(running.id)
    assert flagged.state == "planning" and flagged.cancel_requested
    revision = flagged.revision
    assert store.request_cancel(running.id).revision == revision
    assert store.finish(running.id, epoch, "cancelled",
                        error={"type": "Cancelled", "message": "stopped"})
    assert store.request_cancel(running.id).state == "cancelled"


def test_recover_marks_active_records_interrupted_and_keeps_queued(tmp_path):
    path = tmp_path / "quail.sqlite3"
    store = Store(path)
    queued = create(store)
    active = create(store)
    finished = create(store)
    store.begin(active.id)
    epoch = store.begin(finished.id)
    store.finish(finished.id, epoch, "succeeded", result={"rows": 0})
    store.close()

    reopened = Store(path)
    assert reopened.recover() == [active.id]
    assert reopened.get(active.id).state == "interrupted"
    assert "restarted" in reopened.get(active.id).error["message"]
    assert reopened.get(queued.id).state == "queued"
    assert reopened.get(finished.id).state == "succeeded"
    assert reopened.next_queued().id == queued.id
    assert reopened.recover() == []
    reopened.close()


def test_inputs_are_saved_once(store):
    assert store.get_input("abc") is None
    store.put_input("abc", "/data/inputs/abc.arrow", 12)
    store.put_input("abc", "/other", 99)
    record = store.get_input("abc")
    assert (record.path, record.byte_count) == ("/data/inputs/abc.arrow", 12)
