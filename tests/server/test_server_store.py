"""Saved query records: revisions, query ids, epochs, cancel, recovery."""

import json
from dataclasses import asdict

import pytest

from quail.remote.records import (
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
    defaults = dict(spec=SPEC, config=CONFIG, inputs=INPUTS, timeout_s=1000.0)
    return store.create(**{**defaults, **overrides})


def test_create_saves_a_complete_snapshot_and_client_ids_return_it_or_conflict(
        store):
    status = create(store)
    assert (status.state, status.revision, status.timeout_s) == ("queued", 1, 1000.0)
    assert status.spec == SPEC and status.config == CONFIG
    assert status.inputs == INPUTS and not status.done
    assert store.get(status.id) == status
    # the snapshot round trips through JSON without Quail objects
    assert QueryStatus(**json.loads(json.dumps(asdict(status)))) == status
    with pytest.raises(UnknownQueryError):
        store.get("missing")
    with pytest.raises(InvalidRequestError):
        create(store, timeout_s=0)

    first = create(store, query_id="demo-1")
    assert first.id == "demo-1"
    assert create(store, query_id="demo-1") == first
    assert len(store.list_recent()) == 2
    with pytest.raises(QueryIdConflictError):
        create(store, spec={**SPEC, "sql": "SELECT 1"}, query_id="demo-1")
    assert create(store, query_id="demo-2").id != first.id
    assert len(create(store).id) == 32, "the store picks an id when none is sent"
    for bad in ("", "-x", "a/b", "a" * 129, 7):
        with pytest.raises(InvalidRequestError, match="query id"):
            create(store, query_id=bad)
    assert store.find("demo-1") == first
    assert store.find("nope") is None
