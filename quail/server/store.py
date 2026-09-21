"""SQLite storage for query records and uploaded inputs.

One Store object is the only writer of its database file. Every
change happens in one transaction and bumps the record's revision, so
a read that starts after a write returns sees that revision or a
newer one. Readers inside the same process can wait for a revision
with ``wait``, or register a listener that is called after every
committed write.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from quail.server.records import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    InvalidRequestError,
    QueryIdConflictError,
    QueryStatus,
    UnknownQueryError,
    check_query_id,
    spec_hash,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    spec_hash TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    state TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    timeout_s REAL NOT NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    execution_epoch INTEGER NOT NULL DEFAULT 0,
    progress_json TEXT,
    plan_json TEXT,
    error_json TEXT,
    result_json TEXT
);
CREATE INDEX IF NOT EXISTS queries_by_state ON queries(state, created_at);
CREATE TABLE IF NOT EXISTS inputs (
    content_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    byte_count INTEGER NOT NULL,
    created_at REAL NOT NULL
);
"""

INTERRUPTED_ERROR = {
    "type": "Interrupted",
    "message": "the server restarted while this query was executing",
}


@dataclass(frozen=True)
class InputRecord:
    """One uploaded input snapshot on the server's disk."""

    content_id: str
    path: str
    byte_count: int


def _dumps(value) -> str | None:
    return None if value is None else json.dumps(value, default=str)


def _loads(text) -> dict | None:
    return None if text is None else json.loads(text)


def _row_status(row) -> QueryStatus:
    return QueryStatus(
        id=row["id"],
        session_id=row["session_id"],
        spec=json.loads(row["spec_json"]),
        config=json.loads(row["config_json"]),
        inputs=json.loads(row["inputs_json"]),
        state=row["state"],
        revision=row["revision"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        timeout_s=row["timeout_s"],
        cancel_requested=bool(row["cancel_requested"]),
        progress=_loads(row["progress_json"]),
        plan=_loads(row["plan_json"]),
        error=_loads(row["error_json"]),
        result=_loads(row["result_json"]),
    )


class Store:
    """Query records and input snapshots in one SQLite file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._listeners = []
        # committed write transactions since open; a checkpoint compares it
        self.write_count = 0
        self.durable_write_count = 0
        self.closed = False
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # WAL lets the scheduler write while status readers read.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript(SCHEMA)
        columns = {row["name"] for row in
                   self._conn.execute("PRAGMA table_info(queries)")}
        if "session_id" not in columns:
            # a database file written before records carried a session id
            self._conn.execute("ALTER TABLE queries ADD COLUMN session_id TEXT")

    def close(self) -> None:
        with self._lock:
            self.closed = True
            self._conn.close()

    def add_listener(self, listener) -> None:
        """Call ``listener()`` after every committed write, on the writer's thread.

        The listener must return at once; it is meant to wake waiters
        elsewhere, not to do work.
        """
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    # -- transactions -----------------------------------------------------

    def _write(self, statement: str, parameters=(), *,
               durable: bool = True) -> int:
        """Run one write statement in its own transaction; return rowcount.

        ``durable=False`` marks a write whose loss on a crash costs
        nothing (a progress counter); ``durable_write_count`` skips it so
        a checkpoint can copy the file less often.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._conn.execute(statement, parameters)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            self.write_count += 1
            if durable:
                self.durable_write_count += 1
            self._changed.notify_all()
            listeners = list(self._listeners)
        for listener in listeners:
            listener()
        return cursor.rowcount

    def backup(self, path: str | Path) -> None:
        """Write a complete, consistent copy of the database to ``path``.

        Uses SQLite's online backup, so the copy includes every committed
        transaction, including those still in the WAL file. The copy is
        written beside ``path`` and renamed into place.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        with self._lock:
            target = sqlite3.connect(str(temporary))
            try:
                self._conn.backup(target)
            finally:
                target.close()
        temporary.replace(path)

    def _row(self, query_id: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM queries WHERE id = ?", (query_id,)).fetchone()
        if row is None:
            raise UnknownQueryError(f"unknown query {query_id!r}")
        return row

    # -- records ----------------------------------------------------------

    def create(self, *, spec: dict, config: dict, inputs: dict,
               timeout_s: float, query_id: str | None = None,
               session_id: str | None = None) -> QueryStatus:
        """Save a new record in the queued state and return its snapshot.

        The client may choose ``query_id``; the store picks one when it is
        None. A repeated id with the same specification returns the
        record it created before, so a retry after a lost response is
        safe. The same id with a different specification raises
        QueryIdConflictError.
        """
        if timeout_s <= 0:
            raise InvalidRequestError("timeout_s must be positive")
        query_id = (uuid.uuid4().hex if query_id is None
                    else check_query_id(query_id))
        digest = spec_hash(spec, config, inputs, timeout_s)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM queries WHERE id = ?", (query_id,)).fetchone()
            if row is not None:
                if row["spec_hash"] != digest:
                    raise QueryIdConflictError(
                        f"query id {query_id!r} was already used for a "
                        "different specification")
                return _row_status(row)
            now = time.time()
            self._write(
                "INSERT INTO queries (id, session_id, spec_hash, spec_json, "
                "config_json, inputs_json, state, revision, created_at, "
                "updated_at, timeout_s) VALUES (?, ?, ?, ?, ?, ?, 'queued', 1, "
                "?, ?, ?)",
                (query_id, session_id, digest, _dumps(spec), _dumps(config),
                 _dumps(inputs), now, now, timeout_s))
            return self.get(query_id)

    def discard(self, query_id: str) -> bool:
        """Remove a record that was never acknowledged to its client.

        A queued record is deleted, so the same id can be submitted
        again. A record the scheduler already started is asked to
        cancel instead. Returns True when the record was deleted.
        """
        with self._lock:
            deleted = self._write(
                "DELETE FROM queries WHERE id = ? AND state = 'queued'",
                (query_id,))
            if deleted == 1:
                return True
            self.request_cancel(query_id)
            return False

    def get(self, query_id: str) -> QueryStatus:
        return _row_status(self._row(query_id))

    def find(self, query_id: str) -> QueryStatus | None:
        """The record with this id, or None."""
        try:
            return self.get(query_id)
        except UnknownQueryError:
            return None

    def execution_epoch(self, query_id: str) -> int:
        return int(self._row(query_id)["execution_epoch"])

    def list_recent(self, limit: int = 50,
                    session_id: str | None = None) -> list[QueryStatus]:
        """The newest records first, all of them or one session's."""
        with self._lock:
            if session_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM queries ORDER BY created_at DESC LIMIT ?",
                    (limit,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM queries WHERE session_id = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit)).fetchall()
        return [_row_status(row) for row in rows]

    def wait(self, query_id: str, after: int, timeout: float) -> QueryStatus:
        """Return the snapshot once its revision passes ``after``.

        Returns the current snapshot when ``timeout`` seconds pass first.
        """
        deadline = time.monotonic() + timeout
        with self._changed:
            while True:
                status = self.get(query_id)
                remaining = deadline - time.monotonic()
                if status.revision > after or remaining <= 0:
                    return status
                self._changed.wait(remaining)

    def wait_for_change(self, timeout: float) -> None:
        """Block until any record changes or ``timeout`` seconds pass."""
        with self._changed:
            self._changed.wait(timeout)

    def next_queued(self) -> QueryStatus | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM queries WHERE state = 'queued' "
                "ORDER BY created_at, rowid LIMIT 1").fetchone()
        return None if row is None else _row_status(row)

    def begin(self, query_id: str) -> int:
        """Move a queued record to planning and return its execution epoch.

        The epoch identifies this execution attempt. Later updates must
        carry it, so an execution the server has already closed cannot
        change the record.
        """
        with self._lock:
            row = self._row(query_id)
            if row["state"] != "queued":
                raise InvalidRequestError(
                    f"query {query_id} is {row['state']}, not queued")
            epoch = int(row["execution_epoch"]) + 1
            now = time.time()
            self._write(
                "UPDATE queries SET state = 'planning', execution_epoch = ?, "
                "started_at = ?, revision = revision + 1, updated_at = ? "
                "WHERE id = ? AND state = 'queued'",
                (epoch, now, now, query_id))
            return epoch

    def update(self, query_id: str, epoch: int, *, state: str | None = None,
               progress: dict | None = None, plan: dict | None = None
               ) -> bool:
        """Record execution progress. Returns False for a closed execution."""
        if state is not None and state not in ACTIVE_STATES:
            raise ValueError(f"update() cannot set state {state!r}")
        assignments = ["revision = revision + 1", "updated_at = ?"]
        parameters: list = [time.time()]
        if state is not None:
            assignments.append("state = ?")
            parameters.append(state)
        if progress is not None:
            assignments.append("progress_json = ?")
            parameters.append(_dumps(progress))
        if plan is not None:
            assignments.append("plan_json = ?")
            parameters.append(_dumps(plan))
        parameters.extend([query_id, epoch, *sorted(ACTIVE_STATES)])
        count = self._write(
            f"UPDATE queries SET {', '.join(assignments)} WHERE id = ? "
            "AND execution_epoch = ? AND state IN (?, ?)", parameters,
            durable=state is not None or plan is not None)
        return count == 1

    def finish(self, query_id: str, epoch: int, state: str, *,
               error: dict | None = None, result: dict | None = None
               ) -> bool:
        """Close an execution. Returns False when the record was closed already.

        Only the current execution epoch can close a record, and a record
        in a terminal state never changes again.
        """
        if state not in TERMINAL_STATES:
            raise ValueError(f"finish() needs a terminal state, not {state!r}")
        if state == "succeeded" and result is None:
            raise ValueError("a succeeded record needs its result manifest")
        count = self._write(
            "UPDATE queries SET state = ?, error_json = ?, result_json = ?, "
            "revision = revision + 1, updated_at = ? WHERE id = ? "
            "AND execution_epoch = ? AND state IN (?, ?)",
            (state, _dumps(error), _dumps(result), time.time(), query_id,
             epoch, *sorted(ACTIVE_STATES)))
        return count == 1

    def request_cancel(self, query_id: str) -> QueryStatus:
        """Cancel a queued record now, or ask a running one to stop.

        A terminal record is returned unchanged.
        """
        with self._lock:
            status = self.get(query_id)
            if status.state == "queued":
                self._write(
                    "UPDATE queries SET state = 'cancelled', error_json = ?, "
                    "revision = revision + 1, updated_at = ? WHERE id = ? "
                    "AND state = 'queued'",
                    (_dumps({"type": "Cancelled",
                             "message": "cancelled before execution started"}),
                     time.time(), query_id))
            elif status.state in ACTIVE_STATES and not status.cancel_requested:
                self._write(
                    "UPDATE queries SET cancel_requested = 1, "
                    "revision = revision + 1, updated_at = ? WHERE id = ?",
                    (time.time(), query_id))
            return self.get(query_id)

    def recover(self) -> list[str]:
        """Mark records left active by an earlier process as interrupted.

        Returns the ids it changed. Queued records stay queued.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM queries WHERE state IN (?, ?)",
                sorted(ACTIVE_STATES)).fetchall()
            ids = [row["id"] for row in rows]
            for query_id in ids:
                self._write(
                    "UPDATE queries SET state = 'interrupted', error_json = ?, "
                    "revision = revision + 1, updated_at = ? WHERE id = ?",
                    (_dumps(INTERRUPTED_ERROR), time.time(), query_id))
        return ids

    # -- inputs -----------------------------------------------------------

    def put_input(self, content_id: str, path: str, byte_count: int) -> None:
        self._write(
            "INSERT OR IGNORE INTO inputs (content_id, path, byte_count, "
            "created_at) VALUES (?, ?, ?, ?)",
            (content_id, path, byte_count, time.time()))

    def get_input(self, content_id: str) -> InputRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM inputs WHERE content_id = ?",
                (content_id,)).fetchone()
        if row is None:
            return None
        return InputRecord(row["content_id"], row["path"], row["byte_count"])
