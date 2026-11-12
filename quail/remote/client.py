"""The client side of a remote Session: Arrow Flight calls to Quail Server.

``ServerClient`` wraps the server's Flight calls. ``RemoteQuery`` and
``QueryRun`` are what ``Session.sql`` and ``Session.get_run`` return when
the session has an endpoint.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, fields

import pyarrow as pa
import pyarrow.flight as flight

from quail.catalog.providers import ScanRequest
from quail.execution.result import QueryResult

DEFAULT_POLL_S = 1.0
FINISHED_STATES = frozenset({"succeeded", "failed", "cancelled"})


@dataclass(frozen=True)
class QueryStatus:
    """The server's saved status of one submitted query.

    Attributes:
        id: The submission's id.
        sql: The query.
        state: "queued", "running", "succeeded", "failed", or "cancelled".
        dialect: The SQL dialect, or None for Snowflake.
        error: Why the query failed or was cancelled.
        timeout_s: The most seconds the query may run.
        submitted_at: Seconds since 1970 when the server took the query.
        started_at: Seconds since 1970 when the query started.
        finished_at: Seconds since 1970 when the query finished.
        num_rows: How many rows the query returned.
        progress: The running model region's progress, while it runs.
    """

    id: str
    sql: str
    state: str
    dialect: str | None = None
    error: str | None = None
    timeout_s: float | None = None
    submitted_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    num_rows: int | None = None
    progress: dict | None = None

    @property
    def done(self) -> bool:
        """Return whether the query has finished, whatever its state."""
        return self.state in FINISHED_STATES


class QueryFailedError(RuntimeError):
    """Raised by result() when the query finished without rows."""

    def __init__(self, status: QueryStatus):
        self.status = status
        super().__init__(
            f"query {status.id} {status.state}: {status.error or status.state}")


def _status(body: bytes) -> QueryStatus:
    found = json.loads(body)
    names = {field.name for field in fields(QueryStatus)}
    return QueryStatus(**{name: found[name] for name in names if name in found})


class ServerClient:
    """Arrow Flight calls to one Quail Server.

    Args:
        endpoint: The server's address, such as ``grpc://127.0.0.1:8642``.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._flight = flight.FlightClient(endpoint)

    def close(self) -> None:
        """Close the connection."""
        self._flight.close()

    def _act(self, name: str, body) -> bytes:
        text = body if isinstance(body, str) else json.dumps(body)
        answer, = self._flight.do_action(flight.Action(name, text.encode()))
        return answer.body.to_pybytes()

    def upload(self, name: str, reader: pa.RecordBatchReader,
               id_col: str) -> None:
        """Register ``reader``'s rows on the server as the table ``name``."""
        descriptor = flight.FlightDescriptor.for_path(name, id_col)
        writer, _ = self._flight.do_put(descriptor, reader.schema)
        try:
            for batch in reader:
                writer.write_batch(batch)
        finally:
            writer.close()

    def submit(self, sql: str, *, dialect: str | None = None,
               query_id: str | None = None,
               timeout_s: float | None = None) -> QueryStatus:
        """Queue a query; submitting one id twice runs the query once."""
        asked = {"sql": sql, "dialect": dialect, "id": query_id,
                 "timeout_s": timeout_s}
        return _status(self._act("submit", {
            key: value for key, value in asked.items() if value is not None}))

    def status(self, query_id: str) -> QueryStatus:
        """Return a submission's status."""
        return _status(self._act("status", query_id))

    def cancel(self, query_id: str) -> QueryStatus:
        """Stop a submission."""
        return _status(self._act("cancel", query_id))

    def submissions(self) -> list[QueryStatus]:
        """Return every submission the server has."""
        return [_status(json.dumps(found))
                for found in json.loads(self._act("list", ""))]

    def explain(self, sql: str, dialect: str | None = None) -> str:
        """Return the server's plan of a query."""
        asked = {"sql": sql} if dialect is None else {
            "sql": sql, "dialect": dialect}
        return self._act("explain", asked).decode()

    def result_batches(self, query_id: str) -> pa.RecordBatchReader:
        """Return the saved rows of a submission that succeeded."""
        ticket = flight.Ticket(json.dumps({"query_id": query_id}).encode())
        return self._flight.do_get(ticket).to_reader()


class QueryRun:
    """One query the server took: its status, and its rows once it succeeds."""

    def __init__(self, client: ServerClient, query_id: str):
        self._client = client
        self.id = query_id

    def __repr__(self) -> str:
        return f"QueryRun({self.id!r})"

    def status(self) -> QueryStatus:
        """Return the query's status now."""
        return self._client.status(self.id)

    def watch(self, poll_s: float = DEFAULT_POLL_S) -> Iterator[QueryStatus]:
        """Yield the query's status every ``poll_s`` seconds until it finishes."""
        while True:
            status = self.status()
            yield status
            if status.done:
                return
            time.sleep(poll_s)

    def cancel(self) -> QueryStatus:
        """Stop the query."""
        return self._client.cancel(self.id)

    def wait(self, poll_s: float = DEFAULT_POLL_S) -> QueryStatus:
        """Wait for the query to finish; return its last status."""
        for status in self.watch(poll_s):
            pass
        return status

    def stream_result(self, poll_s: float = DEFAULT_POLL_S
                      ) -> pa.RecordBatchReader:
        """Wait for the query, then stream its rows.

        Raises:
            QueryFailedError: The query failed or was cancelled.
        """
        status = self.wait(poll_s)
        if status.state != "succeeded":
            raise QueryFailedError(status)
        return self._client.result_batches(self.id)

    def result(self, poll_s: float = DEFAULT_POLL_S) -> QueryResult:
        """Wait for the query, then return its rows.

        Raises:
            QueryFailedError: The query failed or was cancelled.
        """
        return QueryResult.from_table(self.stream_result(poll_s).read_all())


class RemoteConnection:
    """A remote Session's connection: its tables and queries on one server."""

    def __init__(self, endpoint: str):
        self.client = ServerClient(endpoint)

    def register(self, name: str, provider) -> None:
        """Upload a provider's rows as the server's table ``name``."""
        reader = provider.scan(ScanRequest(columns=provider.columns))
        try:
            self.client.upload(name, reader, provider.id_col)
        finally:
            reader.close()

    def close(self) -> None:
        """Close the connection; the server keeps its tables and queries."""
        self.client.close()

    def run(self, query_id: str) -> QueryRun:
        """Return the run of a query the server already has."""
        return QueryRun(self.client, query_id)


class RemoteQuery:
    """A SQL query that Quail Server plans and runs."""

    def __init__(self, session, sql: str, *, dialect: str):
        self.session = session
        self.text = sql
        self.dialect = dialect

    def submit(self, *, query_id: str | None = None,
               timeout_s: float | None = None) -> QueryRun:
        """Queue the query on the server; return its run at once.

        Args:
            query_id: A name for the submission. Submitting the same
                query under one id twice runs it once.
            timeout_s: The most seconds the query may run.
        """
        client = self.session._remote.client
        status = client.submit(self.text, dialect=self.dialect,
                               query_id=query_id, timeout_s=timeout_s)
        return QueryRun(client, status.id)

    def run(self, plan=None, *, timeout_s: float | None = None) -> QueryResult:
        """Run the query on the server and wait for its rows.

        Raises:
            ValueError: ``plan`` is given; the server plans its queries.
            QueryFailedError: The query failed or was cancelled.
        """
        if plan is not None:
            raise ValueError("a remote query runs the server's plan")
        return self.submit(timeout_s=timeout_s).result()

    def collect(self, limit: int | None = None,
                batch_rows: int = 65_536) -> pa.Table:
        """Run the query and return its rows."""
        return self.run().collect(limit=limit, batch_rows=batch_rows)

    def execute_stream(self, batch_rows: int = 65_536,
                       limit: int | None = None) -> pa.RecordBatchReader:
        """Run the query and stream its rows."""
        return self.run().execute_stream(batch_rows=batch_rows, limit=limit)

    def explain(self, **_options) -> str:
        """Return the server's plan, each model region as one operator."""
        return self.session._remote.client.explain(self.text, self.dialect)

    def plan(self):
        """A remote query has no local plan; ``explain()`` shows the server's."""
        raise RuntimeError(
            "a remote query is planned on the server; call explain()")
