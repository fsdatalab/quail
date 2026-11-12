"""The client side of a remote Session: Arrow Flight calls to Quail Server.

``ServerClient`` wraps the server's Flight calls. ``RemoteQuery`` and
``QueryRun`` are what ``Session.sql`` and ``Session.get_run`` return when
the session has an endpoint.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, fields

import pyarrow as pa
import pyarrow.flight as flight

from quail.catalog.providers import ScanRequest
from quail.execution.result import QueryResult

DEFAULT_WAIT_S = 1.0
TOKEN_VARIABLE = "QUAIL_SERVER_TOKEN"
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
        revision: A number that grows each time the status changes.
        plans: Each model region's plan once planned: its ``text`` and
            the planner's ``estimated_seconds``.
        ready: Whether the workers have had the model loaded.
        num_answers: How many answer events the run has given so far.
        tables: The operator port tables saved once the query
            succeeded: each one's ``region``, ``node``, ``port``, and
            ``num_rows``.
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
    progress: list | None = None
    revision: int = 0
    plans: tuple = ()
    ready: bool = False
    num_answers: int = 0
    tables: tuple = ()

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
        endpoint: The server's address, such as ``grpc://127.0.0.1:50051``.
        token: The server's token; ``QUAIL_SERVER_TOKEN`` when omitted.
    """

    def __init__(self, endpoint: str, token: str | None = None):
        self.endpoint = endpoint
        self._flight = flight.FlightClient(endpoint)
        token = token if token is not None else os.environ.get(TOKEN_VARIABLE)
        self._options = flight.FlightCallOptions(headers=[] if not token else [
            (b"authorization", f"Bearer {token}".encode())])

    def close(self) -> None:
        """Close the connection."""
        self._flight.close()

    def _act(self, name: str, body) -> bytes:
        text = body if isinstance(body, str) else json.dumps(body)
        answer, = self._flight.do_action(
            flight.Action(name, text.encode()), self._options)
        return answer.body.to_pybytes()

    def upload(self, name: str, reader: pa.RecordBatchReader,
               id_col: str) -> None:
        """Register ``reader``'s rows on the server as the table ``name``."""
        descriptor = flight.FlightDescriptor.for_path(name, id_col)
        writer, _ = self._flight.do_put(
            descriptor, reader.schema, self._options)
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

    def status(self, query_id: str, *, after: int | None = None,
               wait_s: float = 0.0) -> QueryStatus:
        """Return a submission's status.

        Args:
            query_id: The submission's id.
            after: A revision. The server answers once the status is
                newer, or after ``wait_s`` seconds at the latest.
            wait_s: The most seconds the server waits, 60 at most.
        """
        if after is None:
            return _status(self._act("status", query_id))
        return _status(self._act("status", {
            "id": query_id, "after": after, "wait_s": wait_s}))

    def answers(self, query_id: str, *, after: int = 0,
                limit: int = 1000) -> dict:
        """Return a run's answer events from the one numbered ``after`` on.

        Returns:
            ``answers``, the events as the engine reported them, and
            ``next``, the number to pass as ``after`` for the events
            that follow.
        """
        return json.loads(self._act("answers", {
            "id": query_id, "after": after, "limit": limit}))

    def report(self, query_id: str) -> list[dict]:
        """Return each model region's report of a query that succeeded."""
        return json.loads(self._act("report", query_id))["regions"]

    def table(self, query_id: str, index: int) -> pa.Table:
        """Return the operator port table a status lists at ``index``."""
        ticket = flight.Ticket(json.dumps({
            "query_id": query_id, "table": index}).encode())
        return self._flight.do_get(ticket, self._options).read_all()

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
        return self._flight.do_get(ticket, self._options).to_reader()


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

    def watch(self, wait_s: float = DEFAULT_WAIT_S) -> Iterator[QueryStatus]:
        """Yield the query's status each time it changes, until it finishes.

        Args:
            wait_s: The most seconds between two statuses.
        """
        status = self.status()
        while True:
            yield status
            if status.done:
                return
            status = self._client.status(
                self.id, after=status.revision, wait_s=wait_s)

    def cancel(self) -> QueryStatus:
        """Stop the query."""
        return self._client.cancel(self.id)

    def answers(self, after: int = 0, limit: int = 1000) -> dict:
        """Return the run's answer events, as ``ServerClient.answers`` does."""
        return self._client.answers(self.id, after=after, limit=limit)

    def report(self) -> list[dict]:
        """Return each model region's report, once the query succeeded."""
        return self._client.report(self.id)

    def wait(self, wait_s: float = DEFAULT_WAIT_S) -> QueryStatus:
        """Wait for the query to finish; return its last status."""
        for status in self.watch(wait_s):
            pass
        return status

    def stream_result(self, wait_s: float = DEFAULT_WAIT_S
                      ) -> pa.RecordBatchReader:
        """Wait for the query, then stream its rows.

        Raises:
            QueryFailedError: The query failed or was cancelled.
        """
        status = self.wait(wait_s)
        if status.state != "succeeded":
            raise QueryFailedError(status)
        return self._client.result_batches(self.id)

    def result(self, wait_s: float = DEFAULT_WAIT_S) -> QueryResult:
        """Wait for the query, then return its rows.

        Raises:
            QueryFailedError: The query failed or was cancelled.
        """
        return QueryResult.from_table(self.stream_result(wait_s).read_all())


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
