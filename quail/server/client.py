"""The client side of a remote Session: submit, watch, cancel, fetch.

Uses only the standard library for HTTP, so a client installation
needs no server extra. Control messages are JSON. Table data travels
as Arrow IPC: an input snapshot is uploaded from its file, and a
result is read as a stream of record batches. ``ServerClient`` wraps
the routes; ``RemoteQuery`` and ``QueryRun`` are what ``Session.sql``
and ``Session.get_run`` return when the session has an endpoint.
"""

from __future__ import annotations

import http.client
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from quail.execution.result import QueryResult
from quail.progress import say
from quail.server.artifacts import result_from_parts
from quail.server.inputs import PreparedInput, describe
from quail.server.records import QueryFailedError, QueryStatus, ServerError

DEFAULT_POLL_S = 30.0
TOKEN_VARIABLE = "QUAIL_SERVER_TOKEN"


class ServerClient:
    """HTTP calls to one Quail Server."""

    def __init__(self, endpoint: str, token: str | None = None,
                 timeout_s: float = 120.0):
        self.endpoint = endpoint.rstrip("/")
        self.token = token if token is not None else os.environ.get(
            TOKEN_VARIABLE)
        self.timeout_s = timeout_s

    def _headers(self, extra: dict | None = None) -> dict:
        headers = {"accept": "application/json"}
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        headers.update(extra or {})
        return headers

    def _open(self, method: str, path: str, *, body=None, headers=None,
              timeout: float | None = None):
        """Send one request and return the open response."""
        data = None
        request_headers = self._headers(headers)
        if body is not None and not isinstance(body, (bytes, bytearray)) \
                and not hasattr(body, "read"):
            data = json.dumps(body).encode("utf-8")
            request_headers["content-type"] = "application/json"
        elif body is not None:
            data = body
        request = urllib.request.Request(
            self.endpoint + path, data=data, method=method,
            headers=request_headers)
        try:
            return urllib.request.urlopen(
                request, timeout=timeout or self.timeout_s)
        except urllib.error.HTTPError as error:
            raise _server_error(error) from None
        except (urllib.error.URLError, http.client.HTTPException,
                OSError) as error:
            raise ServerError(
                f"cannot reach Quail Server at {self.endpoint}: "
                f"{error}") from error

    def _request(self, method: str, path: str, *, body=None, headers=None,
                 timeout: float | None = None, raw: bool = False):
        with self._open(method, path, body=body, headers=headers,
                        timeout=timeout) as response:
            payload = response.read()
        if raw:
            return payload
        return json.loads(payload) if payload else None

    def capabilities(self) -> dict:
        return self._request("GET", "/v1/capabilities")

    def has_input(self, content_id: str) -> bool:
        try:
            self._request("HEAD", f"/v1/inputs/{content_id}")
        except ServerError as error:
            if error.status == 404:
                return False
            raise
        return True

    def upload_input(self, prepared: PreparedInput) -> None:
        """Upload one snapshot unless the server already has it."""
        if prepared.upload_path is None or self.has_input(prepared.content_id):
            return
        size = prepared.upload_path.stat().st_size
        with open(prepared.upload_path, "rb") as handle:
            self._request(
                "PUT", f"/v1/inputs/{prepared.content_id}", body=handle,
                headers={"content-type": "application/vnd.apache.arrow.file",
                         "content-length": str(size)},
                timeout=max(self.timeout_s, size / (1 << 20)))

    def submit(self, body: dict) -> QueryStatus:
        return QueryStatus.from_dict(self._request("POST", "/v1/queries",
                                                   body=body))

    def status(self, query_id: str, *, after: int | None = None,
               wait: float = 0.0) -> QueryStatus:
        query = ""
        if after is not None and wait > 0:
            query = "?" + urllib.parse.urlencode(
                {"after": after, "wait": f"{wait:g}"})
        data = self._request("GET", f"/v1/queries/{query_id}{query}",
                             timeout=self.timeout_s + wait)
        return QueryStatus.from_dict(data)

    def cancel(self, query_id: str) -> QueryStatus:
        return QueryStatus.from_dict(
            self._request("POST", f"/v1/queries/{query_id}/cancel"))

    def answers(self, query_id: str, *, after: int = 0,
                limit: int = 1000) -> dict:
        query = urllib.parse.urlencode({"after": after, "limit": limit})
        return self._request("GET", f"/v1/queries/{query_id}/answers?{query}")

    def file(self, query_id: str, name: str) -> bytes:
        return self._request("GET", f"/v1/queries/{query_id}/files/{name}",
                             raw=True)

    def result_batches(self, query_id: str) -> pa.RecordBatchReader:
        """Open the saved result as a stream of Arrow record batches.

        The connection stays open until the reader is exhausted or
        closed.
        """
        response = self._open(
            "GET", f"/v1/queries/{query_id}/result",
            headers={"accept": "application/vnd.apache.arrow.stream"})
        reader = ipc.open_stream(pa.PythonFile(response, mode="r"))

        def batches():
            try:
                yield from reader
            finally:
                reader.close()
                response.close()

        return pa.RecordBatchReader.from_batches(reader.schema, batches())


def _server_error(error: urllib.error.HTTPError) -> ServerError:
    try:
        detail = json.loads(error.read().decode("utf-8"))["error"]
        message = f"{detail['type']}: {detail['message']}"
    except (ValueError, KeyError, TypeError):
        message = f"HTTP {error.code}"
    failure = ServerError(message)
    failure.status = error.code
    return failure


def _table(payload: bytes) -> pa.Table:
    with ipc.open_file(pa.BufferReader(payload)) as reader:
        return reader.read_all()


def _limited(reader: pa.RecordBatchReader, limit: int) -> pa.RecordBatchReader:
    """The first ``limit`` rows of a reader, then it is closed."""

    def batches():
        left = limit
        try:
            for batch in reader:
                if left <= 0:
                    break
                yield batch.slice(0, min(left, batch.num_rows))
                left -= batch.num_rows
        finally:
            reader.close()

    return pa.RecordBatchReader.from_batches(reader.schema, batches())


class QueryRun:
    """A handle to one accepted execution on the server."""

    def __init__(self, client: ServerClient, query_id: str, codecs=None):
        self._client = client
        self.id = query_id
        self._codecs = codecs

    def __repr__(self) -> str:
        return f"QueryRun({self.id!r})"

    def status(self) -> QueryStatus:
        """Return the current saved status snapshot."""
        return self._client.status(self.id)

    def watch(self, poll_s: float = DEFAULT_POLL_S) -> Iterator[QueryStatus]:
        """Yield the current snapshot, then each newer revision, until done.

        Intermediate progress snapshots may be skipped. A lost connection
        raises ServerError; calling watch() again resumes from the
        current snapshot.
        """
        status = self.status()
        yield status
        while not status.done:
            newer = self._client.status(self.id, after=status.revision,
                                        wait=poll_s)
            if newer.revision > status.revision:
                status = newer
                yield status

    def answers(self, after: int = 0, limit: int = 1000) -> dict:
        """Return the answers saved so far, from entry ``after`` on.

        Each entry's ``kind`` says what finished. A ``"filter"`` entry is
        one chunk of documents: ``documents`` lists ``[row index, last
        stage asked, passed]``, and a document passed the filter when it
        passed its last stage. A ``"join"`` entry is one finished anchor:
        ``document`` (its row index in the anchor table), ``matches`` (the
        partner row index tuples that answered true), and ``asked`` (how
        many pairs were evaluated; the rest answered false). A ``"score"``
        entry is one batch of reranker scores: ``rows`` and ``scores``
        line up. An ``"evict"`` entry is one document prefix dropped
        from KV: ``alias``, ``document`` (its row index), and
        ``tokens`` (the prefix length). ``next`` is the entry to ask
        for next and ``done`` says whether more can still arrive.
        """
        return self._client.answers(self.id, after=after, limit=limit)

    def stream_answers(self, poll_s: float = DEFAULT_POLL_S
                       ) -> Iterator[dict]:
        """Yield each saved answer entry as the server saves it.

        Ends when the query is done and every saved entry was yielded.
        See ``answers`` for the entry kinds.
        """
        seen = 0
        for status in self.watch(poll_s):
            saved = (status.progress or {}).get("answers_saved", 0)
            if saved <= seen and not status.done:
                continue
            while True:
                page = self.answers(after=seen)
                yield from page["answers"]
                seen = page["next"]
                if not page["answers"]:
                    break
            if status.done:
                return

    def cancel(self) -> QueryStatus:
        """Ask the server to stop this query. Returns the snapshot after."""
        return self._client.cancel(self.id)

    def wait(self, poll_s: float = DEFAULT_POLL_S) -> QueryStatus:
        """Wait for the query to end and return its final snapshot.

        Raises QueryFailedError when the record ended failed, interrupted,
        or cancelled.
        """
        for status in self.watch(poll_s):
            if status.done:
                break
        if status.state != "succeeded":
            raise QueryFailedError(status)
        return status

    def stream_result(self, poll_s: float = DEFAULT_POLL_S
                      ) -> pa.RecordBatchReader:
        """Wait for completion and stream the result rows batch by batch."""
        self.wait(poll_s)
        return self._client.result_batches(self.id)

    def result(self, poll_s: float = DEFAULT_POLL_S) -> QueryResult:
        """Wait for completion and return the saved result.

        Raises QueryFailedError when the record ended failed, interrupted,
        or cancelled.
        """
        status = self.wait(poll_s)
        files = status.result["files"]
        table = self._client.result_batches(self.id).read_all()
        report = json.loads(self._client.file(self.id, files["report"]))
        filters = {
            (entry["alias"], entry["position"]):
                _table(self._client.file(self.id, entry["file"]))
            for entry in files["answers"]["filters"]
        }
        joins = {
            entry["position"]: _table(self._client.file(self.id, entry["file"]))
            for entry in files["answers"]["joins"]
        }
        return result_from_parts(table, report, filters, joins, self._codecs)


class RemoteConnection:
    """What a Session with an endpoint holds: the client and its inputs.

    ``session_id`` is chosen here and sent with every submission, so
    the server can list the queries one client session submitted.
    """

    def __init__(self, endpoint: str, codecs, *, token: str | None = None,
                 resolve_revision=None):
        self.client = ServerClient(endpoint, token=token)
        self.codecs = codecs
        self.session_id = uuid.uuid4().hex
        self._resolve_revision = resolve_revision
        self._workdir = tempfile.TemporaryDirectory(prefix="quail-inputs-")
        self.inputs: dict[str, PreparedInput] = {}

    def register(self, name: str, provider) -> None:
        options = {}
        if self._resolve_revision is not None:
            options["resolve_revision"] = self._resolve_revision
        self.inputs[name] = describe(provider, Path(self._workdir.name),
                                     **options)

    def close(self) -> None:
        self._workdir.cleanup()

    def run(self, query_id: str) -> QueryRun:
        return QueryRun(self.client, query_id, self.codecs)

    def submit(self, spec: dict, config: dict, *, query_id: str,
               timeout_s: float | None) -> QueryRun:
        for prepared in self.inputs.values():
            self.client.upload_input(prepared)
        body = {
            **spec,
            "config": config,
            "inputs": {name: prepared.spec
                       for name, prepared in self.inputs.items()},
            "query_id": query_id,
            "session_id": self.session_id,
            "timeout_s": timeout_s,
        }
        status = self.client.submit(body)
        return self.run(status.id)


class RemoteQuery:
    """A query held as text until the server compiles and plans it."""

    def __init__(self, session, sql: str, *, order: str | None,
                 dialect: str):
        self.session = session
        self.sql = sql
        self.order = order
        self.dialect = dialect

    def _spec(self) -> dict:
        return {"sql": self.sql, "order": self.order, "dialect": self.dialect}

    def submit(self, *, query_id: str | None = None,
               timeout_s: float | None = None) -> QueryRun:
        """Submit to the server and return once it has saved the record.

        Args:
            query_id: The id the run will have; a new random one when
                omitted. Submitting the same id and the same query
                again returns the same run, so a retry after a lost
                response is safe. The same id with a different query
                is an error.
            timeout_s: Execution time limit; the server default when
                omitted.
        """
        config = self.session.config
        run = self.session._remote.submit(
            self._spec(),
            {"model": config.model, "device": config.device,
             "gpus": config.gpus, "backend": config.backend},
            query_id=query_id or uuid.uuid4().hex, timeout_s=timeout_s)
        say(f"submitted query {run.id} to {self.session._remote.client.endpoint}")
        return run

    def run(self, plan=None, *, timeout_s: float | None = None) -> QueryResult:
        """Submit, wait for the saved result, and return it.

        Args:
            plan: Must be None. Remote queries use the server's plan.
            timeout_s: Execution time limit, or the server default.

        Raises:
            RuntimeError: If an edited plan is passed.
            QueryFailedError: If the saved run does not succeed.
        """
        if plan is not None:
            raise RuntimeError(
                "a remote query runs the server's plan; edited plans are "
                "not supported with an endpoint")
        return self.submit(timeout_s=timeout_s).result()

    def collect(self, limit: int | None = None,
                batch_rows: int = 65_536) -> pa.Table:
        return self.run().collect(limit=limit, batch_rows=batch_rows)

    def execute_stream(self, batch_rows: int = 65_536,
                       limit: int | None = None) -> pa.RecordBatchReader:
        """Submit, wait, and stream the result rows as Arrow record batches.

        The server chooses the batch size; ``batch_rows`` is accepted for
        the local signature and ignored.
        """
        reader = self.submit().stream_result()
        return reader if limit is None else _limited(reader, limit)

    def explain(self, **_options) -> str:
        raise RuntimeError(
            "explain() is not available on a remote query before it runs; "
            "submit() it and read run.status().plan['text']")

    def plan(self):
        raise RuntimeError(
            "plan() is not available on a remote query; the server plans "
            "it and saves the plan on the run's status")
