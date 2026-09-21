"""The client side of a remote Session: submit, watch, cancel, fetch.

Uses only the standard library for HTTP, so a client installation
needs no service extra. ``ServiceClient`` wraps the routes;
``RemoteQuery`` and ``QueryRun`` are what ``Session.sql`` and
``Session.get_run`` return when the session has an endpoint.
"""

from __future__ import annotations

import http.client
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from quail.execution.result import QueryResult
from quail.progress import say
from quail.service.artifacts import result_from_parts
from quail.service.inputs import PreparedInput, describe
from quail.service.records import QueryFailedError, QueryStatus, ServiceError

DEFAULT_POLL_S = 30.0


class ServiceClient:
    """HTTP calls to one query service."""

    def __init__(self, endpoint: str, token: str | None = None,
                 timeout_s: float = 120.0):
        self.endpoint = endpoint.rstrip("/")
        self.token = token if token is not None else os.environ.get(
            "QUAIL_SERVICE_TOKEN")
        self.timeout_s = timeout_s

    def _headers(self, extra: dict | None = None) -> dict:
        headers = {"accept": "application/json"}
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        headers.update(extra or {})
        return headers

    def _request(self, method: str, path: str, *, body=None, headers=None,
                 timeout: float | None = None, raw: bool = False):
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
            response = urllib.request.urlopen(
                request, timeout=timeout or self.timeout_s)
        except urllib.error.HTTPError as error:
            raise _service_error(error) from None
        except (urllib.error.URLError, http.client.HTTPException,
                OSError) as error:
            raise ServiceError(
                f"cannot reach the Quail service at {self.endpoint}: "
                f"{error}") from error
        with response:
            payload = response.read()
        if raw:
            return payload
        return json.loads(payload) if payload else None

    def capabilities(self) -> dict:
        return self._request("GET", "/v1/capabilities")

    def has_input(self, content_id: str) -> bool:
        try:
            self._request("HEAD", f"/v1/inputs/{content_id}")
        except ServiceError as error:
            if error.status == 404:
                return False
            raise
        return True

    def upload_input(self, prepared: PreparedInput) -> None:
        """Upload one snapshot unless the service already has it."""
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


def _service_error(error: urllib.error.HTTPError) -> ServiceError:
    try:
        detail = json.loads(error.read().decode("utf-8"))["error"]
        message = f"{detail['type']}: {detail['message']}"
    except (ValueError, KeyError, TypeError):
        message = f"HTTP {error.code}"
    failure = ServiceError(message)
    failure.status = error.code
    return failure


def _table(payload: bytes) -> pa.Table:
    with ipc.open_file(pa.BufferReader(payload)) as reader:
        return reader.read_all()


class QueryRun:
    """A handle to one accepted execution on the service."""

    def __init__(self, client: ServiceClient, query_id: str, codecs=None):
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
        raises ServiceError; calling watch() again resumes from the
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
        """Return the join answers saved so far, from entry ``after`` on.

        Each entry is one finished anchor: ``document`` (its row index in
        the anchor table), ``matches`` (the partner row index tuples that
        answered true), and ``asked`` (how many pairs were evaluated; the
        rest answered false). ``next`` is the entry to ask for next and
        ``done`` says whether more can still arrive.
        """
        return self._client.answers(self.id, after=after, limit=limit)

    def stream_answers(self, poll_s: float = DEFAULT_POLL_S
                       ) -> Iterator[dict]:
        """Yield each anchor's answers as the service saves them.

        Ends when the query is done and every saved entry was yielded.
        Filters and scores do not stream; they arrive with the result.
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
        """Ask the service to stop this query. Returns the snapshot after."""
        return self._client.cancel(self.id)

    def result(self, poll_s: float = DEFAULT_POLL_S) -> QueryResult:
        """Wait for completion and return the saved result.

        Raises QueryFailedError when the record ended failed, interrupted,
        or cancelled.
        """
        for status in self.watch(poll_s):
            if status.done:
                break
        if status.state != "succeeded":
            raise QueryFailedError(status)
        files = status.result["files"]
        table = _table(self._client.file(self.id, files["result"]))
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
    """What a Session with an endpoint holds: the client and its inputs."""

    def __init__(self, endpoint: str, codecs, *, token: str | None = None,
                 resolve_revision=None):
        self.client = ServiceClient(endpoint, token=token)
        self.codecs = codecs
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

    def submit(self, spec: dict, config: dict, *, request_key: str | None,
               timeout_s: float | None) -> QueryRun:
        for prepared in self.inputs.values():
            self.client.upload_input(prepared)
        body = {
            **spec,
            "config": config,
            "inputs": {name: prepared.spec
                       for name, prepared in self.inputs.items()},
            "request_key": request_key,
            "timeout_s": timeout_s,
        }
        status = self.client.submit(body)
        return self.run(status.id)


class RemoteQuery:
    """A query held as text until the service compiles and plans it."""

    def __init__(self, session, sql: str, *, order: str | None,
                 dialect: str):
        self.session = session
        self.sql = sql
        self.order = order
        self.dialect = dialect

    def _spec(self) -> dict:
        return {"sql": self.sql, "order": self.order, "dialect": self.dialect}

    def submit(self, *, request_key: str | None = None,
               timeout_s: float | None = None) -> QueryRun:
        """Submit to the service and return once it has saved the record.

        Args:
            request_key: A client-chosen key; resubmitting with the same
                key and the same query returns the same run.
            timeout_s: Execution time limit; the service default when
                omitted.
        """
        config = self.session.config
        run = self.session._remote.submit(
            self._spec(),
            {"model": config.model, "device": config.device,
             "gpus": config.gpus, "backend": config.backend},
            request_key=request_key, timeout_s=timeout_s)
        say(f"submitted query {run.id} to {self.session._remote.client.endpoint}")
        return run

    def run(self, plan=None, *, timeout_s: float | None = None) -> QueryResult:
        """Submit, wait for the saved result, and return it."""
        if plan is not None:
            raise RuntimeError(
                "a remote query runs the service's plan; edited plans are "
                "not supported with an endpoint")
        return self.submit(timeout_s=timeout_s).result()

    def collect(self, limit: int | None = None,
                batch_rows: int = 65_536) -> pa.Table:
        return self.run().collect(limit=limit, batch_rows=batch_rows)

    def explain(self, **_options) -> str:
        raise RuntimeError(
            "explain() is not available on a remote query before it runs; "
            "submit() it and read run.status().plan['text']")

    def plan(self):
        raise RuntimeError(
            "plan() is not available on a remote query; the service plans "
            "it and saves the plan on the run's status")
