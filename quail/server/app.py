"""The HTTP application of Quail Server.

Needs the ``server`` installation extra (Starlette and uvicorn). The
Python client in ``quail.server.client`` and a browser both talk to
these routes. Control messages are JSON; table data is Arrow IPC.
Status reads return the complete saved snapshot; a client can wait for
a newer revision with ``?after=<revision>&wait=<s>`` or subscribe with
server-sent events at ``/events``. Waiting happens on the event loop,
so a waiting reader does not hold a worker thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.ipc as ipc
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

from quail.builtins import built_in_registry
from quail.execution.session import RefusalError, Session
from quail.logical import CompileError
from quail.planner.plan import EngineConfig
from quail.server.artifacts import (
    ANSWERS_FILE,
    RESULT_FILE,
    iter_ipc_stream,
    read_answers,
)
from quail.server.executor import (
    ChildProcessExecutor,
    Hooks,
    InProcessExecutor,
)
from quail.server.inputs import resolve
from quail.server.records import (
    InvalidRequestError,
    NotReadyError,
    QueryStatus,
    ServerError,
    check_query_id,
)
from quail.server.scheduler import DEFAULT_TIMEOUT_S, Scheduler
from quail.server.store import Store

MAX_WAIT_S = 60.0
SSE_KEEPALIVE_S = 15.0
STATIC_DIR = Path(__file__).parent / "static"
ARROW_STREAM = "application/vnd.apache.arrow.stream"
ARROW_FILE = "application/vnd.apache.arrow.file"


@dataclass(frozen=True)
class ServerSettings:
    """What one deployed server can run and where it keeps its data."""

    data_dir: Path
    models: tuple[str, ...]
    device: str
    gpus: tuple[int, ...] = (1,)
    backends: tuple[str, ...] = ("quail",)
    default_timeout_s: float = DEFAULT_TIMEOUT_S
    max_timeout_s: float = 4 * 3600.0
    max_upload_bytes: int = 8 << 30
    token: str | None = None
    # the live SQLite file; default data_dir/quail.sqlite3. Set it to a
    # local disk when data_dir is a network file system such as a Modal
    # Volume, and keep a copy there with quail.server.checkpoint.
    db_path: Path | None = None
    # tests: run jobs on a thread with these hooks instead of a child
    hooks: Hooks | None = field(default=None, compare=False)
    hooks_reference: str | None = None
    in_process: bool = False


class _Changes:
    """Wake event-loop waiters after store writes, from any thread.

    ``current()`` returns the event a waiter should hold before it
    reads the store, so a write between the read and the wait still
    wakes it. Every write replaces the event with a fresh one.
    """

    def __init__(self):
        self._loop = None
        self._event = None

    def current(self) -> asyncio.Event:
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            # first use, or the app is now served by another event loop
            self._loop = loop
            self._event = asyncio.Event()
        return self._event

    def notify(self) -> None:
        loop = self._loop
        if loop is None:
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._advance)

    def _advance(self) -> None:
        event, self._event = self._event, asyncio.Event()
        event.set()

    @staticmethod
    async def wait(event: asyncio.Event, timeout: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(event.wait(), timeout)


class Server:
    """The store, the scheduler, and the accept path behind the routes."""

    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.data_dir = Path(settings.data_dir)
        self.inputs_dir = self.data_dir / "inputs"
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(settings.db_path or self.data_dir / "quail.sqlite3")
        self.recovered = self.store.recover()
        self.changes = _Changes()
        self.store.add_listener(self.changes.notify)
        executor = (InProcessExecutor(settings.hooks) if settings.in_process
                    else ChildProcessExecutor(settings.hooks_reference))
        self.scheduler = Scheduler(self.store, executor, self.data_dir)
        self.registry = built_in_registry()
        self._tokenizers = {}
        self._syncs = []
        self._closers = []

    def add_sync(self, sync) -> None:
        """Run ``sync()`` after a submission is saved, before it is acknowledged.

        A sync makes the saved record durable beyond the live database
        file, such as the checkpoint copy on a Modal Volume. When it
        raises, the submission is discarded and the client gets an
        error instead of an id the server might forget.
        """
        self._syncs.append(sync)

    def add_closer(self, close) -> None:
        """Run ``close()`` at shutdown, after the scheduler, before the store."""
        self._closers.append(close)

    def start(self) -> None:
        self.scheduler.start()

    def stop(self) -> None:
        self.scheduler.stop()
        for close in self._closers:
            close()
        self.store.close()

    def capabilities(self) -> dict:
        settings = self.settings
        return {
            "models": list(settings.models),
            "device": settings.device,
            "gpus": list(settings.gpus),
            "backends": list(settings.backends),
            "default_timeout_s": settings.default_timeout_s,
            "max_timeout_s": settings.max_timeout_s,
            "max_upload_bytes": settings.max_upload_bytes,
        }

    def check_config(self, config: dict) -> EngineConfig:
        try:
            engine = EngineConfig(**config)
        except TypeError as error:
            raise InvalidRequestError(f"bad engine config: {error}") from error
        settings = self.settings
        if engine.model not in settings.models:
            raise InvalidRequestError(
                f"this server does not run model {engine.model!r}; "
                f"it runs {list(settings.models)}")
        if engine.device != settings.device:
            raise InvalidRequestError(
                f"this server runs on device {settings.device!r}, "
                f"not {engine.device!r}")
        if engine.gpus not in settings.gpus:
            raise InvalidRequestError(
                f"this server runs with {list(settings.gpus)} GPUs, "
                f"not {engine.gpus}")
        if engine.backend not in settings.backends:
            raise InvalidRequestError(
                f"this server runs backends {list(settings.backends)}, "
                f"not {engine.backend!r}")
        return engine

    def snapshot_paths(self, inputs: dict) -> dict:
        paths = {}
        for name, spec in inputs.items():
            if not isinstance(spec, dict) or "id_col" not in spec:
                raise InvalidRequestError(f"input {name!r} needs an id_col")
            content_id = spec.get("content_id")
            if content_id is None:
                continue
            record = self.store.get_input(content_id)
            if record is None:
                raise InvalidRequestError(
                    f"input {name!r}: snapshot {content_id!r} was not uploaded")
            paths[content_id] = record.path
        return paths

    def _tokenizer(self, engine: EngineConfig):
        hooks = self.settings.hooks
        if hooks is not None and hooks.tokenizer is not None:
            return hooks.tokenizer
        if engine.model not in self._tokenizers:
            session = Session(engine, registry=self.registry)
            self._tokenizers[engine.model] = session.tokenizer
        return self._tokenizers[engine.model]

    def compile(self, spec: dict, engine: EngineConfig, inputs: dict,
                snapshot_paths: dict) -> None:
        """Compile the SQL against the inputs so bad queries fail now."""
        session = Session(engine, tokenizer=self._tokenizer(engine),
                          registry=self.registry)
        try:
            for name, input_spec in inputs.items():
                session.register(name, resolve(input_spec, snapshot_paths))
            session.sql(spec["sql"], order=spec.get("order"),
                        dialect=spec.get("dialect") or "snowflake")
        except (CompileError, ValueError, RefusalError) as error:
            raise InvalidRequestError(
                f"{type(error).__name__}: {error}") from error
        finally:
            session.close()

    def accept(self, body: dict) -> QueryStatus:
        """Validate a submission, save it as a queued record, make it durable."""
        if not isinstance(body, dict):
            raise InvalidRequestError("the submission must be a JSON object")
        sql = body.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise InvalidRequestError("the submission needs sql text")
        spec = {"sql": sql, "dialect": body.get("dialect") or "snowflake",
                "order": body.get("order")}
        inputs = body.get("inputs")
        if not isinstance(inputs, dict) or not inputs:
            raise InvalidRequestError("the submission needs at least one input")
        engine = self.check_config(body.get("config") or {})
        timeout_s = body.get("timeout_s")
        if timeout_s is None:
            timeout_s = self.settings.default_timeout_s
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise InvalidRequestError("timeout_s must be a positive number")
        if timeout_s > self.settings.max_timeout_s:
            raise InvalidRequestError(
                f"timeout_s {timeout_s:g} is over this server's maximum of "
                f"{self.settings.max_timeout_s:g} s")
        query_id = body.get("query_id")
        if query_id is not None:
            check_query_id(query_id)
        session_id = body.get("session_id")
        if session_id is not None and not isinstance(session_id, str):
            raise InvalidRequestError("session_id must be a string")
        paths = self.snapshot_paths(inputs)
        self.compile(spec, engine, inputs, paths)
        config = {"model": engine.model, "device": engine.device,
                  "gpus": engine.gpus, "backend": engine.backend}
        existed = query_id is not None and self.store.find(query_id) is not None
        status = self.store.create(spec=spec, config=config, inputs=inputs,
                                   timeout_s=float(timeout_s),
                                   query_id=query_id, session_id=session_id)
        if existed:
            return status
        try:
            for sync in self._syncs:
                sync()
        except Exception as error:
            self.store.discard(status.id)
            raise ServerError(
                f"the submission could not be saved durably: {error}"
            ) from error
        return status

    async def wait_status(self, query_id: str, after: int,
                          timeout: float) -> QueryStatus:
        """Return the snapshot once its revision passes ``after``.

        Returns the current snapshot when ``timeout`` seconds pass first.
        Waits on the event loop, not on a worker thread.
        """
        deadline = time.monotonic() + timeout
        while True:
            event = self.changes.current()
            status = await run_in_threadpool(self.store.get, query_id)
            remaining = deadline - time.monotonic()
            if status.revision > after or remaining <= 0:
                return status
            await self.changes.wait(event, remaining)

    def result_directory(self, status: QueryStatus) -> Path:
        if status.state != "succeeded" or status.result is None:
            raise NotReadyError(
                f"query {status.id} is {status.state}; no result is saved")
        return self.data_dir / status.result["directory"]

    def result_file(self, status: QueryStatus, name: str) -> Path:
        directory = self.result_directory(status)
        files = status.result["files"]
        allowed = {files["result"], files["report"]}
        allowed.update(entry["file"] for kind in ("filters", "joins")
                       for entry in files["answers"][kind])
        if name not in allowed:
            raise InvalidRequestError(f"{name!r} is not a file of this result")
        return directory / name

    def saved_answers(self, status: QueryStatus, after: int,
                      limit: int) -> dict:
        """Answer entries saved so far, from line ``after`` on.

        Available while the query runs and after it ends. ``next`` is
        the line to ask for next; ``done`` says whether more can come.
        """
        path = self.scheduler.result_directory(status.id) / ANSWERS_FILE
        items = read_answers(path, after, limit)
        return {"answers": items, "next": after + len(items),
                "done": status.done, "revision": status.revision}

    async def save_upload(self, request: Request, content_id: str) -> int:
        """Stream one snapshot upload to disk and check its hash."""
        if self.store.get_input(content_id) is not None:
            # drain the body so the connection can be reused
            async for _ in request.stream():
                pass
            return 200
        temporary = self.inputs_dir / f"upload-{uuid.uuid4().hex}.tmp"
        digest = hashlib.sha256()
        size = 0
        try:
            with open(temporary, "wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > self.settings.max_upload_bytes:
                        raise ServerError(
                            f"upload exceeds {self.settings.max_upload_bytes} "
                            "bytes")
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if digest.hexdigest() != content_id:
                raise InvalidRequestError(
                    "the uploaded bytes do not hash to the given content id")
            path = self.inputs_dir / f"{content_id}.arrow"
            with ipc.open_file(str(temporary)):
                pass
            os.replace(temporary, path)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        self.store.put_input(content_id, str(path), size)
        return 201


class _TokenMiddleware:
    """Require ``Authorization: Bearer <token>`` on every /v1 route."""

    def __init__(self, app, token: str):
        self.app = app
        self.expected = f"Bearer {token}"

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/v1/"):
            header = Headers(scope=scope).get("authorization", "")
            if header != self.expected:
                response = _error(ServerError("a bearer token is required"), 401)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _error(error: Exception, status: int | None = None) -> JSONResponse:
    code = status or getattr(error, "status", 500)
    return JSONResponse(
        {"error": {"type": type(error).__name__, "message": str(error)}},
        status_code=code)


def _status_response(status: QueryStatus, code: int = 200) -> JSONResponse:
    return JSONResponse(status.to_dict(), status_code=code)


def _sse_event(status: QueryStatus) -> str:
    return f"id: {status.revision}\ndata: {json.dumps(status.to_dict())}\n\n"


def create_app(settings: ServerSettings) -> Starlette:
    """Build the Starlette application; ``app.state.server`` owns the data."""
    server = Server(settings)

    async def capabilities(request):
        return JSONResponse(server.capabilities())

    async def input_head(request):
        content_id = request.path_params["content_id"]
        code = 200 if server.store.get_input(content_id) is not None else 404
        return Response(status_code=code)

    async def input_put(request):
        content_id = request.path_params["content_id"]
        if len(content_id) != 64 or any(c not in "0123456789abcdef"
                                        for c in content_id):
            return _error(InvalidRequestError(
                "content id must be a lowercase sha256 hex digest"))
        try:
            code = await server.save_upload(request, content_id)
        except ServerError as error:
            status = 413 if "exceeds" in str(error) else error.status
            return _error(error, status)
        except Exception as error:
            return _error(InvalidRequestError(
                f"upload is not an Arrow IPC file: {error}"))
        return JSONResponse({"content_id": content_id}, status_code=code)

    async def submit(request):
        try:
            body = await request.json()
        except ValueError:
            return _error(InvalidRequestError("the body must be JSON"))
        try:
            status = await run_in_threadpool(server.accept, body)
        except ServerError as error:
            return _error(error)
        return _status_response(status, 201)

    async def list_queries(request):
        requested_limit = request.query_params.get("limit", "50")
        limit = (None if requested_limit == "all"
                 else min(int(requested_limit), 500))
        session_id = request.query_params.get("session_id")
        items = [status.to_dict() for status in await run_in_threadpool(
            server.store.list_recent, limit, session_id)]
        return JSONResponse({"queries": items})

    async def get_status(request):
        query_id = request.path_params["query_id"]
        after = request.query_params.get("after")
        wait = min(float(request.query_params.get("wait", "0")), MAX_WAIT_S)
        try:
            if after is not None and wait > 0:
                status = await server.wait_status(query_id, int(after), wait)
            else:
                status = await run_in_threadpool(server.store.get, query_id)
        except ServerError as error:
            return _error(error)
        return _status_response(status)

    async def events(request):
        query_id = request.path_params["query_id"]
        try:
            status = server.store.get(query_id)
        except ServerError as error:
            return _error(error)
        after = request.headers.get("last-event-id")
        after = int(after) if after and after.isdigit() else 0

        async def stream():
            current = status
            if current.revision > after:
                yield _sse_event(current)
            while not current.done:
                if await request.is_disconnected():
                    return
                try:
                    newer = await server.wait_status(
                        query_id, current.revision, SSE_KEEPALIVE_S)
                except Exception:
                    # the server is shutting down; the client reconnects
                    return
                if newer.revision > current.revision:
                    current = newer
                    yield _sse_event(current)
                else:
                    yield ": keep-alive\n\n"

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"})

    async def cancel(request):
        query_id = request.path_params["query_id"]
        try:
            status = server.store.request_cancel(query_id)
        except ServerError as error:
            return _error(error)
        return _status_response(status)

    async def result_stream(request):
        """The result rows as an Arrow IPC stream, one batch at a time."""
        query_id = request.path_params["query_id"]
        try:
            status = server.store.get(query_id)
            path = server.result_file(status, RESULT_FILE)
        except ServerError as error:
            return _error(error)
        return StreamingResponse(
            iter_ipc_stream(path), media_type=ARROW_STREAM,
            headers={"x-quail-rows": str(status.result["rows"])})

    async def result_file(request):
        query_id = request.path_params["query_id"]
        name = request.path_params["name"]
        try:
            status = server.store.get(query_id)
            path = server.result_file(status, name)
        except ServerError as error:
            return _error(error)
        media = "application/json" if name.endswith(".json") else ARROW_FILE
        return FileResponse(str(path), media_type=media)

    async def saved_answers(request):
        query_id = request.path_params["query_id"]
        after = max(int(request.query_params.get("after", "0")), 0)
        limit = min(int(request.query_params.get("limit", "1000")), 10_000)
        try:
            status = server.store.get(query_id)
        except ServerError as error:
            return _error(error)
        return JSONResponse(await run_in_threadpool(
            server.saved_answers, status, after, limit))

    async def rows(request):
        query_id = request.path_params["query_id"]
        limit = min(int(request.query_params.get("limit", "20")), 1000)
        try:
            status = server.store.get(query_id)
            path = server.result_file(status, RESULT_FILE)
        except ServerError as error:
            return _error(error)

        def read():
            with ipc.open_file(str(path)) as reader:
                table = reader.read_all().slice(0, limit)
            return {"columns": table.column_names,
                    "rows": [list(row.values()) for row in table.to_pylist()],
                    "total_rows": status.result["rows"]}

        return JSONResponse(await run_in_threadpool(read))

    async def page(request):
        return HTMLResponse((STATIC_DIR / "status.html").read_text("utf-8"))

    routes = [
        Route("/v1/capabilities", capabilities),
        Route("/v1/inputs/{content_id}", input_head, methods=["HEAD"]),
        Route("/v1/inputs/{content_id}", input_put, methods=["PUT"]),
        Route("/v1/queries", submit, methods=["POST"]),
        Route("/v1/queries", list_queries, methods=["GET"]),
        Route("/v1/queries/{query_id}", get_status),
        Route("/v1/queries/{query_id}/events", events),
        Route("/v1/queries/{query_id}/cancel", cancel, methods=["POST"]),
        Route("/v1/queries/{query_id}/result", result_stream),
        Route("/v1/queries/{query_id}/files/{name:path}", result_file),
        Route("/v1/queries/{query_id}/rows", rows),
        Route("/v1/queries/{query_id}/answers", saved_answers),
        Route("/", page),
        Route("/queries", page),
        Route("/queries/{query_id}", page),
    ]
    middleware = []
    if settings.token:
        middleware.append(Middleware(_TokenMiddleware, token=settings.token))

    @contextlib.asynccontextmanager
    async def lifespan(app):
        server.start()
        try:
            yield
        finally:
            server.stop()

    app = Starlette(routes=routes, middleware=middleware, lifespan=lifespan,
                    max_body_size=settings.max_upload_bytes)
    app.state.server = server
    return app
