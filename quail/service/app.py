"""The HTTP application of the query service.

Needs the ``service`` installation extra (Starlette and uvicorn). The
Python client in ``quail.service.client`` and a browser both talk to
these routes. Status reads return the complete saved snapshot; a
client can wait for a newer revision with ``?after=<revision>&wait=<s>``
or subscribe with server-sent events at ``/events``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
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
from quail.service.executor import (
    ChildProcessExecutor,
    Hooks,
    InProcessExecutor,
)
from quail.service.inputs import resolve
from quail.service.records import (
    InvalidRequestError,
    NotReadyError,
    QueryStatus,
    ServiceError,
)
from quail.service.scheduler import DEFAULT_TIMEOUT_S, Scheduler
from quail.service.store import Store

MAX_WAIT_S = 60.0
STATIC_DIR = Path(__file__).parent / "static"


@dataclass(frozen=True)
class ServiceSettings:
    """What one deployed service can run and where it keeps its data."""

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
    # Volume, and keep a copy there with quail.service.checkpoint.
    db_path: Path | None = None
    # tests: run jobs on a thread with these hooks instead of a child
    hooks: Hooks | None = field(default=None, compare=False)
    hooks_reference: str | None = None
    in_process: bool = False


class Service:
    """The store, the scheduler, and the accept path behind the routes."""

    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.data_dir = Path(settings.data_dir)
        self.inputs_dir = self.data_dir / "inputs"
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(settings.db_path or self.data_dir / "quail.sqlite3")
        self.recovered = self.store.recover()
        executor = (InProcessExecutor(settings.hooks) if settings.in_process
                    else ChildProcessExecutor(settings.hooks_reference))
        self.scheduler = Scheduler(self.store, executor, self.data_dir)
        self.registry = built_in_registry()
        self._tokenizers = {}
        self._closers = []

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
                f"this service does not run model {engine.model!r}; "
                f"it runs {list(settings.models)}")
        if engine.device != settings.device:
            raise InvalidRequestError(
                f"this service runs on device {settings.device!r}, "
                f"not {engine.device!r}")
        if engine.gpus not in settings.gpus:
            raise InvalidRequestError(
                f"this service runs with {list(settings.gpus)} GPUs, "
                f"not {engine.gpus}")
        if engine.backend not in settings.backends:
            raise InvalidRequestError(
                f"this service runs backends {list(settings.backends)}, "
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
        """Validate a submission and save it as a queued record."""
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
                f"timeout_s {timeout_s:g} is over this service's maximum of "
                f"{self.settings.max_timeout_s:g} s")
        request_key = body.get("request_key")
        if request_key is not None and not isinstance(request_key, str):
            raise InvalidRequestError("request_key must be a string")
        paths = self.snapshot_paths(inputs)
        self.compile(spec, engine, inputs, paths)
        config = {"model": engine.model, "device": engine.device,
                  "gpus": engine.gpus, "backend": engine.backend}
        return self.store.create(spec=spec, config=config, inputs=inputs,
                                 timeout_s=float(timeout_s),
                                 request_key=request_key)

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
                        raise ServiceError(
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
                response = _error(ServiceError("a bearer token is required"), 401)
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


def create_app(settings: ServiceSettings) -> Starlette:
    """Build the Starlette application; ``app.state.service`` owns the data."""
    service = Service(settings)

    async def capabilities(request):
        return JSONResponse(service.capabilities())

    async def input_head(request):
        content_id = request.path_params["content_id"]
        code = 200 if service.store.get_input(content_id) is not None else 404
        return Response(status_code=code)

    async def input_put(request):
        content_id = request.path_params["content_id"]
        if len(content_id) != 64 or any(c not in "0123456789abcdef"
                                        for c in content_id):
            return _error(InvalidRequestError(
                "content id must be a lowercase sha256 hex digest"))
        try:
            code = await service.save_upload(request, content_id)
        except ServiceError as error:
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
            status = await run_in_threadpool(service.accept, body)
        except ServiceError as error:
            return _error(error)
        return _status_response(status, 201)

    async def list_queries(request):
        limit = min(int(request.query_params.get("limit", "50")), 500)
        items = [status.to_dict() for status in service.store.list_recent(limit)]
        return JSONResponse({"queries": items})

    async def get_status(request):
        query_id = request.path_params["query_id"]
        after = request.query_params.get("after")
        wait = min(float(request.query_params.get("wait", "0")), MAX_WAIT_S)
        try:
            if after is not None and wait > 0:
                status = await run_in_threadpool(
                    service.store.wait, query_id, int(after), wait)
            else:
                status = service.store.get(query_id)
        except ServiceError as error:
            return _error(error)
        return _status_response(status)

    async def events(request):
        query_id = request.path_params["query_id"]
        try:
            status = service.store.get(query_id)
        except ServiceError as error:
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
                    newer = await run_in_threadpool(
                        service.store.wait, query_id, current.revision, 15.0)
                except Exception:
                    # the service is shutting down; the client reconnects
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
            status = service.store.request_cancel(query_id)
        except ServiceError as error:
            return _error(error)
        return _status_response(status)

    async def result_file(request):
        query_id = request.path_params["query_id"]
        name = request.path_params.get("name", "result.arrow")
        try:
            status = service.store.get(query_id)
            path = service.result_file(status, name)
        except ServiceError as error:
            return _error(error)
        media = ("application/json" if name.endswith(".json")
                 else "application/vnd.apache.arrow.file")
        return FileResponse(str(path), media_type=media)

    async def rows(request):
        query_id = request.path_params["query_id"]
        limit = min(int(request.query_params.get("limit", "20")), 1000)
        try:
            status = service.store.get(query_id)
            path = service.result_file(status, status.result["files"]["result"])
        except ServiceError as error:
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
        Route("/v1/queries/{query_id}/result", result_file),
        Route("/v1/queries/{query_id}/files/{name:path}", result_file),
        Route("/v1/queries/{query_id}/rows", rows),
        Route("/", page),
        Route("/queries/{query_id}", page),
    ]
    middleware = []
    if settings.token:
        middleware.append(Middleware(_TokenMiddleware, token=settings.token))

    @contextlib.asynccontextmanager
    async def lifespan(app):
        service.start()
        try:
            yield
        finally:
            service.stop()

    app = Starlette(routes=routes, middleware=middleware, lifespan=lifespan,
                    max_body_size=settings.max_upload_bytes)
    app.state.service = service
    return app
