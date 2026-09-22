"""Run one saved query record and report events about it.

``run_job`` does the work in the calling process with a local Session.
``ChildProcessExecutor`` runs it in one long-lived child process, so a
GPU fault cannot take the server down and a stop request can kill the
child. ``InProcessExecutor`` runs it on a thread and exists for tests.
The executor never touches the database; it only emits events.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

from quail.server.inputs import resolve

Emit = Callable[[str, dict], None]


@dataclass(frozen=True)
class Job:
    """Everything the executor needs to run one record."""

    id: str
    spec: dict
    config: dict
    inputs: dict
    snapshot_paths: dict
    artifact_dir: str
    model_phase: dict | None = None


@dataclass(frozen=True)
class Hooks:
    """Test seams: a fake model executor and a fake tokenizer."""

    physical_executor: Callable | None = None
    tokenizer: Callable | None = None


def load_hooks(reference: str | None) -> Hooks | None:
    """Import ``module:attribute`` naming a Hooks object, or return None."""
    if reference is None:
        return None
    module_name, _, attribute = reference.partition(":")
    hooks = getattr(importlib.import_module(module_name), attribute)
    if not isinstance(hooks, Hooks):
        raise TypeError(f"{reference} is not a quail.server.executor.Hooks")
    return hooks


def error_event(error: BaseException) -> dict:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


def phase_event(name: str, message: str, **details) -> dict:
    """Build one timestamped phase payload."""
    return {
        "name": name,
        "message": message,
        "recorded_at": time.time(),
        **details,
    }


def run_job(job: Job, emit: Emit, hooks: Hooks | None = None) -> None:
    """Execute one job in this process and emit its events.

    Emits ``plan`` once the planner has chosen, ``state`` when execution
    starts, throttled ``progress`` counts, then ``finished`` with the
    result manifest or ``failed`` with the error. Never raises.
    """
    from quail.execution.execute import execute_query
    from quail.execution.session import RefusalError, Session
    from quail.planner.plan import EngineConfig, Refusal
    from quail.progress import set_answer_sink, set_progress_sink
    from quail.server.artifacts import write_result

    hooks = hooks or Hooks()

    def progress(label, done, total, unit):
        emit("progress", {"label": label, "done": done, "total": total,
                          "unit": unit, "recorded_at": time.time()})

    def answers(payload):
        emit("answers", payload)

    try:
        config = EngineConfig(**job.config)
        with Session(config, tokenizer=hooks.tokenizer) as session:
            for name, spec in job.inputs.items():
                session.register(name, resolve(spec, job.snapshot_paths))
            emit("phase", phase_event("planning", "Planning the query"))
            query = session.sql(
                job.spec["sql"], order=job.spec.get("order"),
                dialect=job.spec.get("dialect") or "snowflake")
            plan = query.plan()
            if isinstance(plan, Refusal):
                raise RefusalError(plan)
            emit("plan", {
                "text": query.explain(),
                "estimated_seconds": plan.estimated_seconds,
                "backend": plan.backend,
                "workers": plan.workers,
                "envelope": plan.to_envelope(session.registry.codecs),
            })
            if job.model_phase is not None:
                emit("phase", {
                    **job.model_phase,
                    "recorded_at": time.time(),
                })
            emit("state", {
                "state": "running",
                "phase": phase_event("executing", "Executing the query"),
            })
            set_progress_sink(progress)
            set_answer_sink(answers)
            try:
                result = execute_query(
                    query, physical_executor=hooks.physical_executor)
                manifest = write_result(result, job.artifact_dir)
            finally:
                set_progress_sink(None, owner=progress)
                set_answer_sink(None, owner=answers)
        emit("finished", manifest)
    except Exception as error:
        emit("failed", error_event(error))


class Execution(Protocol):
    """One running job."""

    def wait(self, timeout: float) -> bool:
        """Return True once the job has emitted finished or failed."""
        ...

    def stop(self) -> None:
        """Stop the job as soon as possible; the job emits nothing after."""
        ...


class Executor(Protocol):
    def start(self, job: Job, emit: Emit) -> Execution: ...

    def close(self) -> None: ...


class _ThreadExecution:
    def __init__(self, job: Job, emit: Emit, hooks: Hooks | None):
        self._done = threading.Event()
        self._stopped = False

        def guarded(kind, payload):
            if not self._stopped:
                emit(kind, payload)

        def target():
            try:
                run_job(job, guarded, hooks)
            finally:
                self._done.set()

        self._thread = threading.Thread(
            target=target, name=f"quail-job-{job.id[:8]}", daemon=True)
        self._thread.start()

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout)

    def stop(self) -> None:
        # A thread cannot be killed; later events are dropped instead.
        self._stopped = True


class InProcessExecutor:
    """Run jobs on a thread of the server process (tests and CPU-only)."""

    def __init__(self, hooks: Hooks | None = None):
        self.hooks = hooks
        self._loaded_model = None

    def start(self, job: Job, emit: Emit) -> Execution:
        model = job.config.get("model")
        if self._loaded_model is None:
            model_phase = {
                "name": "loading_model",
                "message": f"Loading and warming model {model}",
                "model": model,
            }
        elif self._loaded_model != model:
            model_phase = {
                "name": "switching_model",
                "message": (
                    f"Switching from {self._loaded_model} and loading {model}"
                ),
                "previous_model": self._loaded_model,
                "model": model,
            }
        else:
            model_phase = None
        self._loaded_model = model
        job = replace(job, model_phase=model_phase)
        return _ThreadExecution(job, emit, self.hooks)

    def close(self) -> None:
        self._loaded_model = None


def _child_main(conn, hooks_reference: str | None) -> None:
    hooks = load_hooks(hooks_reference)
    # The loop thread only appends to this queue. A pipe write blocks
    # once its buffer is full, so sending from the loop would stall the
    # GPU whenever the parent falls behind on saving events.
    events: queue.SimpleQueue = queue.SimpleQueue()

    def send_events():
        while True:
            item = events.get()
            if item is None:
                return
            try:
                conn.send(item)
            except (OSError, ValueError):
                return      # the parent is gone; nothing left to tell

    sender = threading.Thread(target=send_events, name="quail-events",
                              daemon=True)
    sender.start()
    try:
        while True:
            try:
                job = conn.recv()
            except (EOFError, KeyboardInterrupt):
                # the parent closed the pipe or the container is stopping
                return
            if job is None:
                return
            run_job(job, lambda kind, payload: events.put((kind, payload)),
                    hooks)
            events.put(("done", {}))
    finally:
        events.put(None)
        sender.join(30)


@dataclass
class _ChildExecution:
    executor: ChildProcessExecutor
    emit: Emit
    job: Job
    _done: threading.Event = field(default_factory=threading.Event)
    _stopped: bool = False

    def run(self) -> None:
        conn = self.executor._conn
        try:
            while True:
                try:
                    kind, payload = conn.recv()
                except (EOFError, OSError):
                    if not self._stopped:
                        self.emit("failed", {
                            "type": "ExecutorExit",
                            "message": "the executor process exited during "
                                       f"query {self.job.id}",
                            "traceback": "",
                        })
                    with self.executor._lock:
                        self.executor._discard_child()
                    return
                if kind == "done":
                    return
                if not self._stopped:
                    self.emit(kind, payload)
        finally:
            self._done.set()

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout)

    def stop(self) -> None:
        self._stopped = True
        self.executor._kill_child()


class ChildProcessExecutor:
    """Run jobs in one long-lived child process that keeps the model warm.

    ``stop`` kills the child; the next job starts a fresh one, which
    pays the model boot again. That is the reliable stop the current
    GPU loop offers. A job that names a different model than the child
    has loaded also gets a fresh child: one model's weights and KV take
    the whole GPU, and a new process is the sure way to give them back.
    """

    def __init__(self, hooks_reference: str | None = None):
        self.hooks_reference = hooks_reference
        self._process = None
        self._conn = None
        self._loaded_model = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _ensure_child(self) -> None:
        import multiprocessing as mp

        if self._process is not None and self._process.is_alive():
            return
        self._discard_child()
        context = mp.get_context("spawn")
        parent_conn, child_conn = context.Pipe()
        # not a daemon: a multi-GPU run starts its own worker processes,
        # which a daemonic process may not do. The child exits on its own
        # when the pipe closes, and close() kills it at interpreter exit.
        process = context.Process(
            target=_child_main, args=(child_conn, self.hooks_reference),
            name="quail-server-executor", daemon=False)
        process.start()
        child_conn.close()
        self._process, self._conn = process, parent_conn

    def _discard_child(self) -> None:
        conn, self._conn = self._conn, None
        self._process = None
        if conn is not None:
            with contextlib.suppress(OSError):
                conn.close()

    def _kill_child(self) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.is_alive():
                process.kill()
                process.join(30)

    def start(self, job: Job, emit: Emit) -> Execution:
        with self._lock:
            model = job.config.get("model")
            alive = self._process is not None and self._process.is_alive()
            loaded_model = self._loaded_model if alive else None
            if loaded_model not in (None, model):
                model_phase = {
                    "name": "switching_model",
                    "message": (
                        f"Switching from {loaded_model} and loading {model}"
                    ),
                    "previous_model": loaded_model,
                    "model": model,
                }
                self._close_child_unlocked()
            elif loaded_model is None:
                model_phase = {
                    "name": "loading_model",
                    "message": f"Loading and warming model {model}",
                    "model": model,
                }
            else:
                model_phase = None
            self._ensure_child()
            self._loaded_model = model
            job = replace(job, model_phase=model_phase)
            self._conn.send(job)
            execution = _ChildExecution(self, emit, job)
        threading.Thread(target=execution.run, daemon=True,
                         name=f"quail-job-{job.id[:8]}").start()
        return execution

    def close(self) -> None:
        with self._lock:
            self._close_child_unlocked()

    def _close_child_unlocked(self) -> None:
        """Ask the child to exit, then make sure it did."""
        if self._conn is not None and self._process is not None \
                and self._process.is_alive():
            try:
                self._conn.send(None)
            except (OSError, ValueError):
                pass
            self._process.join(10)
        self._kill_child_unlocked()

    def _kill_child_unlocked(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.kill()
            self._process.join(30)
        self._discard_child()
        self._loaded_model = None
