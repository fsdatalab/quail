"""Run queued records one at a time and save what the executor reports.

The scheduler thread is the only path from executor events to the
store. It also enforces cancellation and the per-query timeout: it
closes the record first, then stops the execution, so any event the
execution still sends is rejected as stale.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from quail.server.artifacts import ANSWERS_FILE, Answers, verify_result
from quail.server.executor import Executor, Job
from quail.server.records import InvalidRequestError, QueryStatus
from quail.server.store import Store

logger = logging.getLogger("quail.server")

DEFAULT_TIMEOUT_S = 1000.0
# progress writes are one fsync each; a loop can report far faster
PROGRESS_WRITE_INTERVAL_S = 0.5
# how long a finished execution may take to report done before it is killed
FINISH_GRACE_S = 30.0


class Scheduler:
    """Pick the oldest queued record and run it to a terminal state."""

    def __init__(self, store: Store, executor: Executor, data_dir: str | Path,
                 *, poll_s: float = 0.25):
        self.store = store
        self.executor = executor
        self.data_dir = Path(data_dir)
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="quail-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(30)
            self._thread = None
        self.executor.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            status = self.store.next_queued()
            if status is None:
                self.store.wait_for_change(self.poll_s)
                continue
            try:
                self.run_one(status)
            except Exception:
                logger.exception("query %s: scheduler error", status.id)

    def result_directory(self, query_id: str) -> Path:
        return self.data_dir / "results" / query_id

    def _job(self, status: QueryStatus) -> Job:
        snapshot_paths = {}
        for spec in status.inputs.values():
            content_id = spec.get("content_id")
            if content_id is None:
                continue
            record = self.store.get_input(content_id)
            if record is None:
                raise InvalidRequestError(
                    f"input snapshot {content_id!r} was not uploaded")
            snapshot_paths[content_id] = record.path
        directory = self.result_directory(status.id)
        directory.mkdir(parents=True, exist_ok=True)
        return Job(status.id, status.spec, status.config, status.inputs,
                   snapshot_paths, str(directory))

    def run_one(self, status: QueryStatus) -> QueryStatus:
        """Execute one queued record and return its final snapshot."""
        query_id = status.id
        try:
            epoch = self.store.begin(query_id)
        except InvalidRequestError:
            # cancelled between next_queued() and begin()
            return self.store.get(query_id)
        try:
            job = self._job(status)
        except Exception as error:
            self.store.finish(query_id, epoch, "failed", error={
                "type": type(error).__name__, "message": str(error)})
            return self.store.get(query_id)

        last_progress = [0.0]
        counts = [None]        # the loop's last progress payload
        written = [0]          # answers_saved on the record
        answers = Answers(Path(job.artifact_dir) / ANSWERS_FILE)

        def write_progress(payload):
            written[0] = answers.count
            self._apply(query_id, epoch, job.artifact_dir, "progress",
                        {**payload, "answers_saved": answers.count})

        def emit(kind, payload):
            if kind == "answers":
                # saved at once; the record learns the count on the
                # progress timer, and a reader fetches what it has not seen
                answers.append(payload)
                complete = False
                payload = counts[0] or {}
            elif kind == "progress":
                counts[0] = payload
                complete = payload.get("total") and (
                    payload.get("done") == payload.get("total"))
            if kind in ("answers", "progress"):
                now = time.monotonic()
                if not complete and now - last_progress[0] < \
                        PROGRESS_WRITE_INTERVAL_S:
                    return
                last_progress[0] = now
                write_progress(payload)
                return
            if kind in ("finished", "failed") and written[0] != answers.count:
                # the record's count must be right before it closes
                write_progress(counts[0] or {})
            self._apply(query_id, epoch, job.artifact_dir, kind, payload)

        execution = self.executor.start(job, emit)
        deadline = time.monotonic() + status.timeout_s
        while not execution.wait(self.poll_s):
            current = self.store.get(query_id)
            if current.done:
                # Our own finished or failed event closed the record and
                # the executor is about to report done. Killing it here
                # would throw away the loaded model for the next query.
                if not execution.wait(FINISH_GRACE_S):
                    execution.stop()
                break
            if current.cancel_requested:
                self.store.finish(query_id, epoch, "cancelled", error={
                    "type": "Cancelled",
                    "message": "cancelled while executing"})
                execution.stop()
                break
            if time.monotonic() > deadline:
                self.store.finish(query_id, epoch, "failed", error={
                    "type": "TimeoutError",
                    "message": f"execution exceeded the query's "
                               f"{status.timeout_s:g} s timeout"})
                execution.stop()
                break
        answers.close()
        final = self.store.get(query_id)
        if not final.done:
            self.store.finish(query_id, epoch, "failed", error={
                "type": "ExecutorExit",
                "message": "the execution ended without a result"})
            final = self.store.get(query_id)
        return final

    def _apply(self, query_id: str, epoch: int, artifact_dir: str,
               kind: str, payload: dict) -> None:
        if kind == "state":
            self.store.update(
                query_id, epoch, state=payload["state"],
                phase=payload.get("phase"))
        elif kind == "phase":
            self.store.update(query_id, epoch, phase=payload)
        elif kind == "plan":
            self.store.update(query_id, epoch, plan=payload)
        elif kind == "progress":
            self.store.update(query_id, epoch, progress=payload)
        elif kind == "failed":
            self.store.finish(query_id, epoch, "failed", error=payload)
        elif kind == "finished":
            try:
                verify_result(artifact_dir, payload)
            except Exception as error:
                self.store.finish(query_id, epoch, "failed", error={
                    "type": "PublishError",
                    "message": f"the saved result could not be read back: "
                               f"{error}"})
                return
            relative = Path(artifact_dir).relative_to(self.data_dir)
            self.store.finish(query_id, epoch, "succeeded",
                              result={**payload, "directory": str(relative)})
        else:
            logger.warning("query %s: unknown executor event %r", query_id, kind)
