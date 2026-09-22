"""Keep a durable copy of the live SQLite file on another disk.

On Modal the server's data directory is a Volume: a network file
system with no file locking, where a file should have one writer and
changes reach the Volume only on ``commit()``. SQLite's live file (and
its WAL) therefore stays on the container's local disk, and this module
copies it to the Volume after changes, then commits. Results and
uploads are write-once files and go to the Volume directly.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from quail.server.store import Store

logger = logging.getLogger("quail.server")


def restore(source: Path, destination: Path) -> bool:
    """Copy a saved database into place when no live database exists yet.

    Returns True when a copy was made.
    """
    if destination.exists() or not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return True


class Checkpoint:
    """Copy the store to ``target`` after changes, then call ``commit``.

    A state change (a record created, started, finished, cancelled, or
    given its plan) is attempted within ``min_interval_s``. A write that
    only moved a progress counter is copied at most every
    ``progress_interval_s``; losing it costs nothing, because a restart
    marks an unfinished record interrupted anyway. Every copy is a full
    copy of the database file plus one ``commit`` (``volume.commit`` on
    Modal), so this keeps a long-running query from copying the file
    once a second. Failures are logged and retried; a restart can only
    restore the latest copy that completed successfully.
    """

    def __init__(self, store: Store, target: Path,
                 commit: Callable[[], None] | None = None,
                 min_interval_s: float = 1.0,
                 progress_interval_s: float = 30.0):
        self.store = store
        self.target = Path(target)
        self.commit = commit
        self.min_interval_s = min_interval_s
        self.progress_interval_s = progress_interval_s
        self._seen = -1
        self._seen_durable = -1
        self._last_copy = 0.0
        self._stop = threading.Event()
        self._thread = None
        # one copy at a time: the thread and sync() callers share it
        self._lock = threading.Lock()
        self.copies = 0

    def run_once(self, force: bool = False) -> bool:
        """Copy and commit when a copy is due. Returns True if it did.

        ``force`` copies any new write now, ignoring the progress timer.
        """
        with self._lock:
            count = self.store.write_count
            durable = self.store.durable_write_count
            if count == self._seen or self.store.closed:
                return False
            progress_only = durable == self._seen_durable
            if (progress_only and not force and time.monotonic() - self._last_copy
                    < self.progress_interval_s):
                return False
            self.store.backup(self.target)
            if self.commit is not None:
                self.commit()
            self._seen, self._seen_durable = count, durable
            self._last_copy = time.monotonic()
            self.copies += 1
            return True

    def sync(self) -> None:
        """Copy and commit every write so far, before the caller goes on.

        The server calls this before it acknowledges a submission, so a
        record the client has an id for is on the durable copy. Raises
        when the copy or the commit fails.
        """
        self.run_once(force=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("checkpoint of %s failed", self.target)
            self._stop.wait(self.min_interval_s)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._loop, name="quail-checkpoint", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the thread and make one final copy. Safe to call twice."""
        if self._stop.is_set():
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(30)
            self._thread = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                self.run_once(force=True)
                return
            except Exception:
                logger.exception("final checkpoint of %s failed", self.target)
                time.sleep(1)
