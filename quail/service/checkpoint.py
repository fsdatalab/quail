"""Keep a durable copy of the live SQLite file on another disk.

On Modal the service's data directory is a Volume: a network file
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

from quail.service.store import Store

logger = logging.getLogger("quail.service")


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
    """Copy the store to ``target`` after each change, then call ``commit``.

    Copies are made at most once per ``min_interval_s`` and only when the
    store has new committed writes. ``commit`` is what makes the copy and
    every other file written since durable (``volume.commit`` on Modal).
    The window of loss on a sudden container death is one interval.
    """

    def __init__(self, store: Store, target: Path,
                 commit: Callable[[], None] | None = None,
                 min_interval_s: float = 1.0):
        self.store = store
        self.target = Path(target)
        self.commit = commit
        self.min_interval_s = min_interval_s
        self._seen = -1
        self._stop = threading.Event()
        self._thread = None
        self.copies = 0

    def run_once(self) -> bool:
        """Copy and commit when there are new writes. Returns True if it did."""
        count = self.store.write_count
        if count == self._seen or self.store.closed:
            return False
        self.store.backup(self.target)
        if self.commit is not None:
            self.commit()
        self._seen = count
        self.copies += 1
        return True

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
                self.run_once()
                return
            except Exception:
                logger.exception("final checkpoint of %s failed", self.target)
                time.sleep(1)
