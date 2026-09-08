"""Progress messages for the long steps of a query, on the "quail" logger."""

import contextlib
import logging
import sys
import time


class _StdoutHandler(logging.StreamHandler):
    """Write to the sys.stdout of the moment, so output capture sees it."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


logger = logging.getLogger("quail")
if not logger.handlers:
    # INFO lines reach stdout unless the program configures the logger itself
    _handler = _StdoutHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s quail %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

_QUIET = 0


def say(message: str) -> None:
    """Log one progress message at INFO, unless inside quiet()."""
    if not _QUIET:
        logger.info(message)


@contextlib.contextmanager
def quiet():
    """Suppress progress messages, for warmup passes that reuse the loops."""
    global _QUIET
    _QUIET += 1
    try:
        yield
    finally:
        _QUIET -= 1


class Progress:
    """Report a running count, logging at most once every `every` seconds."""

    def __init__(self, label: str, total: int | None = None,
                 unit: str = "documents", every: float = 5.0):
        self.label = label
        self.total = total
        self.unit = unit
        self.every = every
        self.done = 0
        self.started = time.perf_counter()
        self._last = self.started

    def update(self, done: int) -> None:
        """Record the count and log a line when enough time has passed."""
        self.done = done
        now = time.perf_counter()
        if now - self._last >= self.every:
            self._last = now
            say(self._line(self.label, now))

    def finish(self, label: str, extra: str = "") -> None:
        """Log the final line under a past tense label."""
        line = self._line(label, time.perf_counter())
        say(f"{line}, {extra}" if extra else line)

    def _line(self, label: str, now: float) -> str:
        elapsed = now - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        count = f"{self.done:,}"
        if self.total is not None:
            count += f"/{self.total:,}"
        return (f"{label}: {count} {self.unit}, {elapsed:.1f} s, "
                f"{rate:,.0f} {self.unit}/s")
