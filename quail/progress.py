"""Progress messages for the long steps of a query, on the "quail" logger."""

import contextlib
import logging
import sys
import time

_GPU_INDEX = None


def set_gpu_index(index: int | None) -> None:
    """Label this process's Quail messages with its GPU index."""
    global _GPU_INDEX
    _GPU_INDEX = index


class _GpuFilter(logging.Filter):
    def filter(self, record):
        record.gpu = "" if _GPU_INDEX is None else f"[GPU {_GPU_INDEX}]"
        return True


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
        "%(levelname)s %(asctime)s [quail]%(gpu)s %(message)s",
        datefmt="%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.addFilter(_GpuFilter())
    logger.setLevel(logging.INFO)
    logger.propagate = False

_QUIET = 0
_SINK = None


def set_progress_sink(sink) -> None:
    """Send every throttled progress count to ``sink`` as well as the log.

    The sink is called as ``sink(label, done, total, unit)`` from the
    thread that runs the loop, so it must return quickly and must not
    block. Pass None to remove it.
    """
    global _SINK
    _SINK = sink


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
                 unit: str = "documents", every: float = 5.0, *, emit=say):
        self.label = label
        self.total = total
        self.unit = unit
        self.every = every
        self.emit = emit
        self.done = 0
        self.started = time.perf_counter()
        self._last = self.started

    def update(self, done: int) -> None:
        """Record the count and log a line when enough time has passed."""
        self.done = done
        now = time.perf_counter()
        if now - self._last >= self.every:
            self._last = now
            self.emit(self._line(self.label, now))
            self._report()

    def finish(self, label: str, extra: str = "") -> None:
        """Log the final line under a past tense label."""
        line = self._line(label, time.perf_counter())
        self.emit(f"{line}, {extra}" if extra else line)
        self._report()

    def _report(self) -> None:
        if _SINK is not None:
            _SINK(self.label, self.done, self.total, self.unit)

    def _line(self, label: str, now: float) -> str:
        elapsed = now - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        count = f"{self.done:,}"
        if self.total is not None:
            count += f"/{self.total:,}"
        return (f"{label}: {count} {self.unit}, {elapsed:.1f} s, "
                f"{rate:,.0f} {self.unit}/s")
