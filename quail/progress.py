"""Progress messages for the long steps of a query, on the "quail" logger.

On a terminal a running count is one bar that updates in place, and the
messages are colored. Anywhere else, such as a log file or a Modal log,
the same messages are plain lines and a count logs a line at each tenth
of its total.
"""

import contextlib
import logging
import os
import sys
import time

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.progress import Progress as Bar

_GPU_INDEX = None
# names the run in every line's prefix, such as a benchmark's query
# family; read from the environment so worker processes inherit it
LOG_TAG_ENV = "QUAIL_LOG_TAG"


def set_gpu_index(index: int | None) -> None:
    """Label this process's Quail messages with its GPU index."""
    global _GPU_INDEX
    _GPU_INDEX = index


class _GpuFilter(logging.Filter):
    def filter(self, record):
        tag = os.environ.get(LOG_TAG_ENV)
        record.gpu = (("" if not tag else f"[{tag}]")
                      + ("" if _GPU_INDEX is None else f"[GPU {_GPU_INDEX}]"))
        return True


_CONSOLES: dict = {}


def console() -> Console:
    """Return the console of the sys.stdout of the moment, so output capture sees it."""
    found = _CONSOLES.get(id(sys.stdout))
    if found is None or found.file is not sys.stdout:
        _CONSOLES.clear()
        found = _CONSOLES[id(sys.stdout)] = Console(file=sys.stdout, highlight=False)
    return found


class _ConsoleHandler(logging.Handler):
    """Write each message to `console()`: styled on a terminal, plain elsewhere."""

    def emit(self, record):
        found = console()
        stamp = time.strftime("%m-%d %H:%M:%S", time.localtime(record.created))
        # a record of a child logger skips this logger's filter; format()
        # adds a logged exception's traceback
        text = f"[quail]{getattr(record, 'gpu', '')} {self.format(record)}"
        if found.is_terminal:
            found.print(f"[dim]{stamp}[/dim]", escape(text), soft_wrap=True)
        else:
            # a log file keeps one unwrapped line per message
            print(f"{record.levelname} {stamp} {text}", file=found.file, flush=True)


logger = logging.getLogger("quail")
if not logger.handlers:
    # INFO lines reach stdout unless the program configures the logger itself
    logger.addHandler(_ConsoleHandler())
    logger.addFilter(_GpuFilter())
    logger.setLevel(logging.INFO)
    logger.propagate = False

_QUIET = 0
_SINK = None
_ANSWER_SINK = None


def set_progress_sink(sink, *, owner=None) -> None:
    """Send every throttled progress count to ``sink`` as well as the log.

    The sink is called as ``sink(label, done, total, unit)`` from the
    thread that runs the loop, so it must return quickly and must not
    block. Pass None to remove it. The sink is one per process: the
    query loop reads it from module state because its threads and
    worker processes cannot carry a per-query value. With ``owner``,
    the change applies only while ``owner`` is the installed sink, so
    a query that outlived its own execution cannot remove the sink of
    the query that came after it.
    """
    global _SINK
    if owner is None or _SINK is owner:
        _SINK = sink


def set_answer_sink(sink, *, owner=None) -> None:
    """Receive model answers the moment the engine has them.

    The sink is called as ``sink(payload)`` with a dict whose ``kind``
    says what finished: ``"filter"`` for one chunk's finished documents
    (row index, last stage asked, passed), ``"join"`` for one anchor's
    matches (its row index, the partner row index tuples that answered
    true, how many pairs were asked), ``"score"`` for one batch of
    reranker scores (row indices and scores), or ``"evict"`` for one
    document prefix dropped from KV (alias, document index, tokens).
    Called from the loop's thread; it must return quickly. Pass None
    to remove it. One sink per process, as with ``set_progress_sink``;
    ``owner`` works the same way.
    """
    global _ANSWER_SINK
    if owner is None or _ANSWER_SINK is owner:
        _ANSWER_SINK = sink


def answer_sink():
    """Return the answer sink, or None when nobody is listening."""
    return _ANSWER_SINK


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
    """Report a running count: a bar on a terminal, lines anywhere else.

    The progress sink hears the count at most once every `every` seconds.
    A line is logged no more often than that, and for a count with a
    total only when another tenth of it is done.
    """

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
        self._logged = 0
        self._bar = self._task = None
        if emit is say and not _QUIET and total and console().is_terminal:
            self._bar = Bar(
                TextColumn(r"\[quail] {task.description}"), BarColumn(),
                MofNCompleteColumn(), TextColumn(unit), TaskProgressColumn(),
                TimeElapsedColumn(), TimeRemainingColumn(),
                console=console(), transient=True)
            self._task = self._bar.add_task(label, total=total)
            self._bar.start()

    def update(self, done: int) -> None:
        """Record the count; move the bar, or log a line when one is due."""
        if self._bar is not None:
            self._bar.advance(self._task, done - self.done)
        self.done = done
        now = time.perf_counter()
        if now - self._last < self.every:
            return
        self._last = now
        self._report()
        # a tenth of the total since the last line, or any change without one
        step = (self.total or 0) / 10
        if self._bar is None and done - self._logged >= step:
            self._logged = done
            self.emit(self._line(self.label, now))

    def finish(self, label: str, extra: str = "") -> None:
        """Log the final line under a past tense label."""
        if self._bar is not None:
            self._bar.stop()
            self._bar = None
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
            count += f" of {self.total:,}"
        return (f"{label}: {count} {self.unit} in {elapsed:.1f} s "
                f"({rate:,.0f} per second)")
