"""Progress messages for the long steps of a query, on the "quail" logger.

On a terminal a query shows its steps: one line for each, and for a
running count one bar that updates in place with the count, the rate,
and the time left. Details, such as how many forward passes a step
took, are left out there. Anywhere else, such as a log file or a Modal
log, every message is one plain line, details included, and a count
logs a line at each tenth of its total.
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
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.progress import Progress as Bar
from rich.text import Text

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
    """Write each message to `console()`: a step on a terminal, else a plain line."""

    def emit(self, record):
        found = console()
        # a record of a child logger skips this logger's filter; format()
        # adds a logged exception's traceback
        tag, text = getattr(record, "gpu", ""), self.format(record)
        if not found.is_terminal:
            stamp = time.strftime("%m-%d %H:%M:%S", time.localtime(record.created))
            # a log file keeps one unwrapped line per message
            print(f"{record.levelname} {stamp} [quail]{tag} {text}",
                  file=found.file, flush=True)
        elif not getattr(record, "detail", False):
            done = getattr(record, "done", False)
            mark = "[green]✓[/green]" if done else "[dim]·[/dim]"
            found.print(f"  {mark} [dim]{escape(tag)}[/dim]{escape(text.strip())}",
                        soft_wrap=True)


class _CountColumn(ProgressColumn):
    """A count beside its total, with thousands separators: 743 of 1,772 documents."""

    def render(self, task):
        return Text(f"{int(task.completed):,} of {int(task.total):,} "
                    f"{task.fields['unit']}")


class _RateColumn(ProgressColumn):
    """The rate the count is moving at, from its recent updates: 16 documents/s."""

    def render(self, task):
        if task.speed is None:
            return Text("")
        return Text(f"{task.speed:,.0f} {task.fields['unit']}/s", style="bold")


def _clock(seconds: float) -> str:
    """Write a duration as minutes and seconds, with hours when it has them: 1:04."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


class _TimeColumn(ProgressColumn):
    """The time a count has run and the time it has left: 0:46 elapsed, 1:04 left."""

    def render(self, task):
        text = f"{_clock(task.elapsed or 0)} elapsed"
        if task.time_remaining is not None:
            text += f", {_clock(task.time_remaining)} left"
        return Text(text, style="dim")


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


def say(message: str, *, done: bool = False) -> None:
    """Log one progress message at INFO, unless inside quiet().

    Args:
        message: The step, such as "model ready".
        done: Whether the message reports a finished step.
    """
    if not _QUIET:
        logger.info(message, extra={"done": done})


def detail(message: str) -> None:
    """Log a detail of the step before it, which a terminal leaves out."""
    if not _QUIET:
        logger.info(message, extra={"detail": True})


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
                TextColumn(" "), SpinnerColumn(), TextColumn("{task.description}"),
                BarColumn(complete_style="green", finished_style="green"),
                _CountColumn(), TaskProgressColumn("{task.percentage:>3.0f}%"),
                _RateColumn(), _TimeColumn(), console=console(), transient=True)
            self._task = self._bar.add_task(label, total=total, unit=unit)
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
        line = f"{line}, {extra}" if extra else line
        if self.emit is say:
            say(line, done=True)
        else:
            self.emit(line)
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
