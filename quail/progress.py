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
_ANSWER_SINK = None
_RUNNING = ()
_LISTENERS = []


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


@contextlib.contextmanager
def running(node_ids):
    """Attribute the Progress counts created inside to these plan node ids."""
    global _RUNNING
    previous, _RUNNING = _RUNNING, tuple(node_ids)
    try:
        yield
    finally:
        _RUNNING = previous


@contextlib.contextmanager
def listening(listener):
    """Call ``listener(progress)`` with each throttled count while inside.

    Listeners run before the progress sink, from the thread that runs
    the loop, so they must return quickly.
    """
    _LISTENERS.append(listener)
    try:
        yield
    finally:
        _LISTENERS.remove(listener)


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
    """Report a running count, logging at most once every `every` seconds.

    node_ids are the plan nodes running when the count was created,
    from ``running()``; empty outside a query's operators.
    """

    def __init__(self, label: str, total: int | None = None,
                 unit: str = "documents", every: float = 5.0, *, emit=say):
        self.label = label
        self.total = total
        self.unit = unit
        self.every = every
        self.emit = emit
        self.done = 0
        self.node_ids = _RUNNING
        self.finished = False
        self.started = time.perf_counter()
        self.elapsed = 0.0
        self._last = self.started

    def update(self, done: int) -> None:
        """Record the count and log a line when enough time has passed."""
        self.done = done
        now = time.perf_counter()
        if now - self._last >= self.every:
            self._last = now
            self.emit(self._line(self.label, now))
            self._report(now)

    def finish(self, label: str, extra: str = "") -> None:
        """Log the final line under a past tense label."""
        now = time.perf_counter()
        line = self._line(label, now)
        self.emit(f"{line}, {extra}" if extra else line)
        self.finished = True
        self._report(now)

    def _report(self, now: float) -> None:
        self.elapsed = now - self.started
        for listener in tuple(_LISTENERS):
            listener(self)
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
