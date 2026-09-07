"""Progress lines for the long steps of a query."""

import time


def say(message: str) -> None:
    """Print one progress line right away."""
    print(f"quail: {message}", flush=True)


class Progress:
    """Report a running count, printing at most once every `every` seconds."""

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
        """Record the count and print a line when enough time has passed."""
        self.done = done
        now = time.perf_counter()
        if now - self._last >= self.every:
            self._last = now
            say(self._line(self.label, now))

    def finish(self, label: str, extra: str = "") -> None:
        """Print the final line under a past tense label."""
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
