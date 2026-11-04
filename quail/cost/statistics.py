"""Token length statistics of a table, as the cost model reads them."""

from dataclasses import dataclass

import pyarrow.compute as pc

from quail.common.work import triangle


@dataclass(frozen=True)
class AliasStats:
    """Length sums and counts below the attention window for one alias."""

    count: int
    total: int
    squared: int
    maximum: int
    window: int = 0
    short_counts: tuple[tuple[int, int], ...] = ()

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def variance(self) -> float:
        if not self.count:
            return 0.0
        return (self.squared * self.count - self.total * self.total) / self.count**2

    def window_pairs(self, offset: float) -> float:
        """Sum sliding pairs for document lengths plus an offset."""
        window = self.window
        linear = window * (self.total + offset * self.count)
        linear -= self.count * window * (window - 1) / 2
        return linear + sum(
            count * (triangle(length + offset)
                     - window * (length + offset) + window * (window - 1) / 2)
            for length, count in self.short_counts if length + offset < window)

    def window_reads(self, offset: float) -> float:
        """Sum retained prefix keys visible to the first suffix token."""
        cap = self.window - 1
        return self.count * cap - sum(
            count * (cap - length - offset)
            for length, count in self.short_counts if length + offset < cap)


def summarize_alias(lengths, window: int = 0, *, memo=None) -> AliasStats:
    """Summarize a length column, optionally reusing a query's cached summary."""
    key = ("alias_stats", id(lengths), window)
    if memo is not None and key in memo:
        return memo[key][1]
    values = lengths.arrow_array
    short_counts = ()
    if window:
        counts = pc.value_counts(pc.filter(values, pc.less(values, window)))
        short_counts = tuple(sorted(zip(
            counts.field("values").to_pylist(), counts.field("counts").to_pylist())))
    summary = AliasStats(
        count=len(values),
        total=pc.sum(values).as_py() or 0,
        squared=pc.sum(pc.multiply(values, values)).as_py() or 0,
        maximum=pc.max(values).as_py() or 0,
        window=window,
        short_counts=short_counts,
    )
    if memo is not None:
        # Keep the source alive so another column cannot reuse its identity.
        memo[key] = (lengths, summary)
    return summary


def scale_alias(stats: AliasStats, fraction: float) -> AliasStats:
    """Return the summary of the fraction of the documents expected to remain."""
    return AliasStats(
        count=round(stats.count * fraction),
        total=round(stats.total * fraction),
        squared=round(stats.squared * fraction),
        maximum=stats.maximum,
        window=stats.window,
        short_counts=tuple((length, round(count * fraction))
                           for length, count in stats.short_counts),
    )
