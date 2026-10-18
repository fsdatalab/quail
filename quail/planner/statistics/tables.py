"""Per-table document-length summaries and the survivors of joins."""

import math
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

from quail.cost.joins import PricedJoin
from quail.cost.work import triangle
from quail.logical import (
    effective_selectivity,
)


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
    values = getattr(lengths, "arrow_array", lengths)
    if not isinstance(values, (pa.Array, pa.ChunkedArray)):
        values = pa.array(values, type=pa.int64())
    else:
        values = values.cast(pa.int64())
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


def alias_stats(lengths: dict, window: int = 0) -> dict[str, AliasStats]:
    """Normalize raw length lists or accept summaries from a caller."""
    return {
        alias: (values if isinstance(values, AliasStats)
                else summarize_alias(values, window))
        for alias, values in lengths.items()
    }


def surviving_documents(num_documents: float, num_partners: float,
                        tuple_selectivity) -> float:
    """Expected documents with at least one matching tuple.

    Tuple matches are taken as independent and evenly spread.
    """
    tuple_selectivity = effective_selectivity(tuple_selectivity)
    if tuple_selectivity >= 1:
        probability = float(num_partners > 0)
    else:
        # log1p and expm1 keep a tiny match probability over many
        # partners from rounding to zero
        probability = -math.expm1(num_partners * math.log1p(-tuple_selectivity))
    return num_documents * probability


def thin(expected_rows: dict, spec: PricedJoin) -> None:
    """Update live document counts after one join's selectivity.

    A tuple matches when the equality conditions keep it and the model
    accepts it.
    """
    sel = effective_selectivity(spec.selectivity) * spec.pair_fraction
    aliases = spec.aliases

    def others(x):
        out = 1.0
        for a in aliases:
            if a != x:
                out *= expected_rows[a]
        return out

    if spec.semantics == "full":
        new = {a: surviving_documents(expected_rows[a], others(a), sel)
               for a in aliases}
        expected_rows.update(new)
    else:
        anchor = spec.anchor
        matched = surviving_documents(expected_rows[anchor], others(anchor), sel)
        expected_rows[anchor] = (matched if spec.semantics == "exists"
                        else expected_rows[anchor] - matched)


def cross_tuples(spec: PricedJoin, expected_rows: dict) -> float:
    """Expected tuples of one join.

    The live cross product, or the fraction of it the join's equality
    conditions keep.
    """
    tuples = 1.0
    for a in spec.aliases:
        tuples *= expected_rows[a]
    return tuples * spec.pair_fraction


def pair_counts(expected_rows, aliases, fraction) -> tuple[float, float]:
    """Return expected pairs and first documents with at least one pair."""
    left, right = aliases
    return (expected_rows[left] * expected_rows[right] * fraction,
            surviving_documents(expected_rows[left], expected_rows[right], fraction))
