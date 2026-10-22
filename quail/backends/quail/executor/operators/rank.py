"""Top-k ranking by pairwise comparisons, with a bounded number of asks."""

from __future__ import annotations

import random

import numpy as np
import pyarrow as pa

from quail.catalog.pairs import columns_key
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows


class RankPart:
    """Prepare one AiTopK node's documents, rows, and value columns."""

    execute_group = None

    @staticmethod
    def model_inputs(node, inputs, context, sequence=None, nodes=None):
        """Return the rows, value columns, and documents the ranker reads."""
        alias = node.spec.aliases[0]
        return {
            "rows": next(iter(inputs.values())),
            "columns": context.sources.get(columns_key(alias)),
            "documents": context.state.get("documents"),
        }


def score_pairs(reranker, spec, documents, pairs, stats) -> list[float]:
    """Score pairs and add the batch's token counts to ``stats``.

    Args:
        reranker: Object with ``score(spec, rows, documents)``.
        spec: The rank specification. Its first alias is the pivot.
        documents: Tokenized documents by alias.
        pairs: ``(pivot, other)`` document positions.
        stats: Mutable token counters for the rank node.

    Returns:
        P(the pivot is better than the other row), one per pair.
    """
    if not pairs:
        return []
    # The pivot is column 0, so the scorer computes its prefix once
    # and reuses it for every other row in the batch.
    pair_rows = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
    batch = reranker.score(spec, pair_rows, documents)
    stats["fresh_tokens"] = stats.get("fresh_tokens", 0) + int(batch.fresh_tokens)
    stats["cached_tokens"] = (
        stats.get("cached_tokens", 0) + int(batch.cached_tokens))
    return [float(value) for value in batch.scores]


class _Order:
    """A total order from pairwise probabilities, with ties broken by id."""

    def __init__(self, ask, ids, margin):
        self.ask = ask
        self.ids = ids
        self.margin = margin
        self.cached = {}
        self.num_comparisons = 0

    def probabilities(self, pairs) -> list[float]:
        """Return P(first is better) for each pair, asking only new ones.

        A probability within ``margin`` of 0.5 is asked again with the
        documents swapped, and the two answers are averaged. That
        cancels a preference for whichever document is shown first.
        """
        missing = [pair for pair in pairs if pair not in self.cached]
        if missing:
            fresh = dict(zip(missing, self._asked(missing)))
            pending = []
            for pair, probability in fresh.items():
                reverse = (pair[1], pair[0])
                if (abs(probability - 0.5) <= self.margin
                        and reverse not in self.cached and reverse not in fresh):
                    pending.append(pair)
            reversed_values = {
                pair: value for pair, value in zip(
                    pending, self._asked([(pair[1], pair[0]) for pair in pending]))
            }
            for pair, probability in fresh.items():
                reverse = (pair[1], pair[0])
                if abs(probability - 0.5) > self.margin:
                    self._store(pair[0], pair[1], probability)
                    continue
                if reverse in fresh:
                    other = fresh[reverse]
                elif reverse in self.cached:
                    other = self.cached[reverse]
                else:
                    other = reversed_values[pair]
                self._store(pair[0], pair[1], (probability + (1.0 - other)) / 2.0)
        return [self.cached[pair] for pair in pairs]

    def better(self, first, second) -> bool:
        """Return whether ``first`` ranks above ``second``."""
        if first == second:
            return False
        probability = self.probabilities([(first, second)])[0]
        if probability > 0.5:
            return True
        if probability < 0.5:
            return False
        return self._smaller_id(first, second)

    def beats(self, item, pivot, pivot_better) -> bool:
        """Return whether ``item`` ranks above ``pivot``.

        Args:
            item: The candidate document position.
            pivot: The pivot document position.
            pivot_better: P(the pivot is better than the candidate).
        """
        if pivot_better < 0.5:
            return True
        if pivot_better > 0.5:
            return False
        return self._smaller_id(item, pivot)

    def sorted(self, items) -> list:
        """Return ``items`` best first."""
        return sorted(items, key=self._sort_key)

    def by_id(self, items) -> list:
        """Return ``items`` in id order, then document position."""
        return sorted(items, key=lambda item: (self.ids[item], item))

    def _sort_key(self, item):
        return _RankKey(self, item)

    def _smaller_id(self, first, second) -> bool:
        left, right = self.ids[first], self.ids[second]
        if left == right:
            return first < second
        try:
            return left < right
        except TypeError:
            return str(left) < str(right)

    def _asked(self, pairs) -> list[float]:
        if not pairs:
            return []
        self.num_comparisons += len(pairs)
        return self.ask(pairs)

    def _store(self, first, second, probability) -> None:
        self.cached[(first, second)] = probability
        self.cached[(second, first)] = 1.0 - probability


class _RankKey:
    """Sort key that orders two positions by ``_Order.better``."""

    def __init__(self, order, item):
        self.order = order
        self.item = item

    def __lt__(self, other):
        return self.order.better(self.item, other.item)


def execute_rank(node, inputs, ask, stats=None) -> NodeResult:
    """Return the k best rows of each partition, in rank order.

    Each round draws pivots, compares the other rows with those pivots,
    and keeps the side that holds the k-th row. The kept rows are then
    ordered. A round that cannot shrink the set orders it by row id.

    Args:
        node: The AiTopK node.
        inputs: ``rows`` and ``columns`` from the rank part.
        ask: ``ask(pairs) -> probabilities`` that the first of each
            pair is better. ``pairs`` are document positions.
        stats: Token counters ``ask`` fills, when the model reports them.

    Returns:
        Rows in rank order, with the rank column starting at 1 in each
        partition, and the number of comparisons in the metrics.
    """
    spec = node.spec
    alias = spec.aliases[0]
    rows = inputs["rows"]
    if not isinstance(rows, Rows):
        raise TypeError("AI.RANK reads document rows")
    positions = rows.ids[alias].to_pylist()
    if not positions or spec.k <= 0:
        return _result(alias, spec.name, [], [], 0, len(positions), stats)
    values = _values(inputs.get("columns"), spec)
    ids = {position: values[spec.id_column][position] for position in positions}
    order = _Order(ask, ids, spec.margin)
    groups = {}
    for position in positions:
        key = tuple(values[name][position] for name in spec.partition)
        groups.setdefault(key, []).append(position)
    kept, ranks = [], []
    for key in sorted(groups):
        chosen = _top(
            groups[key], spec.k, _rng(spec.seed, key), order, spec.num_pivots)
        kept.extend(chosen)
        ranks.extend(range(1, len(chosen) + 1))
    return _result(alias, spec.name, kept, ranks, order.num_comparisons,
                   len(positions), stats)


def _rng(seed, key) -> random.Random:
    """Return the pivot generator for one partition.

    Python's random accepts a string seed, not a tuple. The text of
    the seed and the partition key is the same on every run.
    """
    return random.Random(repr((seed, key)))


def _top(items, k, rng, order, num_pivots) -> list:
    """Return the k best items, best first."""
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return order.sorted(items)
    return order.sorted(_best(list(items), k, rng, order, num_pivots))


def _best(items, k, rng, order, num_pivots) -> list:
    """Return the k best items, not yet ordered."""
    if len(items) <= k:
        return list(items)
    num_pivots = min(num_pivots, len(items) - 1)
    if num_pivots < 1:
        return order.by_id(items)[:k]
    indexes = set(rng.sample(range(len(items)), num_pivots))
    pivots = order.sorted(items[index] for index in indexes)
    rest = [item for index, item in enumerate(items) if index not in indexes]
    pairs = [(pivot, item) for item in rest for pivot in pivots]
    looked = {
        pair: probability
        for pair, probability in zip(pairs, order.probabilities(pairs))}
    buckets = [[] for _ in range(len(pivots) + 1)]
    for item in rest:
        slot = len(pivots)
        for index, pivot in enumerate(pivots):
            if order.beats(item, pivot, looked[(pivot, item)]):
                slot = index
                break
        buckets[slot].append(item)
    kept = []
    for index, group in enumerate(buckets):
        need = k - len(kept)
        if need <= 0:
            break
        if len(group) > need:
            if len(group) >= len(items):
                return kept + order.by_id(group)[:need]
            return kept + _best(group, need, rng, order, num_pivots)
        kept.extend(group)
        if index < len(pivots) and len(kept) < k:
            kept.append(pivots[index])
    return kept


def _values(table, spec) -> dict:
    """Return each rank column as a Python list indexed by position."""
    names = (spec.id_column, *spec.partition)
    if table is None or any(name not in table.column_names for name in names):
        missing = [name for name in names
                   if table is None or name not in table.column_names]
        raise KeyError(f"AI.RANK needs column {missing[0]!r}")
    return {name: table.column(name).combine_chunks().to_pylist() for name in names}


def _result(alias, name, kept, ranks, num_comparisons, num_input, stats):
    rows = Rows(
        {alias: kept},
        pa.table({name: pa.array(ranks, pa.int32())}))
    fresh = 0 if not stats else int(stats.get("fresh_tokens", 0))
    cached = 0 if not stats else int(stats.get("cached_tokens", 0))
    return NodeResult(
        {"rows": rows},
        NodeMetrics(
            input_rows=num_input,
            output_rows=len(kept),
            evaluated_document_pairs=num_comparisons,
            fresh_tokens=fresh,
            cached_tokens=cached,
            extension={"comparisons": num_comparisons}))
