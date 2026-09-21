"""Build score columns and evaluate numeric score comparisons."""

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.backends.base import GpuContext
from quail.execution.result import DEFAULT_BATCH_ROWS, answer_table
from quail.execution.runner import (
    NodeMetrics,
    NodeResult,
)
from quail.physical import AiScore, ScoreFilter, ValueType
from quail.progress import answer_sink

# one kernel per entry of quail.logical.SCORE_COMPARISONS
_COMPARE = {
    "<": pc.less, "<=": pc.less_equal,
    ">": pc.greater, ">=": pc.greater_equal,
}


def compare_score(scores, comparison: str, threshold: float):
    """Compare a score array with a scalar threshold."""
    if comparison not in _COMPARE:
        raise ValueError(f"unsupported AI.SCORE comparison {comparison!r}")
    return _COMPARE[comparison](scores, threshold)


@dataclass(frozen=True)
class ScoreRows:
    """Document indices with an optional unexpanded Cartesian product.

    ``origin`` maps each row, or each left document of a product, to
    its position in the rows this was sharded from. None means the
    rows are in their original order.
    """

    columns: tuple[np.ndarray, ...]
    product: bool = False
    origin: np.ndarray | None = None

    def __len__(self):
        if self.product:
            return len(self.columns[0]) * len(self.columns[1])
        return len(self.columns[0])

    def shard(self, count: int) -> list["ScoreRows"]:
        """Split the rows by first document, so its KV stays on one GPU."""
        first = self.columns[0]
        shards = []
        for worker in range(count):
            keep = np.flatnonzero(first % count == worker)
            origin = keep if self.origin is None else self.origin[keep]
            columns = (
                (first[keep], *self.columns[1:]) if self.product
                else tuple(column[keep] for column in self.columns)
            )
            shards.append(replace(self, columns=columns, origin=origin))
        return shards

    def positions(self, start: int, end: int) -> np.ndarray:
        """Return the original positions of rows start to end."""
        local = np.arange(start, end, dtype=np.int64)
        if not self.product:
            return local if self.origin is None else self.origin[local]
        width = len(self.columns[1])
        left = local // width
        if self.origin is not None:
            left = self.origin[left]
        return left * width + local % width

    def batches(self, size=DEFAULT_BATCH_ROWS):
        """Yield bounded arrays in input order.

        A product batch ends on a first-document boundary when a
        document's partners fit in one batch, so its prefix KV is
        computed once.
        """
        if self.product:
            width = len(self.columns[1])
            if 0 < width <= size:
                size -= size % width
        for start in range(0, max(1, len(self)), size):
            end = min(start + size, len(self))
            if self.product and end > start:
                indices = np.arange(start, end, dtype=np.int64)
                left, right = self.columns
                yield np.column_stack((left[indices // len(right)],
                                       right[indices % len(right)]))
            elif self.product:
                yield np.empty((0, 2), dtype=np.int32)
            else:
                yield np.column_stack([column[start:end] for column in self.columns])


@dataclass(frozen=True)
class RerankerBatch:
    """Scores and token counts returned by one reranker call."""

    scores: np.ndarray
    fresh_tokens: int
    cached_tokens: int


def _score_table(rows, aliases, name, scores) -> pa.Table:
    arrays = {
        alias: pa.array(rows[:, index], type=pa.int32())
        for index, alias in enumerate(aliases)
    }
    arrays[name] = pa.array(scores, type=pa.float64())
    return pa.table(arrays).replace_schema_metadata({
        b"quail.kind": b"score_rows",
        b"quail.aliases": ",".join(aliases).encode("utf-8"),
    })


def _ids(value, alias: str) -> np.ndarray:
    if isinstance(value, pa.Table):
        return value.column(alias).to_numpy()
    return np.asarray(value, dtype=np.int32)


def _table_aliases(table: pa.Table) -> tuple[str, ...]:
    metadata = table.schema.metadata or {}
    return tuple(metadata.get(b"quail.aliases", b"").decode("utf-8").split(","))


def _candidate_rows(node, inputs):
    """Return the rows to score and earlier score tables by alias."""
    spec = node.spec
    by_alias = {}
    priors = {}
    pairs = None
    for port in node.inputs:
        value = inputs[port.name]
        if port.value_type is ValueType.PAIRS:
            pairs = value
            continue
        if port.value_type is ValueType.SCORES:
            for alias in spec.aliases:
                if alias in value.column_names:
                    by_alias[alias] = _ids(value, alias)
                    priors[alias] = value
            continue
        if port.source.port.startswith("ids:"):
            alias = port.source.port.split(":", 1)[1]
            by_alias[alias] = _ids(value, alias)

    if len(spec.aliases) == 1:
        return ScoreRows((by_alias[spec.aliases[0]],)), priors

    left, right = spec.aliases
    if pairs is None:
        rows = ScoreRows((by_alias[left], by_alias[right]), product=True)
        return rows, priors
    selected = pairs.filter(pc.and_(
        pc.is_in(pairs.column(left), value_set=pa.array(by_alias[left])),
        pc.is_in(pairs.column(right), value_set=pa.array(by_alias[right])),
    ))
    return ScoreRows((_ids(selected, left), _ids(selected, right))), priors


def attach_prior_columns(table: pa.Table, priors: Mapping[str, pa.Table]):
    """Carry score columns computed earlier for each alias onto a table.

    Every row of the table names a document that the alias's earlier
    table also holds, so the lookup by document index never misses.
    """
    for alias, prior in priors.items():
        extra = [
            name for name in prior.column_names
            if name not in _table_aliases(prior)
        ]
        if not extra:
            continue
        positions = pc.index_in(
            table.column(alias), value_set=prior.column(alias)
        )
        for name in extra:
            table = table.append_column(
                name, pc.take(prior.column(name), positions)
            )
    return table


def _batches_with_positions(rows: ScoreRows):
    start = 0
    for batch in rows.batches():
        end = start + len(batch)
        yield rows.positions(start, end), batch
        start = end


def scored_batch(node, rows, table) -> dict:
    """The answer-sink payload for one scored batch.

    ``rows`` are the batch's row indices into each alias table, one int
    per row for a single alias and one list per row for a pair;
    ``scores`` line up with them.
    """
    rows = np.asarray(rows)
    return {"kind": "score", "node": node.node_id, "output": node.spec.name,
            "aliases": list(node.spec.aliases),
            "rows": (rows[:, 0].tolist() if rows.shape[1] == 1
                     else rows.tolist()),
            "scores": [round(float(value), 4)
                       for value in table.column(node.spec.name).to_pylist()]}


def score_in_batches(node, inputs, score_batches, shards: int = 1) -> NodeResult:
    """Score the node's candidate rows in bounded batches, in input order.

    Args:
        node: The AiScore node.
        inputs: Its input values by port name.
        score_batches: Callable (node, batches) taking one row batch per
            shard and returning one NodeResult per batch, each with a
            "scores" table holding one row per input row in order.
        shards: Row shards scored side by side; rows that share a first
            document land in the same shard.
    """
    if not isinstance(node, AiScore):
        raise TypeError(type(node).__name__)
    if node.spec is None:
        raise ValueError("AI.SCORE needs a score specification")
    started = time.perf_counter()
    rows, priors = _candidate_rows(node, inputs)
    parts = rows.shard(shards) if shards > 1 else [rows]
    width = len(node.spec.aliases)
    tables = []
    positions = []
    metrics = NodeMetrics()
    streams = [_batches_with_positions(part) for part in parts]
    while streams:
        rounds = [next(stream, None) for stream in streams]
        if all(item is None for item in rounds):
            break
        batches = [
            (np.empty(0, dtype=np.int64), np.empty((0, width), dtype=np.int32))
            if item is None else item
            for item in rounds
        ]
        results = score_batches(node, [batch for _, batch in batches])
        for (where, batch), result in zip(batches, results):
            tables.append(result.outputs["scores"])
            positions.append(where)
            metrics += result.metrics
            sink = answer_sink()
            if sink is not None and len(batch):
                sink(scored_batch(node, batch, result.outputs["scores"]))
    table = pa.concat_tables(tables)
    order = np.concatenate(positions)
    if len(order) and np.any(np.diff(order) < 0):
        table = table.take(pa.array(np.argsort(order, kind="stable")))
    table = attach_prior_columns(table, priors)
    return NodeResult({"scores": table}, replace(
        metrics, wall_s=time.perf_counter() - started,
        extension={"output": node.spec.name, "aliases": list(node.spec.aliases),
                   "input_rows": len(rows)},
    ))


class RerankerModelExecution:
    """Compute score columns without applying their comparisons."""

    def __init__(self, context: GpuContext):
        self.reranker = context.query_settings["reranker"]
        self.documents = context.query_settings["documents"]

    def execute(
        self,
        node: AiScore,
        inputs: Mapping[str, object],
    ) -> NodeResult:
        return score_in_batches(
            node, inputs,
            lambda node, batches: [self.execute_rows(node, b) for b in batches],
        )

    def execute_rows(self, node, rows):
        """Compute a score for each supplied row or document pair."""
        started = time.perf_counter()
        spec = node.spec
        rows = np.asarray(rows, dtype=np.int32).reshape(-1, len(spec.aliases))
        if len(spec.arguments) != len(spec.aliases):
            raise ValueError(
                "AI.SCORE needs one prompt argument per document relation"
            )

        batch = (
            self.reranker.score(spec, rows, self.documents)
            if len(rows) else RerankerBatch(np.empty(0, dtype=np.float32), 0, 0)
        )
        table = _score_table(rows, spec.aliases, spec.name, batch.scores)
        count = len(rows)
        return NodeResult(
            {"scores": table},
            NodeMetrics(
                wall_s=time.perf_counter() - started,
                input_rows=count,
                output_rows=count,
                evaluated_documents=(
                    count if len(spec.aliases) == 1 else 0
                ),
                evaluated_document_pairs=(
                    count if len(spec.aliases) == 2 else 0
                ),
                fresh_tokens=batch.fresh_tokens,
                cached_tokens=batch.cached_tokens,
                extension={
                    "output": spec.name,
                    "aliases": list(spec.aliases),
                    "input_rows": count,
                },
            ),
        )


def _filter_answer_table(node: ScoreFilter, table, answers) -> pa.Table:
    alias = node.aliases[0]
    return pa.table({
        alias: pc.cast(table.column(alias), pa.int32()),
        "predicate": pa.array(
            np.full(table.num_rows, node.written_pos, dtype=np.int32), type=pa.int32()
        ),
        "answer": pa.array(answers, type=pa.bool_()),
    }).replace_schema_metadata({
        b"quail.kind": b"filter_answers",
        b"quail.alias": alias.encode("utf-8"),
    })


class ScoreFilterRuntime:
    """Apply a numeric score comparison and retain the score column."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, ScoreFilter):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("ScoreFilter needs one score input")
        table = next(iter(inputs.values()))
        answers = compare_score(
            table.column(node.score_name), node.comparison, node.threshold,
        )
        filtered = table.filter(answers)
        if len(node.aliases) == 1:
            answer_name = f"filter_answers:{node.aliases[0]}"
            answer_relation = _filter_answer_table(node, table, answers)
        else:
            left, right = node.aliases
            answer_name = f"join_answers:{node.written_pos}"
            answer_relation = answer_table(
                {
                    left: table.column(left),
                    right: table.column(right),
                },
                answers,
                "join_answers",
                metadata={
                    "written_pos": node.written_pos,
                    "anchor": left,
                    "partners": right,
                    "semantics": "full",
                    "comparison": node.comparison,
                    "threshold": node.threshold,
                },
            )
        return NodeResult(
            {
                "scores": filtered,
                answer_name: answer_relation,
            },
            NodeMetrics(
                input_rows=table.num_rows,
                output_rows=filtered.num_rows,
            ),
        )
