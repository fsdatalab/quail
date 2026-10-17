"""Build score columns and evaluate numeric score comparisons."""

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.backends.base import GpuContext
from quail.execution.ids import id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import (
    DEFAULT_BATCH_ROWS,
    answer_table,
    filter_answer_table,
)
from quail.execution.rows import Rows
from quail.logical import PROBABILITIES_SUFFIX
from quail.physical import AiClassify, AiScore, ClassifySpec, Filter, ValueType
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
    """Values and execution metrics from one score or classification batch.

    Attributes:
        scores: Numeric scores or labels, in input row order.
        fresh_tokens: Token positions computed by the model.
        cached_tokens: Token positions reused from KV.
        suffix_tokens: Classification tokens processed after document frames.
        borrowed_tokens: Prefix tokens reused from another document's KV.
        pack_s: Host seconds spent building forward chunks.
        gpu_s: Forward-pass GPU seconds, or zero when timing is disabled.
        chunks: Timed forward passes, or zero when timing is disabled.
        probabilities: Each label's probability per input row when the
            classification asks for them, else None.
    """

    scores: np.ndarray
    fresh_tokens: int
    cached_tokens: int
    suffix_tokens: int = 0
    borrowed_tokens: int = 0
    pack_s: float = 0.0
    gpu_s: float = 0.0
    chunks: int = 0
    probabilities: np.ndarray | None = None


def _value_type(spec) -> pa.DataType:
    """Return the Arrow type for a label or a numeric score."""
    return pa.string() if isinstance(spec, ClassifySpec) else pa.float64()


def probability_column(labels, probabilities) -> pa.MapArray:
    """Build an Arrow map column of label probabilities.

    Args:
        labels: Labels in probability-column order.
        probabilities: Array with shape (documents, labels).

    Returns:
        A MapArray from label to probability for each document. A row with
        a NaN probability is null: its document has no answer.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(
        -1, len(labels))
    count, width = probabilities.shape
    return pa.MapArray.from_arrays(
        pa.array(np.arange(0, count * width + 1, width), pa.int32()),
        pa.array(np.tile(np.asarray(labels, dtype=object), count), pa.string()),
        pa.array(np.nan_to_num(probabilities).ravel(), pa.float64()),
        mask=pa.array(np.isnan(probabilities).any(axis=1)))


def score_rows(rows, aliases, name, scores, value_type=None) -> Rows:
    """The rows a score or classification returns: the ids and one value column."""
    return Rows({alias: rows[:, index] for index, alias in enumerate(aliases)},
                pa.table({name: pa.array(scores, type=value_type or pa.float64())}))


def _ids(value, alias: str) -> np.ndarray:
    if isinstance(value, pa.Table):
        value = value.column(alias)
    return id_positions(value)


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
        if port.value_type is ValueType.ROWS:
            for alias in spec.aliases:
                if alias in value.ids:
                    by_alias[alias] = id_positions(value.ids[alias])
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


def _batches_with_positions(rows: ScoreRows, size: int):
    start = 0
    for batch in rows.batches(size):
        end = start + len(batch)
        yield rows.positions(start, end), batch
        start = end


def scored_batch(node, rows, values) -> dict:
    """The answer-sink payload for one batch of scored rows.

    ``rows`` are the batch's row indices into each alias table, one int
    per row for a single alias and one list per row for a pair;
    ``values`` are their scores or labels, in the same order.
    """
    rows = np.asarray(rows)
    values = list(values)
    payload = {"kind": "score", "node": node.node_id,
               "output": node.spec.name, "aliases": list(node.spec.aliases),
               "rows": (rows[:, 0].tolist() if rows.shape[1] == 1
                        else rows.tolist())}
    if isinstance(node.spec, ClassifySpec):
        return {**payload, "kind": "label", "labels": values}
    return {**payload, "scores": [round(float(value), 4) for value in values]}


def classify_outputs(node, rows: Rows) -> dict:
    """Return the label rows and, for one alias, the labeled document IDs."""
    outputs = {"rows": rows}
    if len(node.spec.aliases) == 1:
        (alias,) = node.spec.aliases
        outputs[f"ids:{alias}"] = rows.ids[alias]
    return outputs


def score_in_batches(node, inputs, score_batches, shards: int = 1,
                     streamed: bool = False) -> NodeResult:
    """Score the node's candidate rows, in input order.

    Each shard's rows are scored in one call of ``score_batches``. A
    Cartesian product is scored in batches of DEFAULT_BATCH_ROWS pairs.

    Args:
        node: The AiScore node.
        inputs: Its input values by port name.
        score_batches: Callable (node, batches) taking one row batch per
            shard and returning one NodeResult per batch, each with a
            "rows" table holding one row per input row in order.
        shards: Row shards scored side by side; rows that share a first
            document land in the same shard.
        streamed: Whether ``score_batches`` sends its answers to the
            answer sink itself. Otherwise each batch's answers are sent
            when the batch is scored.
    """
    if not isinstance(node, (AiScore, AiClassify)):
        raise TypeError(type(node).__name__)
    if node.spec is None:
        raise ValueError("AI.SCORE needs a score specification")
    started = time.perf_counter()
    rows, priors = _candidate_rows(node, inputs)
    parts = rows.shard(shards) if shards > 1 else [rows]
    width = len(node.spec.aliases)
    scored = []
    positions = []
    metrics = NodeMetrics()
    # a product's pairs are built per batch: the whole product may not
    # fit in host memory
    streams = [_batches_with_positions(
        part, DEFAULT_BATCH_ROWS if part.product else max(1, len(part)))
        for part in parts]
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
            scored.append(result.outputs["rows"])
            positions.append(where)
            metrics += result.metrics
            sink = answer_sink()
            if sink is not None and not streamed and len(batch):
                sink(scored_batch(
                    node, batch,
                    result.outputs["rows"].column(node.spec.name).to_pylist()))
    rows = Rows.concat(scored)
    order = np.concatenate(positions)
    if len(order) and np.any(np.diff(order) < 0):
        rows = rows.take(np.argsort(order, kind="stable"))
    for alias, prior in priors.items():
        rows = rows.joined(prior, alias)
    outputs = {"rows": rows}
    if isinstance(node.spec, ClassifySpec):
        # a document whose decoded answer names no label leaves the query
        rows = rows.filter(pc.is_valid(rows.column(node.spec.name)))
        metrics = replace(metrics, output_rows=rows.num_rows)
        outputs = classify_outputs(node, rows)
    return NodeResult(outputs, replace(
        metrics, wall_s=time.perf_counter() - started,
        extension={**metrics.extension, "output": node.spec.name,
                   "aliases": list(node.spec.aliases), "input_rows": len(rows)},
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
            streamed=True,
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

        empty = np.empty(0, dtype=(
            object if isinstance(spec, ClassifySpec) else np.float32))
        sink = answer_sink()
        streams = (sink is not None and len(rows)
                   and getattr(self.reranker, "streams_answers", False))
        extra = {}
        if streams:
            extra["on_answers"] = lambda positions, values: sink(
                scored_batch(node, rows[positions], values))
        batch = (
            self.reranker.score(spec, rows, self.documents, **extra)
            if len(rows) else RerankerBatch(empty, 0, 0)
        )
        if sink is not None and len(rows) and not streams:
            sink(scored_batch(node, rows, batch.scores))
        scored = score_rows(rows, spec.aliases, spec.name, batch.scores,
                           _value_type(spec))
        if isinstance(spec, ClassifySpec) and spec.probabilities:
            scored = scored.with_column(
                spec.name + PROBABILITIES_SUFFIX, probability_column(
                    spec.labels,
                    np.full((len(rows), len(spec.labels)), np.nan)
                    if batch.probabilities is None else batch.probabilities))
        count = len(rows)
        return NodeResult(
            {"rows": scored},
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
                gpu_s=batch.gpu_s,
                chunks=batch.chunks,
                extension={
                    "output": spec.name,
                    "aliases": list(spec.aliases),
                    "input_rows": count,
                    "borrowed_prefix_tokens": batch.borrowed_tokens,
                    **({"suffix_tokens": batch.suffix_tokens,
                        "pack_s": batch.pack_s}
                       if isinstance(spec, ClassifySpec) else {}),
                },
            ),
        )


def _filter_answer_table(node: Filter, rows: Rows, answers) -> pa.Table:
    """Build one answer row per document using the filter answer schema."""
    alias = node.aliases[0]
    return filter_answer_table(
        alias, rows.ids[alias],
        np.full(rows.num_rows, node.written_position, dtype=np.int32), answers)


class FilterRuntime:
    """Keep the rows of a score or label table that pass the predicate."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Filter):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Filter needs one score or label input")
        rows = next(iter(inputs.values()))
        return filter_scores(node, rows)


def filter_scores(node, rows: Rows) -> NodeResult:
    """Apply a filter to rows of scores, labels, or answers."""
    answers = node.predicate.accepts(rows.column(node.column))
    filtered = rows.filter(answers)
    if len(node.aliases) == 1:
        answer_name = f"filter_answers:{node.aliases[0]}"
        answer_relation = _filter_answer_table(node, rows, answers)
    else:
        left, right = node.aliases
        answer_name = f"join_answers:{node.written_position}"
        answer_relation = answer_table(
            {left: rows.ids[left], right: rows.ids[right]},
            answers,
            "join_answers",
            metadata={
                "written_position": node.written_position,
                "anchor": left,
                "partners": right,
                "semantics": "full",
                "comparison": node.predicate.comparison,
                "threshold": node.predicate.value,
            },
        )
    outputs = {"rows": filtered, answer_name: answer_relation}
    if len(node.aliases) == 1:
        outputs[f"ids:{node.aliases[0]}"] = filtered.ids[node.aliases[0]]
    return NodeResult(
        outputs,
        NodeMetrics(input_rows=rows.num_rows, output_rows=filtered.num_rows),
    )
