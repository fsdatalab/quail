"""Build score columns and evaluate numeric score comparisons."""

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.backends.base import GpuContext
from quail.common.expressions import PROBABILITIES_SUFFIX
from quail.execution.ids import id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows, rows_of
from quail.execution.streaming import region_stream
from quail.physical import AiClassify, AiScore, ClassifySpec, ValueType
from quail.progress import answer_sink

# one kernel per entry of quail.common.expressions.SCORE_COMPARISONS
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
    """Document indices with an optional unexpanded Cartesian product."""

    columns: tuple[np.ndarray, ...]
    product: bool = False

    def __len__(self):
        if self.product:
            return len(self.columns[0]) * len(self.columns[1])
        return len(self.columns[0])

    def array(self) -> np.ndarray:
        """Every row as one array, one column per alias, in input order.

        A product's rows run through the second alias for each row of the
        first.
        """
        if self.product:
            left, right = self.columns
            return np.column_stack((np.repeat(left, len(right)),
                                    np.tile(right, len(left))))
        return np.column_stack(self.columns)


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
            if port.source.port.startswith("rows:"):
                value = rows_of(value, port.source.port.split(":", 1)[1])
            for alias in spec.aliases:
                if alias in value.ids:
                    by_alias[alias] = id_positions(value.ids[alias])
                    priors[alias] = value
            continue

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
    """Return the label rows on the node's rows port."""
    return {node.rows_port: rows}


def _scored_outputs(node, scored: Rows, priors) -> dict:
    """The node's outputs for scored rows: their priors' values, labels kept."""
    for alias, prior in priors.items():
        scored = scored.joined(prior, alias)
    if isinstance(node.spec, ClassifySpec):
        # a document whose decoded answer names no label leaves the query
        return classify_outputs(node, scored.filter(
            pc.is_valid(scored.column(node.spec.name))))
    return {node.rows_port: scored}


def _stream_scored(node, priors):
    """Callable(rows, values) sending scored rows to the region's stream, or None.

    None unless the node is the region's last operator; a classification
    that reports label probabilities sends its rows when it finishes.
    """
    stream = region_stream()
    spec = node.spec
    if stream is None or not stream.wants(node) or (
            isinstance(spec, ClassifySpec) and spec.probabilities):
        return None

    def on_scored(rows, values):
        scored = score_rows(np.asarray(rows).reshape(-1, len(spec.aliases)),
                            spec.aliases, spec.name, list(values), _value_type(spec))
        stream.partial(node, _scored_outputs(node, scored, priors))

    return on_scored


def score_node(node, inputs, score_batch) -> NodeResult:
    """Score the node's candidate rows in one call, in input order.

    Under a region hint over the scored table, the call skips each row
    the hint rejects when the row would be admitted, reading the hint
    again whenever it tightens; a skipped row has no output row.

    Args:
        node: The AiScore or AiClassify node.
        inputs: Its input values by port name.
        score_batch: Callable (node, rows, on_scored, skip) returning
            the NodeResult, with one row per input row it did not skip,
            in order; ``on_scored(rows, values)``, when not None, is run
            with each part of the rows as it is scored, and ``skip``,
            when not None, is a callable(row position) -> bool, true for
            a row to leave unasked.
    """
    if not isinstance(node, (AiScore, AiClassify)):
        raise TypeError(type(node).__name__)
    if node.spec is None:
        raise ValueError("AI.SCORE needs a score specification")
    started = time.perf_counter()
    rows, priors = _candidate_rows(node, inputs)
    on_scored = _stream_scored(node, priors)
    stream = region_stream()
    alias = node.spec.aliases[0]
    batch = rows.array()
    skip = None
    if (stream is not None and stream.hint is not None and not rows.product
            and len(node.spec.aliases) == 1 and alias == stream.alias):
        def skip(position):
            return not stream.keeps(alias, int(batch[position, 0]))
    result = score_batch(node, batch, on_scored, skip)
    metrics = result.metrics
    outputs = _scored_outputs(node, result.outputs[node.rows_port], priors)
    rows = outputs[node.rows_port]
    if isinstance(node.spec, ClassifySpec):
        metrics = replace(metrics, output_rows=rows.num_rows)
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
        return score_node(node, inputs, self.execute_rows)

    def execute_rows(self, node, rows, on_scored=None, skip=None):
        """Compute a score for each supplied row or document pair.

        Args:
            node: The AiScore or AiClassify node.
            rows: Document indices, one column per alias.
            on_scored: Optional callable(rows, values) run with each
                part of the rows as its values are read.
            skip: Optional callable(row position) -> bool, asked when a
                row of one alias would be admitted; a row it is true for
                is not asked and has no output row.
        """
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
        streams = ((sink is not None or on_scored is not None) and len(rows)
                   and self.reranker.streams_answers)
        extra = {}
        skipped = np.zeros(len(rows), dtype=bool)
        if skip is not None:
            def skip_row(position):
                skipped[position] = skip(position)
                return bool(skipped[position])

            extra["skip"] = skip_row
        if streams:
            def on_answers(positions, values):
                asked = ~skipped[positions]
                positions = np.asarray(positions)[asked]
                values = [value for value, kept in zip(values, asked) if kept]
                if not len(positions):
                    return
                if sink is not None:
                    sink(scored_batch(node, rows[positions], values))
                if on_scored is not None:
                    on_scored(rows[positions], values)

            extra["on_answers"] = on_answers
        batch = (
            self.reranker.score(spec, rows, self.documents, **extra)
            if len(rows) else RerankerBatch(empty, 0, 0)
        )
        asked = ~skipped
        rows, values = rows[asked], np.asarray(batch.scores)[asked]
        if sink is not None and len(rows) and not streams:
            sink(scored_batch(node, rows, values))
        scored = score_rows(rows, spec.aliases, spec.name, values,
                           _value_type(spec))
        if isinstance(spec, ClassifySpec) and spec.probabilities:
            scored = scored.with_column(
                spec.name + PROBABILITIES_SUFFIX, probability_column(
                    spec.labels,
                    np.full((len(rows), len(spec.labels)), np.nan)
                    if batch.probabilities is None
                    else np.asarray(batch.probabilities)[asked]))
        count = len(rows)
        return NodeResult(
            {node.rows_port: scored},
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
