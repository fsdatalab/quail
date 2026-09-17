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


def compare_score(scores, comparison: str, threshold: float):
    """Compare a score array with a scalar threshold."""
    operations = {
        "<": pc.less, "<=": pc.less_equal,
        ">": pc.greater, ">=": pc.greater_equal,
    }
    if comparison not in operations:
        raise ValueError(f"unsupported AI.SCORE comparison {comparison!r}")
    return operations[comparison](scores, threshold)


@dataclass(frozen=True)
class ScoreRows:
    """Document indices with an optional unexpanded Cartesian product."""

    columns: tuple[np.ndarray, ...]
    product: bool = False

    def __len__(self):
        if self.product:
            return len(self.columns[0]) * len(self.columns[1])
        return len(self.columns[0])

    def batches(self, size=DEFAULT_BATCH_ROWS):
        """Yield bounded arrays in input order."""
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


class RerankerModelExecution:
    """Compute score columns without applying their comparisons."""

    def __init__(self, context: GpuContext):
        self.reranker = context.query_settings["reranker"]
        self.documents = context.query_settings["documents"]

    @staticmethod
    def _sources(node, inputs) -> list[tuple[object, object]]:
        return [
            (port, inputs[port.name])
            for port in node.inputs
        ]

    @staticmethod
    def _candidate_rows(node, inputs):
        spec = node.spec
        sources = RerankerModelExecution._sources(node, inputs)
        by_alias = {}
        pairs = None
        prior = None
        for port, value in sources:
            if port.value_type is ValueType.PAIRS:
                pairs = value
                continue
            if port.value_type is ValueType.SCORES:
                prior = value if len(spec.aliases) == 1 else prior
                for alias in spec.aliases:
                    if alias in value.column_names:
                        by_alias[alias] = _ids(value, alias)
                continue
            if port.source.port.startswith("ids:"):
                alias = port.source.port.split(":", 1)[1]
                by_alias[alias] = _ids(value, alias)

        if len(spec.aliases) == 1:
            alias = spec.aliases[0]
            return ScoreRows((by_alias[alias],)), prior

        left, right = spec.aliases
        if pairs is None:
            return ScoreRows((by_alias[left], by_alias[right]), product=True), None
        selected = pairs.filter(pc.and_(
            pc.is_in(pairs.column(left), value_set=pa.array(by_alias[left])),
            pc.is_in(pairs.column(right), value_set=pa.array(by_alias[right])),
        ))
        return ScoreRows((_ids(selected, left), _ids(selected, right))), None

    def execute(
        self,
        node: AiScore,
        inputs: Mapping[str, object],
    ) -> NodeResult:
        if not isinstance(node, AiScore):
            raise TypeError(type(node).__name__)
        if node.spec is None:
            raise ValueError("AI.SCORE needs a score specification")
        started = time.perf_counter()
        rows, prior = self._candidate_rows(node, inputs)
        results = []
        metrics = NodeMetrics()
        offset = 0
        for batch in rows.batches():
            result = self.execute_rows(
                node, batch,
                None if prior is None else prior.slice(offset, len(batch)),
            )
            results.append(result.outputs["scores"])
            metrics += result.metrics
            offset += len(batch)
        return NodeResult({"scores": pa.concat_tables(results)}, replace(
            metrics, wall_s=time.perf_counter() - started,
            extension={"output": node.spec.name, "aliases": list(node.spec.aliases),
                       "input_rows": len(rows)},
        ))

    def execute_rows(self, node, rows, prior=None):
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
        if prior is None:
            table = _score_table(rows, spec.aliases, spec.name, batch.scores)
        else:
            table = prior.append_column(
                spec.name,
                pa.array(batch.scores, type=pa.float64()),
            )
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
