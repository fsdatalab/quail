"""Build score columns and evaluate numeric score comparisons."""

import itertools
import time
from collections.abc import Mapping
from dataclasses import dataclass

import pyarrow as pa
from pyarrow import compute as pc

from quail.backends.base import GpuContext
from quail.execution.result import answer_table
from quail.execution.runner import (
    NodeMetrics,
    NodeResult,
)
from quail.physical import AiScore, ScoreFilter, ValueType


def compare_score(score: float, comparison: str, threshold: float) -> bool:
    """Evaluate one supported AI.SCORE comparison."""
    if comparison == "<":
        return score < threshold
    if comparison == "<=":
        return score <= threshold
    if comparison == ">":
        return score > threshold
    if comparison == ">=":
        return score >= threshold
    raise ValueError(f"unsupported AI.SCORE comparison {comparison!r}")


@dataclass(frozen=True)
class RerankerBatch:
    """Scores and token counts returned by one reranker call."""

    scores: tuple[float, ...]
    fresh_tokens: int
    cached_tokens: int


def _score_table(rows, aliases, name, scores) -> pa.Table:
    arrays = {
        alias: pa.array(
            (row[index] for row in rows),
            type=pa.int32(),
        )
        for index, alias in enumerate(aliases)
    }
    arrays[name] = pa.array(scores, type=pa.float64())
    return pa.table(arrays).replace_schema_metadata({
        b"quail.kind": b"score_rows",
        b"quail.aliases": ",".join(aliases).encode("utf-8"),
    })


def _ids(value, alias: str) -> list[int]:
    if isinstance(value, pa.Table):
        return [int(item) for item in value.column(alias).to_pylist()]
    return [int(item) for item in value]


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
            rows = [(document,) for document in by_alias[alias]]
            return rows, prior

        left, right = spec.aliases
        allowed_left = set(by_alias[left])
        allowed_right = set(by_alias[right])
        if pairs is None:
            rows = list(itertools.product(by_alias[left], by_alias[right]))
        else:
            rows = [
                (int(a), int(b))
                for a, b in zip(
                    pairs.column(left).to_pylist(),
                    pairs.column(right).to_pylist(),
                )
                if a in allowed_left and b in allowed_right
            ]
        return rows, None

    def execute(
        self,
        node: AiScore,
        inputs: Mapping[str, object],
    ) -> NodeResult:
        if not isinstance(node, AiScore):
            raise TypeError(type(node).__name__)
        if node.spec is None:
            raise ValueError("AI.SCORE needs a score specification")
        rows, prior = self._candidate_rows(node, inputs)
        return self.execute_rows(node, rows, prior)

    def execute_rows(self, node, rows, prior=None):
        """Compute a score for each supplied row or document pair."""
        started = time.perf_counter()
        spec = node.spec
        if len(spec.arguments) != len(spec.aliases):
            raise ValueError(
                "AI.SCORE needs one prompt argument per document relation"
            )

        batch = (
            self.reranker.score(spec, rows, self.documents)
            if rows else RerankerBatch((), 0, 0)
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
            [node.written_pos] * table.num_rows, type=pa.int32()
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
        scores = table.column(node.score_name).to_pylist()
        answers = [
            compare_score(float(score), node.comparison, node.threshold)
            for score in scores
        ]
        mask = pa.array(answers, type=pa.bool_())
        filtered = table.filter(mask)
        if len(node.aliases) == 1:
            answer_name = f"filter_answers:{node.aliases[0]}"
            answer_relation = _filter_answer_table(node, table, answers)
        else:
            left, right = node.aliases
            answer_name = f"join_answers:{node.written_pos}"
            answer_relation = answer_table(
                {
                    left: table.column(left).to_pylist(),
                    right: table.column(right).to_pylist(),
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
