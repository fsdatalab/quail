"""Score one engine's run of one query against the saved labels.

The engine's runner hands over a `RunOutput`: every predicate answer it
produced, keyed by the query's written positions, and the final rows.
Everything here is in terms of the benchmark's own ids, so no engine
object is needed to score a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa

from quail_bench.data import _ids
from quail_bench.labels import GroundTruthCollection
from quail_bench.queries import QuerySpec


@dataclass
class RunOutput:
    """What one engine returned for one query.

    Attributes:
        filter_answers: (alias, written position) to a table with the
            alias's id column and a boolean `answer` column, one row per
            document the engine asked about.
        join_answers: written position to a table with one id column
            per joined alias and a boolean `answer` column, one row per
            evaluated tuple.
        rows: The final rows, one id column per alias in the query.
    """

    filter_answers: dict[tuple[str, int], pa.Table]
    join_answers: dict[int, pa.Table]
    rows: pa.Table


@dataclass
class BinaryCounts:
    correct: int = 0
    evaluated: int = 0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def add(self, predicted: bool, expected: bool) -> None:
        self.evaluated += 1
        self.correct += int(predicted == expected)
        if predicted and expected:
            self.true_positive += 1
        elif predicted:
            self.false_positive += 1
        elif expected:
            self.false_negative += 1
        else:
            self.true_negative += 1

    def merge(self, other: "BinaryCounts") -> None:
        for name in ("correct", "evaluated", "true_positive",
                     "true_negative", "false_positive", "false_negative"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict:
        accuracy = self.correct / self.evaluated if self.evaluated else 0.0
        pden = self.true_positive + self.false_positive
        rden = self.true_positive + self.false_negative
        precision = self.true_positive / pden if pden else 0.0
        recall = self.true_positive / rden if rden else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        return {
            "evaluated": self.evaluated,
            "correct": self.correct,
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
        }


@dataclass
class _PredicateCount:
    predicate_key: str
    op: str
    alias: str | None = None
    counts: BinaryCounts = field(default_factory=BinaryCounts)

    def as_dict(self) -> dict:
        return {
            "predicate_key": self.predicate_key,
            "op": self.op,
            "alias": self.alias,
            **self.counts.as_dict(),
        }


def _row_metrics(predicted_count: int, expected_count: int,
                 matched: int) -> dict:
    if not predicted_count and not expected_count:
        precision = recall = f1 = 1.0
    else:
        precision = matched / predicted_count if predicted_count else 0.0
        recall = matched / expected_count if expected_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
    return {
        "predicted_rows": predicted_count,
        "expected_rows": expected_count,
        "matching_rows": matched,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_match": (predicted_count == expected_count == matched),
        "false_positive_rows": predicted_count - matched,
        "false_negative_rows": expected_count - matched,
    }


def _id_table(columns: dict[str, list]) -> pa.Table:
    return pa.table({
        name: pa.array([str(value) for value in values], type=pa.string())
        for name, values in columns.items()
    })


def _distinct(table: pa.Table) -> pa.Table:
    if table.num_rows == 0:
        return table
    return table.group_by(table.column_names).aggregate([])


def _as_string_ids(table: pa.Table, aliases) -> pa.Table:
    return _distinct(pa.table({
        alias: table.column(alias).cast(pa.string()) for alias in aliases
    }))


def _join_all(tables: list[pa.Table]) -> pa.Table:
    """Inner join relations that share alias columns, in any order."""
    pending = list(tables)
    joined = pending.pop(0)
    while pending:
        for index, table in enumerate(pending):
            shared = sorted(set(joined.column_names) & set(table.column_names))
            if shared:
                joined = joined.join(table, keys=shared, join_type="inner")
                pending.pop(index)
                break
        else:
            raise ValueError("AI join relations form a disconnected graph")
    return joined


def rows_from_answers(spec: QuerySpec, filter_answers, join_answers
                      ) -> pa.Table:
    """Return the final rows an engine's own answers imply.

    For a run that saved its predicate answers but not its rows: a row
    survives when every filter on its alias answered TRUE and every
    join it takes part in answered TRUE.
    """
    survivors = {}
    for alias_spec in spec.aliases:
        alias = alias_spec.alias
        kept = None
        for written_pos in range(len(alias_spec.filters)):
            table = filter_answers[(alias, written_pos)]
            passed = {
                str(row_id)
                for row_id, answer in zip(table.column(alias).to_pylist(),
                                          table.column("answer").to_pylist())
                if answer
            }
            kept = passed if kept is None else kept & passed
        survivors[alias] = kept
    relations = []
    for written_pos, join in enumerate(spec.joins):
        table = join_answers[written_pos]
        mask = table.column("answer")
        relations.append(_as_string_ids(
            table.filter(mask), list(join.aliases)))
    if relations:
        rows = _join_all(relations)
    else:
        rows = _id_table({spec.base_alias: sorted(survivors[spec.base_alias])})
    for alias in rows.column_names:
        if survivors[alias] is not None:
            rows = rows.join(
                _id_table({alias: sorted(survivors[alias])}),
                keys=[alias], join_type="inner")
    return _distinct(rows.select(sorted(rows.column_names)))


class Evaluator:
    """Score answers and final rows against one label collection."""

    def __init__(self, ground_truth: GroundTruthCollection,
                 corpus_rows: dict[str, pa.Table | list[dict]]):
        self.ground_truth = ground_truth
        self.corpus_rows = corpus_rows

    def answer(self, template: str, ids: tuple[str, ...]) -> bool:
        """Return the saved label of one prompt template for one or two ids."""
        key = self.ground_truth.key_for_template(template)
        if len(ids) == 1:
            return self.ground_truth.answer(key, ids[0])
        if len(ids) == 2:
            return self.ground_truth.answer(key, ids[0], ids[1])
        raise NotImplementedError(
            "ground truth evaluation supports one or two prompt arguments")

    def expected_survivors(self, spec: QuerySpec) -> dict[str, list[str]]:
        """Return, per alias, the ids that pass every filter on it."""
        survivors = {}
        for alias_spec in spec.aliases:
            survivors[alias_spec.alias] = [
                str(row_id)
                for row_id in _ids(self.corpus_rows[alias_spec.table])
                if all(self.answer(template, (str(row_id),))
                       for template in alias_spec.filters)
            ]
        return survivors

    def expected_rows(self, spec: QuerySpec) -> pa.Table:
        """Return the final rows the labels say the query should return."""
        survivors = self.expected_survivors(spec)
        relations = []
        for join in spec.joins:
            key = self.ground_truth.key_for_template(join.template)
            labels = self.ground_truth.predicates[key]
            left, right = join.aliases
            columns = {left: [], right: []}
            for (left_id, right_id), answer in labels.answers.items():
                if answer:
                    columns[left].append(left_id)
                    columns[right].append(right_id)
            relations.append(_id_table(columns))
        if relations:
            rows = _join_all(relations)
        else:
            rows = _id_table({spec.base_alias: survivors[spec.base_alias]})
        for alias in rows.column_names:
            rows = rows.join(
                _id_table({alias: survivors[alias]}),
                keys=[alias], join_type="inner")
        return _distinct(rows.select(sorted(rows.column_names)))

    def evaluate(self, spec: QuerySpec, output: RunOutput) -> dict:
        per_predicate = []
        total = BinaryCounts()
        for (alias, written_pos), table in output.filter_answers.items():
            template = spec.alias(alias).filters[written_pos]
            key = self.ground_truth.key_for_template(template)
            item = _PredicateCount(key, "filter", alias)
            ids = table.column(alias).to_pylist()
            answers = table.column("answer").to_pylist()
            for row_id, predicted in zip(ids, answers):
                item.counts.add(
                    bool(predicted), self.answer(template, (str(row_id),)))
            total.merge(item.counts)
            per_predicate.append(item.as_dict())
        for written_pos, table in output.join_answers.items():
            join = spec.joins[written_pos]
            key = self.ground_truth.key_for_template(join.template)
            item = _PredicateCount(key, "join")
            columns = [table.column(alias).to_pylist()
                       for alias in join.aliases]
            answers = table.column("answer").to_pylist()
            for row_index, predicted in enumerate(answers):
                ids = tuple(str(column[row_index]) for column in columns)
                item.counts.add(bool(predicted), self.answer(join.template, ids))
            total.merge(item.counts)
            per_predicate.append(item.as_dict())

        expected = self.expected_rows(spec)
        aliases = expected.column_names
        predicted = _as_string_ids(output.rows, aliases)
        matched = predicted.join(expected, keys=aliases, join_type="inner")
        input_document_rows = sum(
            len(self.corpus_rows[alias_spec.table])
            for alias_spec in spec.aliases)
        unique_documents = {
            (alias_spec.table, str(row_id))
            for alias_spec in spec.aliases
            for row_id in _ids(self.corpus_rows[alias_spec.table])
        }
        return {
            "ground_truth_collection_id": self.ground_truth.collection_id,
            "ground_truth_reference_model": self.ground_truth.reference_model,
            "answer_accuracy": total.as_dict(),
            "output_accuracy": _row_metrics(
                predicted.num_rows, expected.num_rows, matched.num_rows),
            "per_predicate": per_predicate,
            "input_document_rows": input_document_rows,
            "unique_input_documents": len(unique_documents),
        }


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def add_query_metrics(row: dict, evaluation: dict,
                      h100_usd_per_hour: float, gpus: int) -> dict:
    wall_s = float(row["wall_s"])
    boot_s = float(row.get("boot_s") or 0.0)
    tokens = int(row["fresh_tokens"])
    gpu_rate = h100_usd_per_hour * gpus
    inference_cost = wall_s * gpu_rate / 3600
    cost_with_boot = (wall_s + boot_s) * gpu_rate / 3600
    input_rows = int(evaluation["input_document_rows"])
    calls = int(evaluation["answer_accuracy"]["evaluated"])
    row.update({
        "runtime_s": wall_s,
        "runtime_with_boot_s": round(wall_s + boot_s, 2),
        "tokens_processed": tokens,
        "tokens_per_second": round(_divide(tokens, wall_s), 2),
        "input_document_rows": input_rows,
        "unique_input_documents": evaluation["unique_input_documents"],
        "documents_per_second": round(_divide(input_rows, wall_s), 4),
        "inference_calls": calls,
        "inference_calls_per_second": round(_divide(calls, wall_s), 4),
        "inference_cost_usd": round(inference_cost, 8),
        "inference_cost_per_token_usd": round(
            _divide(inference_cost, tokens), 12),
        "inference_cost_per_million_tokens_usd": round(
            _divide(inference_cost, tokens) * 1_000_000, 6),
        "cost_with_boot_usd": round(cost_with_boot, 8),
        "cost_with_boot_per_million_tokens_usd": round(
            _divide(cost_with_boot, tokens) * 1_000_000, 6),
        "accuracy": {
            key: value for key, value in evaluation.items()
            if key not in ("input_document_rows", "unique_input_documents")
        },
    })
    return row


def summarize_queries(rows: list[dict], h100_usd_per_hour: float,
                      gpus: int) -> dict:
    good = [row for row in rows if "error" not in row]
    wall_s = sum(float(row["wall_s"]) for row in good)
    boot_s = sum(float(row.get("boot_s") or 0.0) for row in good)
    tokens = sum(int(row["tokens_processed"]) for row in good)
    input_rows = sum(int(row["input_document_rows"]) for row in good)
    calls = sum(int(row["inference_calls"]) for row in good)
    cost = wall_s * h100_usd_per_hour * gpus / 3600
    cost_with_boot = ((wall_s + boot_s) * h100_usd_per_hour * gpus
                      / 3600)
    counts = BinaryCounts()
    for row in good:
        accuracy = row["accuracy"]["answer_accuracy"]
        counts.merge(BinaryCounts(
            correct=accuracy["correct"],
            evaluated=accuracy["evaluated"],
            true_positive=accuracy["true_positive"],
            true_negative=accuracy["true_negative"],
            false_positive=accuracy["false_positive"],
            false_negative=accuracy["false_negative"],
        ))
    return {
        "queries_completed": len(good),
        "queries_failed": len(rows) - len(good),
        "query_runtime_s": round(wall_s, 2),
        "boot_s": round(boot_s, 2),
        "runtime_with_boot_s": round(wall_s + boot_s, 2),
        "tokens_processed": tokens,
        "tokens_per_second": round(_divide(tokens, wall_s), 2),
        "input_document_rows": input_rows,
        "documents_per_second": round(_divide(input_rows, wall_s), 4),
        "inference_calls": calls,
        "inference_cost_usd": round(cost, 8),
        "inference_cost_per_token_usd": round(
            _divide(cost, tokens), 12),
        "inference_cost_per_million_tokens_usd": round(
            _divide(cost, tokens) * 1_000_000, 6),
        "cost_with_boot_usd": round(cost_with_boot, 8),
        "cost_with_boot_per_million_tokens_usd": round(
            _divide(cost_with_boot, tokens) * 1_000_000, 6),
        "answer_accuracy": counts.as_dict(),
    }
