"""Score one engine's run of one query against the saved labels.

The engine's runner hands over a `RunOutput`: every predicate answer it
produced, keyed by the query's operator IDs, and the final rows.
Everything here is in terms of the benchmark's own ids, so no engine
object is needed to score a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.compute as pc

from quail_b.data import _ids
from quail_b.queries import QuerySpec


@dataclass
class RunOutput:
    """What one engine returned for one query.

    Attributes:
        filter_answers: Filter operator ID to a table with the relation's
            alias column and a boolean `answer` column, one row per document
            the engine asked about.
        join_answers: Join operator ID to a table with one ID column
            per joined relation and a boolean `answer` column, one row per
            evaluated tuple.
        rows: The final rows, one ID column per selected alias.
        runtime_s: Completed query execution time, excluding result collection.
        measurements: Engine-reported numbers. `fresh_tokens` is the
            count of input token positions a model forward pass processed
            instead of reading from existing KV; it is required when
            `prompt_pieces` is set. Other values such as startup duration
            stay optional.
        prompt_pieces: The prompt token ids around each document, as
            `quail_b.minimum.validate_prompt_pieces` describes, or None.
            With the answers and `fresh_tokens`, scoring fills
            `minimum_tokens` and `regret_tokens`.
    """

    filter_answers: dict[str, pa.Table] | None
    join_answers: dict[str, pa.Table] | None
    rows: pa.Table
    runtime_s: float | None = None
    measurements: dict = field(default_factory=dict)
    prompt_pieces: dict | None = None


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


def corpus_ids(spec: QuerySpec, corpus_rows) -> dict:
    """Per alias, the distinct corpus ids as strings, in one fixed order."""
    return {relation.alias: pc.unique(pa.array(
        [str(row_id) for row_id in _ids(corpus_rows[relation.table])],
        type=pa.string()))
        for relation in spec._info.relations}


def encode_ids(table: pa.Table, aliases, references: dict) -> pa.Table:
    """Replace each id column by its position in the alias's corpus ids.

    An id not in the corpus becomes null. The corpus ids are cast to
    the column's type when that works (integer ids stored as integers),
    so a result of hundreds of millions of rows is never cast to
    strings; only the small corpus side is.
    """
    columns = {}
    for alias in aliases:
        column = table.column(alias)
        reference = references[alias]
        if column.type != reference.type:
            try:
                reference = reference.cast(column.type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                column = column.cast(pa.string())
        columns[alias] = pc.index_in(column, value_set=reference)
    return pa.table(columns)


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


def _column_by_id(rows, column: str, ids) -> pa.Array:
    """Return a corpus column's value for each id, null for an unknown id."""
    if isinstance(rows, pa.Table):
        values = rows.column(column)
    else:
        values = pa.array([row[column] for row in rows])
    keys = pa.array([str(row_id) for row_id in _ids(rows)], pa.string())
    return pc.take(values, pc.index_in(ids, value_set=keys))


def _apply_conditions(table: pa.Table, join, spec: QuerySpec,
                      corpus_rows) -> pa.Table:
    """Keep the pairs of string ids that satisfy the join's equalities.

    corpus_rows is needed only when the join has equality conditions.
    """
    if not join.on:
        return table
    if corpus_rows is None:
        raise ValueError(
            f"{spec.id}: the join on {join.on} needs the corpus rows to "
            f"apply its equality conditions")
    left, right = join.relations
    left_rows = corpus_rows[spec._info.relation(left).table]
    right_rows = corpus_rows[spec._info.relation(right).table]
    mask = pa.array([True] * table.num_rows, pa.bool_())
    for left_column, right_column in join.on:
        equal = pc.equal(
            _column_by_id(left_rows, left_column, table.column(left)),
            _column_by_id(right_rows, right_column, table.column(right)))
        mask = pc.and_(mask, pc.fill_null(equal, False))
    return table.filter(mask)


def answer_relations(spec: QuerySpec, filter_answers, join_answers,
                     corpus_rows=None):
    """Return (survivors, relations) an engine's own answers imply.

    survivors maps each alias to the string ids that passed every filter
    on it, or None when it has no filter. relations holds one table per
    join, in written order, with the pairs that answered TRUE, satisfy
    the join's equality conditions, and survived every filter.
    corpus_rows is needed only when a join has equality conditions.
    Returns None when the answers are missing.
    """
    if join_answers is None or filter_answers is None:
        return None
    survivors = {
        relation.alias: None for relation in spec._info.relations
    }
    for filter_spec in spec._info.filters:
        table = filter_answers.get(filter_spec.id)
        if table is None:
            return None
        alias = filter_spec.relation
        passed = {
            str(row_id)
            for row_id, answer in zip(table.column(alias).to_pylist(),
                                      table.column("answer").to_pylist())
            if answer
        }
        kept = survivors[alias]
        survivors[alias] = passed if kept is None else kept & passed
    relations = []
    for join in spec._info.joins:
        table = join_answers.get(join.id)
        if table is None:
            return None
        mask = table.column("answer")
        true_pairs = _as_string_ids(table.filter(mask), list(join.relations))
        true_pairs = _apply_conditions(true_pairs, join, spec, corpus_rows)
        for alias in join.relations:
            if survivors[alias] is not None:
                true_pairs = true_pairs.filter(pc.is_in(
                    true_pairs.column(alias),
                    value_set=pa.array(sorted(survivors[alias]), pa.string())))
        relations.append(true_pairs)
    return survivors, relations


def rows_from_answers(spec: QuerySpec, filter_answers, join_answers,
                      corpus_rows=None) -> pa.Table:
    """Return the final rows an engine's own answers imply.

    For a run that saved its predicate answers but not its rows: a row
    survives when every filter on its alias answered TRUE, every join
    it takes part in answered TRUE, and every join equality holds.
    corpus_rows is needed only when a join has equality conditions.
    """
    survivors, relations = answer_relations(
        spec, filter_answers, join_answers, corpus_rows)
    if relations:
        rows = _join_all(relations)
    else:
        alias = spec._info.base_alias
        rows = _id_table({alias: sorted(survivors[alias])})
    return _distinct(rows.select(sorted(rows.column_names)))


def implied_row_count(spec: QuerySpec, survivors: dict, relations: list) -> int:
    """Count the rows the answers imply without building them.

    The relations are joined in the order _join_all uses, but each step
    keeps only the aliases a later relation still needs, with a weight
    per row that sums the rows it stands for. Every joined alias is a
    selected column, so the join's size is the distinct row count.
    """
    if not relations:
        return len(survivors[spec._info.base_alias])
    pending = list(relations)
    current = pending.pop(0)
    current = current.append_column(
        "weight", pa.array([1] * current.num_rows, pa.int64()))
    while pending:
        for index, table in enumerate(pending):
            shared = sorted((set(current.column_names) - {"weight"})
                            & set(table.column_names))
            if shared:
                pending.pop(index)
                break
        else:
            raise ValueError("AI join relations form a disconnected graph")
        joined = current.join(table, keys=shared, join_type="inner")
        needed = {alias for later in pending for alias in later.column_names}
        keep = sorted((set(joined.column_names) - {"weight"}) & needed)
        if keep:
            current = joined.group_by(keep).aggregate([("weight", "sum")])
            current = current.rename_columns(keep + ["weight"])
        else:
            current = joined
    return pc.sum(current.column("weight")).as_py() or 0


def implied_rows_mask(rows: pa.Table, survivors: dict, relations: list,
                      spec: QuerySpec) -> pa.ChunkedArray:
    """Per row of string ids, whether the answers imply it."""
    mask = pa.array([True] * rows.num_rows, pa.bool_())
    for alias in rows.column_names:
        if survivors.get(alias) is not None:
            mask = pc.and_(mask, pc.is_in(
                rows.column(alias),
                value_set=pa.array(sorted(survivors[alias]), pa.string())))
    for join, relation in zip(spec._info.joins, relations):
        left, right = join.relations
        pairs = pc.binary_join_element_wise(
            rows.column(left), rows.column(right), "\x1f")
        known = pc.binary_join_element_wise(
            relation.column(left), relation.column(right), "\x1f")
        mask = pc.and_(mask, pc.is_in(pairs, value_set=pc.unique(known)))
    if not relations:
        alias = spec._info.base_alias
        base = pa.array(sorted(survivors[alias]), pa.string())
        mask = pc.and_(mask, pc.is_in(rows.column(alias),
                                      value_set=base))
    return mask


def scores_from_answers(spec: QuerySpec, output: RunOutput, corpus_rows):
    """Return (survivors, relations) when the answers can score the rows.

    That needs every answer the query produces, and the selected columns
    must hold every joined alias, so that a row is a join tuple.
    """
    answers = answer_relations(
        spec, output.filter_answers, output.join_answers, corpus_rows)
    if answers is None:
        return None
    selected = {name.split(".")[0] for name in spec._info.select}
    joined = {
        alias for join in spec._info.joins for alias in join.relations
    }
    if not joined <= selected or spec._info.base_alias not in selected:
        return None
    return answers


def reference_answer(ground_truth, template: str, ids: tuple[str, ...]) -> bool:
    """Return the saved label of one prompt template for one or two ids."""
    key = ground_truth.key_for_template(template)
    if len(ids) == 1:
        return ground_truth.answer(key, ids[0])
    if len(ids) == 2:
        return ground_truth.answer(key, ids[0], ids[1])
    raise NotImplementedError(
        "ground truth evaluation supports one or two prompt arguments")


def _labels(ground_truth, template: str):
    """Return the labels of one prompt template as written."""
    return ground_truth.predicates[ground_truth.key_for_template(template)]


def expected_survivors(spec: QuerySpec, ground_truth, corpus_rows
                       ) -> dict[str, list[str]]:
    """Return, per alias, the ids that pass every filter on it."""
    survivors = {}
    for relation in spec._info.relations:
        ids = pa.array(
            [str(row_id) for row_id in _ids(corpus_rows[relation.table])],
            pa.string())
        for filter_spec in spec._info.filters:
            if filter_spec.relation != relation.alias:
                continue
            labels = _labels(ground_truth, filter_spec.prompt)
            unknown = pc.invert(pc.is_in(
                ids, value_set=labels.table.column("left_id")))
            if pc.any(unknown).as_py():
                raise KeyError(
                    f"no ground truth for {labels.key} and "
                    f"{pc.sum(unknown).as_py()} rows of {relation.table}")
            ids = ids.filter(pc.is_in(
                ids, value_set=labels.true_pairs.column("left_id")))
        survivors[relation.alias] = ids.to_pylist()
    return survivors


def expected_rows(spec: QuerySpec, ground_truth, corpus_rows) -> pa.Table:
    """Return the final rows the labels say the query should return."""
    survivors = expected_survivors(spec, ground_truth, corpus_rows)
    relations = []
    for join in spec._info.joins:
        pairs = _labels(ground_truth, join.prompt).true_pairs
        pairs = pairs.select(["left_id", "right_id"]).rename_columns(
            list(join.relations))
        relations.append(_apply_conditions(pairs, join, spec, corpus_rows))
    if relations:
        rows = _join_all(relations)
    else:
        alias = spec._info.base_alias
        rows = _id_table({alias: survivors[alias]})
    for alias in rows.column_names:
        rows = rows.join(
            _id_table({alias: survivors[alias]}),
            keys=[alias], join_type="inner")
    return _distinct(rows.select(sorted(rows.column_names)))


def agreement(table: pa.Table, aliases, labels) -> BinaryCounts:
    """Count an answer table's agreement with the labels, in one join.

    Args:
        table: One id column per alias and a boolean `answer` column.
        aliases: The id columns, in the labels' left then right order.
        labels: The `PredicateLabels` of the predicate.

    Raises:
        KeyError: A row the engine answered has no label.
    """
    aliases = list(aliases)
    answers = pa.table({
        **{alias: pc.cast(table.column(alias), pa.string())
           for alias in aliases},
        "predicted": pc.cast(table.column("answer"), pa.bool_()),
    })
    reference = labels.table.select(
        [*("left_id", "right_id")[:len(aliases)], "answer"]
    ).rename_columns([*aliases, "expected"])
    joined = answers.join(reference, keys=aliases, join_type="left outer")
    expected = joined.column("expected")
    if expected.null_count:
        raise KeyError(
            f"no ground truth for {labels.key} and {expected.null_count} "
            "answered rows")
    predicted = joined.column("predicted")

    def count(mask):
        return pc.sum(mask).as_py() or 0

    counts = BinaryCounts(
        evaluated=joined.num_rows,
        true_positive=count(pc.and_(predicted, expected)),
        true_negative=count(pc.and_(pc.invert(predicted), pc.invert(expected))),
        false_positive=count(pc.and_(predicted, pc.invert(expected))),
        false_negative=count(pc.and_(pc.invert(predicted), expected)),
    )
    counts.correct = counts.true_positive + counts.true_negative
    return counts


def evaluate(spec: QuerySpec, output: RunOutput, ground_truth, corpus_rows) -> dict:
    per_predicate = []
    total = BinaryCounts()
    filters = {
        filter_spec.id: filter_spec for filter_spec in spec._info.filters
    }
    for operator_id, table in (output.filter_answers or {}).items():
        filter_spec = filters[operator_id]
        labels = _labels(ground_truth, filter_spec.prompt)
        item = _PredicateCount(labels.key, "filter", filter_spec.relation)
        item.counts = agreement(table, [filter_spec.relation], labels)
        total.merge(item.counts)
        per_predicate.append(item.as_dict())
    joins = {join.id: join for join in spec._info.joins}
    for operator_id, table in (output.join_answers or {}).items():
        join = joins[operator_id]
        labels = _labels(ground_truth, join.prompt)
        item = _PredicateCount(labels.key, "join")
        item.counts = agreement(table, join.relations, labels)
        total.merge(item.counts)
        per_predicate.append(item.as_dict())

    expected = expected_rows(spec, ground_truth, corpus_rows)
    aliases = [name.split(".")[0] for name in spec._info.select]
    expected = _distinct(expected.select(aliases))
    answers = scores_from_answers(spec, output, corpus_rows)
    if answers is not None:
        # the rows are implied by the answers, which are small: count
        # them and check the expected rows there instead of touching a
        # result that can hold hundreds of millions of rows
        survivors, relations = answers
        predicted_count = implied_row_count(spec, survivors, relations)
        matched_count = pc.sum(implied_rows_mask(
            expected, survivors, relations, spec)).as_py() or 0
    else:
        # an untraced run: score the rows themselves, as small integer
        # codes so that the hashing does not run over strings
        references = corpus_ids(spec, corpus_rows)
        predicted = _distinct(encode_ids(output.rows, aliases, references))
        matched = predicted.join(encode_ids(expected, aliases, references),
                                 keys=aliases, join_type="inner")
        predicted_count, matched_count = predicted.num_rows, matched.num_rows
    input_document_rows = sum(
        len(corpus_rows[relation.table])
        for relation in spec._info.relations)
    unique_documents = {
        (relation.table, str(row_id))
        for relation in spec._info.relations
        for row_id in _ids(corpus_rows[relation.table])
    }
    return {
        "ground_truth_collection_id": ground_truth.collection_id,
        "ground_truth_reference_model": ground_truth.reference_model,
        "answer_accuracy": (total.as_dict() if total.evaluated else None),
        "output_accuracy": _row_metrics(
            predicted_count, expected.num_rows, matched_count),
        "per_predicate": per_predicate,
        "input_document_rows": input_document_rows,
        "unique_input_documents": len(unique_documents),
    }
