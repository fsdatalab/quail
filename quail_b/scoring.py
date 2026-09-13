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


def _values_by_id(rows, column: str) -> dict[str, object]:
    """Map each document id to its value in one corpus column."""
    if isinstance(rows, pa.Table):
        values = rows.column(column).to_pylist()
    else:
        values = [row[column] for row in rows]
    return {str(row_id): value for row_id, value in zip(_ids(rows), values)}


def _allowed_pairs(join, spec: QuerySpec, corpus_rows):
    """Return pair -> allowed for a join's equality conditions, or None."""
    if not join.on:
        return None
    if corpus_rows is None:
        raise ValueError(
            f"{spec.id}: the join on {join.on} needs the corpus rows to "
            f"apply its equality conditions")
    left, right = join.relations
    left_rows = corpus_rows[spec._info.relation(left).table]
    right_rows = corpus_rows[spec._info.relation(right).table]
    columns = [(_values_by_id(left_rows, left_column),
                _values_by_id(right_rows, right_column))
               for left_column, right_column in join.on]

    def allowed(left_id: str, right_id: str) -> bool:
        return all(
            left_values.get(left_id) is not None
            and left_values.get(left_id) == right_values.get(right_id)
            for left_values, right_values in columns)

    return allowed


def _apply_conditions(table: pa.Table, join, allowed) -> pa.Table:
    if allowed is None:
        return table
    left, right = join.relations
    mask = [allowed(left_id, right_id) for left_id, right_id in zip(
        table.column(left).to_pylist(), table.column(right).to_pylist())]
    return table.filter(pa.array(mask, type=pa.bool_()))


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
        true_pairs = _apply_conditions(
            true_pairs, join, _allowed_pairs(join, spec, corpus_rows))
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


def expected_survivors(spec: QuerySpec, ground_truth, corpus_rows
                       ) -> dict[str, list[str]]:
    """Return, per alias, the ids that pass every filter on it."""
    survivors = {}
    for relation in spec._info.relations:
        prompts = [
            filter_spec.prompt
            for filter_spec in spec._info.filters
            if filter_spec.relation == relation.alias
        ]
        survivors[relation.alias] = [
            str(row_id)
            for row_id in _ids(corpus_rows[relation.table])
            if all(reference_answer(ground_truth, template, (str(row_id),))
                   for template in prompts)
        ]
    return survivors


def expected_rows(spec: QuerySpec, ground_truth, corpus_rows) -> pa.Table:
    """Return the final rows the labels say the query should return."""
    survivors = expected_survivors(spec, ground_truth, corpus_rows)
    relations = []
    for join in spec._info.joins:
        key = ground_truth.key_for_template(join.prompt)
        labels = ground_truth.predicates[key]
        left, right = join.relations
        allowed = _allowed_pairs(join, spec, corpus_rows)
        columns = {left: [], right: []}
        for (left_id, right_id), answer in labels.answers.items():
            if answer and (allowed is None or allowed(left_id, right_id)):
                columns[left].append(left_id)
                columns[right].append(right_id)
        relations.append(_id_table(columns))
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


def evaluate(spec: QuerySpec, output: RunOutput, ground_truth, corpus_rows) -> dict:
    per_predicate = []
    total = BinaryCounts()
    filters = {
        filter_spec.id: filter_spec for filter_spec in spec._info.filters
    }
    for operator_id, table in (output.filter_answers or {}).items():
        filter_spec = filters[operator_id]
        key = ground_truth.key_for_template(filter_spec.prompt)
        item = _PredicateCount(key, "filter", filter_spec.relation)
        alias = filter_spec.relation
        ids = table.column(alias).to_pylist()
        answers = table.column("answer").to_pylist()
        for row_id, predicted in zip(ids, answers):
            item.counts.add(
                bool(predicted),
                reference_answer(
                    ground_truth, filter_spec.prompt, (str(row_id),)
                ))
        total.merge(item.counts)
        per_predicate.append(item.as_dict())
    joins = {join.id: join for join in spec._info.joins}
    for operator_id, table in (output.join_answers or {}).items():
        join = joins[operator_id]
        key = ground_truth.key_for_template(join.prompt)
        item = _PredicateCount(key, "join")
        columns = [table.column(alias).to_pylist()
                   for alias in join.relations]
        answers = table.column("answer").to_pylist()
        for row_index, predicted in enumerate(answers):
            ids = tuple(str(column[row_index]) for column in columns)
            item.counts.add(
                bool(predicted), reference_answer(ground_truth, join.prompt, ids))
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
