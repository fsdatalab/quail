"""SQL that DataFusion runs around Quail's model regions.

Quail runs the operators that call the model and those that change
what the model reads; DataFusion runs the rest of the SQL, so any SQL
DataFusion runs works with AI functions in it.
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish, question_text
from helpers.sessions import CONFIG, TOKENIZER

import quail

AI_IF = "AI_FILTER(PROMPT('q: {0}', r.body))"
# the model keeps every review but r1
ORACLE = Oracle(filters=lambda alias, document, position: document != 1,
                labels=lambda spec, ids: "a")


@pytest.fixture
def session():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register("reviews", quail.DocumentProvider.from_table(pa.table({
            "id": ["r0", "r1", "r2"], "body": ["a", "b", "c"],
            "stars": [5, 2, 4], "pid": ["p0", "p1", "p9"]}), id_col="id"))
        session.register("products", quail.DocumentProvider.from_table(pa.table({
            "id": ["p0", "p1"], "text": ["x", "y"]}), id_col="id"))
        yield session


SQL_AROUND_AI_FUNCTIONS = [
    ("SELECT r.id FROM reviews r WHERE r.stars > 3 ORDER BY r.id",
     [("r0",), ("r2",)]),
    (f"SELECT r.id FROM reviews r WHERE {AI_IF} AND r.stars BETWEEN 2 AND 4",
     [("r2",)]),
    (f"SELECT r.id, p.text FROM reviews r JOIN products p ON r.pid = p.id "
     f"WHERE {AI_IF} ORDER BY r.id", [("r0", "x")]),
    (f"SELECT r.id, p.id FROM reviews r CROSS JOIN products p WHERE {AI_IF} "
     "ORDER BY r.id, p.id",
     [("r0", "p0"), ("r0", "p1"), ("r2", "p0"), ("r2", "p1")]),
    (f"SELECT r.id, p.text FROM reviews r LEFT JOIN products p ON r.pid = p.id "
     f"WHERE {AI_IF} ORDER BY r.id", [("r0", "x"), ("r2", None)]),
    (f"SELECT r.id FROM reviews r WHERE {AI_IF} "
     "AND r.stars > (SELECT AVG(stars) FROM reviews) ORDER BY r.id",
     [("r0",), ("r2",)]),
    (f"SELECT r.id FROM reviews r WHERE {AI_IF} INTERSECT "
     "SELECT r.id FROM reviews r WHERE r.stars > 3 ORDER BY 1",
     [("r0",), ("r2",)]),
    ("SELECT r.id, CASE WHEN AI.CLASSIFY(PROMPT('t: {0}', r.body), ['a', 'b']) "
     "= 'a' THEN 1 ELSE 0 END AS f FROM reviews r ORDER BY r.id",
     [("r0", 1), ("r1", 1), ("r2", 1)]),
    # DataFusion's optimizer replaces LIMIT 0 with no rows; no model runs
    (f"SELECT r.id FROM reviews r WHERE {AI_IF} LIMIT 0", []),
]


def test_sql_around_ai_functions_returns_what_datafusion_computes(session):
    for sql, rows in SQL_AROUND_AI_FUNCTIONS:
        query = session.sql(sql)
        # a query with no model call has no region for the oracle to answer
        result = finish(query, ORACLE) if hasattr(query, "logical") else query.run()
        table = result.collect()
        assert [tuple(row.values()) for row in table.to_pylist()] == rows, sql


def test_an_or_of_two_model_calls_and_a_column_test_returns_the_rows_any_accepts(
        session):
    query = session.sql(f"SELECT r.id FROM reviews r WHERE {AI_IF} OR r.stars > 3 "
                        f"OR {AI_IF.replace('q:', 'p:')} ORDER BY r.id")
    rows = finish(query, ORACLE).collect().to_pylist()
    # r1 has two stars and the model rejects it
    assert [tuple(row.values()) for row in rows] == [("r0",), ("r2",)]


def test_an_or_of_a_score_comparison_runs_in_datafusion(session):
    score = "AI.SCORE(PROMPT('s: {0}', r.body)) > 0.5"
    query = session.sql(
        f"SELECT r.id FROM reviews r WHERE {score} OR r.stars > 3 ORDER BY r.id")
    rows = finish(query, ORACLE).collect().to_pylist()
    assert [tuple(row.values()) for row in rows] == [("r0",), ("r2",)]


def test_a_limit_above_a_datafusion_projection_stops_at_its_rows(session):
    # DataFusion runs the CASE, and pushes the LIMIT into the region below
    query = session.sql(
        "SELECT r.id, CASE WHEN AI.CLASSIFY(PROMPT('t: {0}', r.body), ['a', 'b']) "
        "= 'a' THEN 1 ELSE 0 END AS f FROM reviews r LIMIT 2")
    assert finish(query, ORACLE).collect().num_rows == 2


def test_a_region_returns_a_document_column_beside_an_ai_answer(session):
    query = session.sql(
        "SELECT r.id, r.stars, AI.CLASSIFY(PROMPT('t: {0}', r.body), ['a', 'b']) "
        "AS kind FROM reviews r")
    assert "Projection: r.id, r.stars, kind" in query.logical
    assert "HashJoinExec" not in query.explain()
    rows = finish(query, ORACLE).collect()
    assert rows.sort_by("r.id").to_pylist() == [
        {"r.id": "r0", "r.stars": 5, "kind": "a"},
        {"r.id": "r1", "r.stars": 2, "kind": "a"},
        {"r.id": "r2", "r.stars": 4, "kind": "a"}]
    # a label read beside a table's column is DataFusion's string
    assert rows.schema.field("kind").type == pa.string()


def test_an_ai_join_returns_each_matching_pair_once_with_both_tables_columns(
        session, monkeypatch):
    monkeypatch.setenv("DATAFUSION_OPTIMIZER_ENABLE_DYNAMIC_FILTER_PUSHDOWN", "true")
    query = session.sql(
        "SELECT r.id, r.stars, p.id, p.text FROM reviews r JOIN products p "
        "ON AI_FILTER(PROMPT('m {0} {1}', r.body, p.text)) ORDER BY r.id, p.id")
    oracle = Oracle(joins=lambda node, assignment: assignment["r"] != 1)
    rows = finish(query, oracle).to_rows()
    assert rows == [("r0", 5, "p0", "x"), ("r0", 5, "p1", "y"),
                    ("r2", 4, "p0", "x"), ("r2", 4, "p1", "y")]


OUTER_JOINS = [(join_type, accepted) for join_type in ("LEFT", "RIGHT", "FULL")
               for accepted in (True, False)]


def _keyed_join_rows(join_type, accepted):
    """Return the rows of reviews joined to products on `pid` and one answer."""
    expected = [("r0", "p0"), ("r1", "p1")] if accepted else []
    if join_type in ("LEFT", "FULL"):
        expected.extend([("r2", None)] if accepted else
                        [("r0", None), ("r1", None), ("r2", None)])
    if join_type in ("RIGHT", "FULL") and not accepted:
        expected.extend([(None, "p0"), (None, "p1")])
    return sorted(expected, key=repr)


def test_outer_ai_joins_preserve_unmatched_rows_after_the_complete_on_predicate(
        session):
    for join_type, accepted in OUTER_JOINS:
        query = session.sql(
            f"SELECT r.id, p.id FROM reviews r {join_type} JOIN products p "
            "ON r.pid = p.id AND AI.IF(PROMPT('m {0} {1}', r.body, p.text))")
        oracle = Oracle(joins=lambda *_, accepted=accepted: accepted,
                        filters=lambda *_, accepted=accepted: accepted)
        rows = finish(query, oracle).to_rows()
        assert sorted(rows, key=repr) == _keyed_join_rows(join_type, accepted), (
            join_type, accepted)


class ConjunctiveOracle(Oracle):
    """An oracle that answers the 'first' and 'second' questions separately."""

    def __init__(self, first, second):
        """Plant one answer for each of the two questions."""
        super().__init__()
        self.first, self.second = first, second

    def _answer(self, question):
        assert ("first" in question) != ("second" in question)
        return self.first if "first" in question else self.second

    def _filter_answer(self, node, document):
        return self._answer(question_text(node.question_token_ids))

    def _join_answer(self, node, anchor_id, partner_ids):
        return self._answer(
            question_text([*node.frame_token_ids, *node.tail_token_ids]))


def test_a_join_requires_every_conjunctive_ai_on_predicate(session):
    for join_type in ("INNER", "LEFT", "RIGHT", "FULL"):
        query = session.sql(
            f"SELECT r.id, p.id FROM reviews r {join_type} JOIN products p "
            "ON r.pid = p.id "
            "AND AI.IF(PROMPT('first {0} {1}', r.body, p.text)) "
            "AND AI.IF(PROMPT('second {0} {1}', r.body, p.text))")
        for first in (True, False):
            for second in (True, False):
                rows = finish(query, ConjunctiveOracle(first, second)).to_rows()
                expected = _keyed_join_rows(join_type, first and second)
                assert sorted(rows, key=repr) == expected, (
                    join_type, first, second)


def test_a_full_ai_join_does_not_match_null_on_keys(session):
    query = session.sql(
        "SELECT r.id, p.id FROM reviews r FULL JOIN products p "
        "ON NULLIF(r.pid, 'p0') = p.id "
        "AND AI.IF(PROMPT('m {0} {1}', r.body, p.text))")
    oracle = Oracle(joins=lambda node, assignment: True,
                    filters=lambda alias, document, position: True)
    rows = finish(query, oracle).to_rows()
    expected = [("r0", None), ("r1", "p1"), ("r2", None), (None, "p0")]
    assert sorted(rows, key=repr) == sorted(expected, key=repr)


def test_outer_ai_joins_preserve_duplicate_input_rows_and_null_columns(session):
    for join_type, accepted in OUTER_JOINS:
        query = session.sql(
            "SELECT l.body, r.body, l.tag, r.tag FROM "
            "(SELECT 'same' AS body, CAST(NULL AS VARCHAR) AS tag FROM reviews) l "
            f"{join_type} JOIN "
            "(SELECT 'partner' AS body, CAST(NULL AS VARCHAR) AS tag "
            "FROM products) r "
            "ON AI.IF(PROMPT('m {0} {1}', l.body, r.body))")
        oracle = Oracle(joins=lambda *_, accepted=accepted: accepted,
                        filters=lambda *_, accepted=accepted: accepted)
        rows = finish(query, oracle).to_rows()
        expected = [("same", "partner", None, None)] * 6 if accepted else []
        if not accepted and join_type in ("LEFT", "FULL"):
            expected.extend([("same", None, None, None)] * 3)
        if not accepted and join_type in ("RIGHT", "FULL"):
            expected.extend([(None, "partner", None, None)] * 2)
        assert sorted(rows, key=repr) == sorted(expected, key=repr), (
            join_type, accepted)


def test_outer_ai_on_keeps_duplicate_keys_but_does_not_match_null_keys(session):
    # Neither side exposes its original unique id: matching and unmatched
    # duplicates must retain their multiplicity rather than be deduplicated.
    for join_type, accepted in OUTER_JOINS:
        query = session.sql(
            "SELECT l.body, r.body, l.key, r.key FROM "
            "(SELECT 'same' AS body, CASE WHEN id = 'r2' THEN NULL ELSE 'key' END "
            "AS key FROM reviews) l "
            f"{join_type} JOIN "
            "(SELECT 'partner' AS body, CASE WHEN id = 'p1' THEN NULL ELSE 'key' "
            "END AS key FROM products) r "
            "ON l.key = r.key AND AI.IF(PROMPT('m {0} {1}', l.body, r.body))")
        oracle = Oracle(joins=lambda *_, accepted=accepted: accepted,
                        filters=lambda *_, accepted=accepted: accepted)
        rows = finish(query, oracle).to_rows()
        expected = [("same", "partner", "key", "key")] * 2 if accepted else []
        if join_type in ("LEFT", "FULL"):
            expected.append(("same", None, None, None))
            if not accepted:
                expected.extend([("same", None, "key", None)] * 2)
        if join_type in ("RIGHT", "FULL"):
            expected.append((None, "partner", None, None))
            if not accepted:
                expected.append((None, "partner", None, "key"))
        assert sorted(rows, key=repr) == sorted(expected, key=repr), (
            join_type, accepted)


def test_a_field_of_an_extracted_struct_is_selected_beside_a_joined_column(
        session):
    query = session.sql(
        "SELECT x.id, x.stars, x.found.who AS who FROM (SELECT r.id, r.stars, "
        "AI.EXTRACT(r.body, STRUCT('Who wrote it?' AS who)) AS found "
        "FROM reviews r) x "
        "ORDER BY x.id", dialect="bq")
    # an extraction copies its answer from the document
    oracle = Oracle(answers=lambda spec, document: "abc"[document])
    rows = finish(query, oracle).to_rows()
    assert rows == [("r0", 5, "a"), ("r1", 2, "b"), ("r2", 4, "c")]
