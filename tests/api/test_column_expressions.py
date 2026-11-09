"""A struct field or a function of a column stands wherever a column does."""


import pyarrow as pa
import pytest
from helpers.planning import operators_of
from helpers.sessions import CONFIG, TOKENIZER, _run, encoder, make_executor

import quail


@pytest.fixture
def session():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": ["d0", "d1", "d2"],
            "details": [{"body": "one", "url": "u1", "kind": "a"},
                        {"body": "two words", "url": "u2", "kind": "b"},
                        {"body": "three words here", "url": "u3", "kind": "a"}],
        }), id_col="id"))
        session.register("partners", quail.DocumentProvider.from_table(pa.table({
            "id": ["p0", "p1", "p2"],
            "text": ["partner text"] * 3,
            "url": ["u2", "u3", "u9"],
        }), id_col="id"))
        yield session


def test_a_prompt_reads_a_struct_field_as_the_document(session):
    query = session.sql(
        "SELECT r.id FROM docs r WHERE AI_FILTER(PROMPT('q: {0}', r.details.body))")
    (scan,) = operators_of(query).scans
    assert str(scan.column) == "r.details.body"
    result = _run(query, make_executor({"r": {"q:": [1, 0, 1]}}))
    assert result.to_rows() == [("d0",), ("d2",)]
    # the field's text was tokenized, not the struct's
    bodies = ("one", "two words", "three words here")
    assert list(query.token_inputs()["r"].lengths) == [
        len(encoder()(text)) for text in bodies]
    # DataFusion's plan of the region reads the field
    assert 'get_field(r.details, Utf8("body"))' in query.explain()


FILTERED = "FROM docs r WHERE AI_FILTER(PROMPT('q: {0}', r.details.body))"


def test_select_order_and_group_take_expressions(session):
    executor = make_executor({"r": {"q:": [1, 1, 1]}})
    rows = _run(session.sql(
        f"SELECT r.id, r.details.body, r.details.kind {FILTERED} "
        "ORDER BY LENGTH(r.details.body) DESC"), executor)
    assert rows.to_rows() == [("d2", "three words here", "a"),
                              ("d1", "two words", "b"), ("d0", "one", "a")]
    assert rows.columns == ["r.id", "r.details.body", "r.details.kind"]
    grouped = session.sql(
        f"SELECT r.details.kind, COUNT(*) AS n {FILTERED} "
        "GROUP BY r.details.kind ORDER BY r.details.kind")
    assert _run(grouped, executor).to_rows() == [("a", 2), ("b", 1)]
    distinct = session.sql(
        f"SELECT DISTINCT r.details.kind {FILTERED} ORDER BY 1")
    assert _run(distinct, executor).to_rows() == [("a",), ("b",)]
