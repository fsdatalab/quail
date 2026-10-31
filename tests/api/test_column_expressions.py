"""A struct field or a function of a column stands wherever a column does."""


import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, _run, fake_tok, make_executor

import quail
from quail import col, prompt
from quail.frontend.expressions import count
from quail.logical import Scan


@pytest.fixture
def session():
    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
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
    built = (session.docs("docs").alias("r")
             .ai_filter(prompt("q: {0}", col("r.details").field("body")))
             .select("r.id"))
    sql = session.sql(
        "SELECT r.id FROM docs r WHERE AI_FILTER(PROMPT('q: {0}', r.details.body))")
    assert sql.logical == built.logical
    scan = next(node for node in built.logical.walk() if isinstance(node, Scan))
    assert str(scan.column) == "r.details.body"
    result = _run(built, make_executor({"r": {"q:": [1, 0, 1]}}))
    assert result.to_rows() == [("d0",), ("d2",)]
    # the field's text was tokenized, not the struct's
    assert list(session.inputs.token_lengths("docs", scan.column)) == [1, 2, 3]
    assert "details.body" in built.explain()


def test_select_order_and_group_take_expressions(session):
    body = col("r.details").field("body")
    kind = col("r.details").field("kind")

    def filtered():
        return session.docs("docs").alias("r").ai_filter(prompt("q: {0}", body))

    executor = make_executor({"r": {"q:": [1, 1, 1]}})
    rows = _run(filtered().order_by(body.length().desc())
                .select("r.id", body, kind), executor)
    assert rows.to_rows() == [("d2", "three words here", "a"),
                              ("d1", "two words", "b"), ("d0", "one", "a")]
    assert rows.columns == ["r.id", "r.details.body", "r.details.kind"]
    grouped = (filtered().group_by(kind).agg(n=count()).order_by(kind)
               .select(kind, "n"))
    sql = session.sql("""
        SELECT r.details.kind, COUNT(*) AS n FROM docs r
        WHERE AI_FILTER(PROMPT('q: {0}', r.details.body))
        GROUP BY r.details.kind ORDER BY r.details.kind
    """)
    assert sql.logical == grouped.logical
    assert _run(grouped, executor).to_rows() == [("a", 2), ("b", 1)]
    distinct = filtered().distinct().order_by(kind).select(kind)
    assert _run(distinct, executor).to_rows() == [("a",), ("b",)]
