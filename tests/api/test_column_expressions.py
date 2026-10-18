"""A struct field or a function of a column stands wherever a column does."""

import json

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, _run, fake_tok, make_executor

import quail
from quail import col, prompt
from quail.frontend.expressions import count
from quail.logical import CompileError, Scan
from quail.physical import Foreign, HashJoin, decode_graph, encode_graph


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


def test_join_keys_and_apply_columns_take_expressions(session):
    seen = {}

    def note_lengths(tables):
        seen.update(tables["r"].to_pydict())
        return tables["r"].column("r")

    query = (session.docs("docs").alias("r")
             .apply(note_lengths, columns=[col("r.details").field("body").length()],
                    ids="preserve")
             .join(session.docs("partners").alias("p"),
                   on=col("r.details").field("url") == col("p.url"))
             .ai_filter(prompt("j: {0} {1}", col("r.details").field("body"),
                               col("p.text")))
             .select("r.id", "p.id"))
    sql = session.sql("""
        SELECT r.id, p.id FROM docs r
        JOIN partners p ON r.details.url = p.url
         AND AI_FILTER(PROMPT('j: {0} {1}', r.details.body, p.text))
    """)
    assert [str(condition) for join in sql.logical.operators().joins
            for condition in join.input.on] == ["r.details.url = p.url"]
    result = _run(query, oracle_executor(
        Oracle(joins=lambda node, assignment: True),
        functions=session.registry.functions))
    assert result.to_rows() == [("d1", "p0"), ("d2", "p1")]
    assert seen == {"r": [0, 1, 2], "length(details.body)": [3, 9, 16]}


def test_a_physical_plan_round_trips_column_expressions(session):
    query = (session.docs("docs").alias("r")
             .apply(lambda tables: tables["r"].column("r"), name="keep",
                    columns=[col("r.details").field("url")], ids="preserve")
             .join(session.docs("partners").alias("p"),
                   on=col("r.details").field("url").length() == col("p.url").length())
             .ai_filter(prompt("j: {0} {1}", col("r.details").field("body"),
                               col("p.text")))
             .select("r.id"))
    graph = query.plan().graph
    codecs = session.registry.codecs
    restored = decode_graph(json.loads(json.dumps(encode_graph(graph, codecs))),
                            codecs)
    for node in graph.nodes:
        if isinstance(node, (HashJoin, Foreign)):
            assert restored.node(node.node_id) == node
    hash_join = graph.nodes_by_type(HashJoin.type_name)[0]
    assert [(str(left), str(right)) for left, right in hash_join.on] == [
        ("length(r.details.url)", "length(p.url)")]


def test_a_bad_field_or_function_fails_before_the_query_runs(session):
    docs = session.docs("docs").alias("r")
    with pytest.raises(CompileError, match="has no field 'title'"):
        docs.ai_filter(prompt("q: {0}", col("r.details").field("title")))
    with pytest.raises(CompileError, match="has no fields"):
        docs.ai_filter(prompt("q: {0}", col("r.id").field("x")))
    with pytest.raises(CompileError, match="length\\(r.details\\) cannot"):
        docs.ai_filter(prompt("q: {0}", col("r.details").length()))
    with pytest.raises(CompileError, match="has no field 'title'"):
        session.sql("SELECT r.id FROM docs r "
                    "WHERE AI_FILTER(PROMPT('q: {0}', r.details.title))")


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


def test_a_column_returns_under_an_as_name(session):
    length = col("r.details").field("body").length()
    built = (session.docs("docs").alias("r")
             .ai_filter(prompt("q: {0}", col("r.details").field("body")))
             .order_by("n").select(col("r.id").named("key"), length.named("n")))
    sql = session.sql("""
        SELECT r.id AS key, LENGTH(r.details.body) AS n FROM docs r
        WHERE AI_FILTER(PROMPT('q: {0}', r.details.body)) ORDER BY n
    """)
    assert sql.logical == built.logical
    assert built.explain().count("length(r.details.body) AS n") == 1
    result = _run(built, make_executor({"r": {"q:": [1, 1, 1]}}))
    assert result.columns == ["key", "n"]
    assert result.to_rows() == [("d0", 3), ("d1", 9), ("d2", 16)]
    with pytest.raises(CompileError, match="unique"):
        session.sql("""
            SELECT r.id AS n, LENGTH(r.details.body) AS n FROM docs r
            WHERE AI_FILTER(PROMPT('q: {0}', r.details.body))
        """)
