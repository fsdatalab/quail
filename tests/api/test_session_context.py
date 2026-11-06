"""quail.SessionContext: DataFusion binds the query, Quail runs each model region."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER, check_table

import quail
from quail.physical import encode_graph

pytest.importorskip("_quail_native")

REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3"],
    "body": ["good ending " * 5, "slow pacing " * 5, "great cast " * 5,
             "weak ending " * 5],
    "stars": [5, 2, 4, 1],
})
ASPECTS = pa.table({"id": ["a0", "a1"], "aspect": ["ending", "pacing"]})
CONTRACTS = pa.table({
    "id": ["c0", "c1", "c2"],
    "text": ["Lease ends in 2027. " * 4, "Lease ends in 2030. " * 4,
             "No term. " * 4],
})

LABELS = ["positive", "negative", "positive", "mixed"]
ORACLE = Oracle(
    filters={"r": {"q1:": [1, 0, 1, 1], "q2:": [1, 1, 0, 1]},
             "c": {"q3:": [1, 1, 0]}},
    joins=lambda node, assignment: (assignment["r"] + assignment["a"]) % 2 == 0,
    labels=lambda spec, ids: LABELS[ids[0]],
    answers=lambda spec, document: ["2027", "2030", None][document],
)

# Each query has the shape of a check query, named after it.
QUERIES = {
    "IMDB-1 one filter":
        "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
    "IMDB-2 one join":
        "SELECT r.id, a.id FROM reviews r JOIN aspects a "
        "ON AI_FILTER(PROMPT('q4: {0} {1}', r.body, a.aspect))",
    "IMDB-4 two filters, then a join":
        "SELECT r.id, a.id FROM reviews r JOIN aspects a "
        "ON AI_FILTER(PROMPT('q4: {0} {1}', r.body, a.aspect)) "
        "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) "
        "AND AI_FILTER(PROMPT('q2: {0}', r.body))",
    "IMDB-11 classify":
        "SELECT r.id, AI.CLASSIFY(PROMPT('q5: {0}', r.body), "
        "['positive', 'negative', 'mixed']) AS sentiment FROM reviews r",
    "IMDB-15 classify filtered by its alias":
        "SELECT r.id, AI.CLASSIFY(PROMPT('q5: {0}', r.body), "
        "['positive', 'negative', 'mixed']) AS sentiment FROM reviews r "
        "WHERE sentiment IN ('negative', 'mixed')",
    "CUAD-1 extract":
        "SELECT c.id, AI.EXTRACT(c.text, {'end': 'q6: When does it end?'}) "
        "AS lease FROM contracts c",
    "CUAD-15 filter, extract, sort, and limit":
        "SELECT c.id, AI.EXTRACT(c.text, {'end': 'q6: When does it end?'}) "
        "AS lease FROM contracts c WHERE AI_FILTER(PROMPT('q3: {0}', c.text)) "
        "ORDER BY c.id DESC LIMIT 1",
    "CUAD-11 classify, then aggregate":
        "SELECT AI.CLASSIFY(PROMPT('q5: {0}', r.body), "
        "['positive', 'negative', 'mixed']) AS sentiment, COUNT(*) AS reviews "
        "FROM reviews r GROUP BY sentiment ORDER BY reviews DESC, sentiment",
    "SALES-5 relational filter, score, then top 2":
        "SELECT r.id, AI.SCORE(PROMPT('q7: {0}', r.body)) AS score "
        "FROM reviews r WHERE r.stars > 1 ORDER BY score DESC, r.id LIMIT 2",
    "IMDB-24 filter, then maps":
        "SELECT r.id, AI.MAP(r.body, 'q8: Summarize.', 16) AS summary "
        "FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
}


@pytest.fixture()
def context():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    ORACLE.catalog = session.catalog
    value = quail.SessionContext(session, physical_executor=oracle_executor(ORACLE))
    value.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    value.register("aspects", quail.DocumentProvider.from_table(ASPECTS, id_col="id"))
    value.register("contracts",
                   quail.DocumentProvider.from_table(CONTRACTS, id_col="id"))
    yield value
    session.close()


def _rows(table: pa.Table) -> list[tuple]:
    """The table's rows by column position; a join returns two `id` columns."""
    return list(zip(*(column.to_pylist() for column in table.columns)))


@pytest.mark.parametrize("name", QUERIES)
def test_a_region_plans_and_answers_as_the_written_query(context, name):
    text = QUERIES[name]
    session = context.session
    (region,) = context.region_sql(text)
    codecs = session.registry.codecs
    assert (encode_graph(session.sql(region).plan().graph, codecs)
            == encode_graph(session.sql(text).plan().graph, codecs)), region
    expected = finish(session.sql(text), ORACLE).collect()
    table = context.sql(text)
    assert table.column_names == expected.column_names
    assert _rows(table) == _rows(expected)


@pytest.mark.parametrize("name", QUERIES)
def test_rust_binds_each_region_to_the_plan_session_sql_compiles(context, name):
    text = QUERIES[name]
    (plan,) = context.logical_plans(text)
    expected = context.session.sql(text).logical
    assert plan == expected


def test_operators_outside_a_region_run_in_datafusion(context):
    text = ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) "
            "UNION ALL SELECT a.id FROM aspects a ORDER BY id")
    assert len(context.region_sql(text)) == 1
    table = context.sql(text)
    # a union's column belongs to no one table
    assert table.column_names == ["id"]
    assert _rows(table) == [("a0",), ("a1",), ("r0",), ("r2",), ("r3",)]


def test_session_sql_runs_a_query_that_needs_sql_around_its_model_region(context):
    text = ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) "
            "UNION ALL SELECT a.id FROM aspects a ORDER BY id")
    query = context.session.sql(text)
    # one model region: a Query, with DataFusion's union and sort around it
    assert isinstance(query, quail.api.query.Query) and not query.whole
    assert query.plan().graph.nodes
    assert "Union" in query.explain() and "model region:" in query.explain()
    table = finish(query, ORACLE).collect()
    assert _rows(table) == [("a0",), ("a1",), ("r0",), ("r2",), ("r3",)]
    # two model regions: a DataFusionQuery
    two = context.session.sql(
        "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) "
        "UNION ALL SELECT r.id FROM reviews r "
        "WHERE AI_FILTER(PROMPT('q2: {0}', r.body)) ORDER BY id")
    assert isinstance(two, quail.api.context.DataFusionQuery)
    assert len(two.plan()) == 2


def test_an_extraction_returns_and_filters_on_its_answer_positions(context):
    text = ("SELECT c.id, AI.EXTRACT(c.text, {'end': 'q6: When does it end?'}) "
            "AS lease FROM contracts c WHERE lease_span.end.start < 15")
    expected = finish(context.session.sql(text), ORACLE).collect()
    table = context.sql(text)
    assert table.column_names == ["c.id", "lease", "lease_span"]
    assert _rows(table) == _rows(expected)
    assert len(_rows(table)) == 2


def test_every_check_query_plans_the_same_graph_through_datafusion():
    from tools.dump_check_queries import check_query_entries

    for entry in check_query_entries():
        session = quail.Session(CONFIG, tokenizer=TOKENIZER)
        context = quail.SessionContext(session)
        for name, columns in entry["tables"].items():
            context.register(name, quail.DocumentProvider.from_table(
                check_table(columns), id_col="id"))
        (region,) = context.region_sql(entry["sql"])
        codecs = session.registry.codecs
        assert (encode_graph(session.sql(region).plan().graph, codecs)
                == encode_graph(session.sql(entry["sql"]).plan().graph, codecs)
                ), entry["id"]
        session.close()
