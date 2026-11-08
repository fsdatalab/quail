"""quail.SessionContext: DataFusion binds the query, Quail runs each model region."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER, check_table

import quail
from quail.api.engine import Engine

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


def _same_rows(table: pa.Table, expected: pa.Table, ordered: bool) -> bool:
    """Whether two tables hold the same rows, in the same order if `ordered`."""
    if ordered:
        return _rows(table) == _rows(expected)
    return sorted(_rows(table), key=repr) == sorted(_rows(expected), key=repr)


@pytest.mark.parametrize("name", QUERIES)
def test_a_region_answers_as_the_written_query(context, name):
    text = QUERIES[name]
    session = context.session
    expected = finish(session.sql(text), ORACLE).collect()
    table = context.sql(text)
    assert table.column_names == expected.column_names
    assert _same_rows(table, expected, ordered="ORDER BY" in text)


@pytest.mark.parametrize("name", QUERIES)
def test_rust_derives_each_region_as_session_sql_compiles_it(context, name):
    text = QUERIES[name]
    (region,) = context.region_plans(text)
    query = context.session.sql(text)
    assert (region["operators"], region["logical"]) == (query.operators, query.logical)


def test_operators_outside_a_region_run_in_datafusion(context):
    text = ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) "
            "UNION ALL SELECT a.id FROM aspects a ORDER BY id")
    assert len(context.region_plans(text)) == 1
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
    # two model regions
    two = context.session.sql(
        "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) "
        "UNION ALL SELECT r.id FROM reviews r "
        "WHERE AI_FILTER(PROMPT('q2: {0}', r.body)) ORDER BY id")
    assert two.num_regions == 2
    assert len(two.plan()) == 2


def test_an_extraction_returns_and_filters_on_its_answer_positions(context):
    text = ("SELECT c.id, AI.EXTRACT(c.text, {'end': 'q6: When does it end?'}) "
            "AS lease FROM contracts c WHERE lease_span.end.start < 15")
    expected = finish(context.session.sql(text), ORACLE).collect()
    table = context.sql(text)
    assert table.column_names == ["c.id", "lease", "lease_span"]
    assert _same_rows(table, expected, ordered=False)
    assert len(_rows(table)) == 2


def test_every_check_query_derives_one_region_as_session_sql_compiles_it():
    from tools.dump_check_queries import check_query_entries

    for entry in check_query_entries():
        session = quail.Session(CONFIG, tokenizer=TOKENIZER)
        context = quail.SessionContext(session)
        for name, columns in entry["tables"].items():
            context.register(name, quail.DocumentProvider.from_table(
                check_table(columns), id_col="id"))
        (region,) = context.region_plans(entry["sql"])
        query = session.sql(entry["sql"])
        assert (region["operators"], region["logical"]) == (
            query.operators, query.logical), entry["id"]
        session.close()


def test_a_region_hands_on_rows_before_it_finishes():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    oracle = Oracle(scores=lambda spec, ids: ids[0] / 10, streams=True)
    events = []
    executor = oracle_executor(oracle)

    def run(request):
        response = executor(request)
        events.append("finished")
        return response

    context = quail.SessionContext(session, physical_executor=run)
    context.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    text = "SELECT r.id, AI.SCORE(PROMPT('q7: {0}', r.body)) AS score FROM reviews r"
    ((_, planned),) = context._native.plans(text)
    batches = []

    def on_rows(table):
        events.append("rows")
        batches.append(table)

    assert Engine(session, run).execute(planned, {}, on_rows) is None
    # one batch per scored row, each before the region finished
    assert events == ["rows"] * 4 + ["finished"]
    streamed = pa.concat_tables(batches)
    assert _same_rows(streamed, context.sql(text), ordered=False)
    session.close()


def test_a_large_input_runs_in_windows_with_the_same_rows():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    calls = []
    executor = oracle_executor(ORACLE)

    def run(request):
        calls.append(request)
        return executor(request)

    context = quail.SessionContext(session, physical_executor=run)
    context.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    text = ("SELECT r.id, AI.CLASSIFY(PROMPT('q5: {0}', r.body), "
            "['positive', 'negative', 'mixed']) AS sentiment "
            "FROM reviews r WHERE r.stars > 1")
    whole = context.sql(text)
    assert len(calls) == 1
    # each window holds at least one row of the three DataFusion keeps
    context._engine.input_window_rows = 1
    windowed = context.sql(text)
    assert len(calls) > 2
    assert _same_rows(windowed, whole, ordered=False)
    assert whole.num_rows == 3
    session.close()
