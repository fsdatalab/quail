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
    "IMDB-15 classify filtered by the same call":
        "SELECT r.id, AI.CLASSIFY(PROMPT('q5: {0}', r.body), "
        "['positive', 'negative', 'mixed']) AS sentiment FROM reviews r "
        "WHERE AI.CLASSIFY(PROMPT('q5: {0}', r.body), "
        "['positive', 'negative', 'mixed']) IN ('negative', 'mixed')",
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


def test_a_region_plans_and_answers_as_the_written_query(context):
    for name, text in QUERIES.items():
        query = context.session.sql(text)
        (region,) = context.region_plans(text)
        assert (region["operators"], region["logical"]) == (
            query.operators, query.logical), name
        expected = finish(query, ORACLE).collect()
        table = context.sql(text)
        assert table.column_names == expected.column_names, name
        assert _same_rows(table, expected, ordered="ORDER BY" in text), name


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
    fields = "c.text, {'end': 'q6: When does it end?'}"
    text = (f"SELECT c.id, AI.EXTRACT({fields}) AS lease, "
            f"AI.EXTRACT_SPAN({fields}) AS lease_span FROM contracts c "
            f"WHERE AI.EXTRACT_SPAN({fields})['end']['start'] < 15")
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


# per operator family: a query of that operator alone, and the oracle
# truth it is answered from; `answered` records each answer
FAMILIES = {
    "filter": ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
               lambda answered: Oracle(filters=lambda alias, document, position:
                                       answered(document % 2 == 0))),
    "classify": (QUERIES["IMDB-11 classify"],
                 lambda answered: Oracle(labels=lambda spec, ids:
                                         answered(LABELS[ids[0]]))),
    "score documents": (
        "SELECT r.id, AI.SCORE(PROMPT('q7: {0}', r.body)) AS score FROM reviews r",
        lambda answered: Oracle(scores=lambda spec, ids: answered(ids[0] / 10))),
    "score pairs": (
        "SELECT r.id, a.id, AI.SCORE(PROMPT('q9: {0} {1}', r.body, a.aspect)) "
        "AS fit FROM reviews r CROSS JOIN aspects a",
        lambda answered: Oracle(scores=lambda spec, ids: answered(sum(ids) / 10))),
    "extract": (QUERIES["CUAD-1 extract"],
                lambda answered: Oracle(answers=lambda spec, document: answered(
                    ["2027", "2030", None][document]))),
    "map": ("SELECT r.id, AI.MAP(r.body, 'q8: Summarize.', 16) AS summary "
            "FROM reviews r",
            lambda answered: Oracle(maps=lambda spec, document: answered(
                f"summary {document}"))),
    "join": (QUERIES["IMDB-2 one join"],
             lambda answered: Oracle(joins=lambda node, assignment: answered(
                 (assignment["r"] + assignment["a"]) % 2 == 0))),
}


def test_each_operator_hands_on_its_first_rows_before_the_region_finishes():
    for family, (text, truth) in FAMILIES.items():
        _check_first_rows_precede_the_last_answer(family, text, truth)


def _check_first_rows_precede_the_last_answer(family, text, truth):
    """Run `text` on a new session and check rows arrive before it finishes."""
    events = []

    def answered(value):
        events.append("answered")
        return value

    oracle = truth(answered)
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    oracle.catalog = session.catalog
    executor = oracle_executor(oracle)

    def run(request):
        response = executor(request)
        events.append("finished")
        return response

    context = quail.SessionContext(session, physical_executor=run)
    for name, table in (("reviews", REVIEWS), ("aspects", ASPECTS),
                        ("contracts", CONTRACTS)):
        context.register(name, quail.DocumentProvider.from_table(table, id_col="id"))
    ((_, planned),) = context._native.plans(text)
    batches = []

    def on_rows(table):
        events.append("rows")
        batches.append(table)

    assert Engine(session, run).execute(planned, {}, on_rows) is None, family
    # the first rows went on before the model gave its last answer
    last_answer = len(events) - 1 - events[::-1].index("answered")
    assert events.index("rows") < last_answer, family
    assert events[-1] == "finished", family
    events.clear()
    streamed = pa.concat_tables(batches)
    assert _same_rows(streamed, context.sql(text), ordered=False), family
    session.close()


def test_a_region_plans_and_runs_once_over_its_whole_input(monkeypatch):
    # one row a batch: DataFusion hands the region its rows in pieces
    monkeypatch.setenv("DATAFUSION_EXECUTION_BATCH_SIZE", "1")
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
    table = context.sql(text)
    assert len(calls) == 1
    assert table.num_rows == 3
    session.close()


def test_a_failure_inside_a_region_names_its_line_in_the_error():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)

    def run(request):
        raise ValueError("the executor failed")

    context = quail.SessionContext(session, physical_executor=run)
    context.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    with pytest.raises(RuntimeError) as failure:
        context.sql("SELECT r.id FROM reviews r "
                    "WHERE AI_FILTER(PROMPT('q1: {0}', r.body))")
    message = str(failure.value)
    assert "ValueError: the executor failed" in message
    # the Python traceback is lost once the error crosses Rust, so the
    # error text carries it
    assert "Traceback (most recent call last)" in message
    assert 'raise ValueError("the executor failed")' in message
    session.close()


def _reviews_context(oracle, rows: int):
    """A context over `rows` reviews r00, r01, ..., answered by `oracle`."""
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    context = quail.SessionContext(session, physical_executor=oracle_executor(oracle))
    reviews = pa.table({
        "id": [f"r{index:02d}" for index in range(rows)],
        "body": [f"review number {index} " * 4 for index in range(rows)],
        "stars": [index % 5 for index in range(rows)],
    })
    context.register("reviews", quail.DocumentProvider.from_table(reviews, id_col="id"))
    return context


def test_a_top_k_on_an_input_column_hands_the_region_its_threshold():
    text = ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) "
            "ORDER BY r.id LIMIT 2")
    # once r00 and r01 pass, the sort holds two rows and tightens
    every = Oracle(filters=lambda alias, document, position: True,
                   tightens_after={1})
    context = _reviews_context(every, 12)
    assert "ModelRegionExec: hint=[DynamicFilter" in context._native.explain(text)
    table = context.sql(text)
    assert _rows(table) == [("r00",), ("r01",)]
    # every later id is past the top two
    assert every.asked == [("r", 0), ("r", 1)]
    context.session.close()


def _scored_rows(oracle) -> int:
    """The rows the oracle scored, over every score call."""
    return sum(len(rows) for _, rows in oracle.calls)


# even reviews score 1.0, the highest a score can be, odd ones 0.3
TOP_K_SCORES = {"scores": lambda spec, ids: 1.0 if ids[0] % 2 == 0 else 0.3}
TOP_K_ROWS = [("r00", 1.0), ("r02", 1.0), ("r04", 1.0)]


def test_a_top_k_over_a_score_asks_fewer_rows_once_it_holds_k(monkeypatch):
    # one partition: DataFusion hands the region the rows in table order
    monkeypatch.setenv("DATAFUSION_EXECUTION_TARGET_PARTITIONS", "1")
    for where in ("", "WHERE r.stars > -1"):
        _check_a_top_k_over_a_score_skips_rows(where)


def _check_a_top_k_over_a_score_skips_rows(where):
    """Run a top 3 by score, with and without its LIMIT, and count scored rows."""
    # with a WHERE the region reads DataFusion's rows; without one it
    # reads the table; either way one run skips rows as the hint tightens
    text = (f"SELECT r.id, AI.SCORE(PROMPT('q7: {{0}}', r.body)) AS score "
            f"FROM reviews r {where} ORDER BY score DESC, r.id LIMIT 3")
    # the sort holds three rows after r02, and r00, r02, and r04 at 1.0
    # after r04
    hinted = Oracle(**TOP_K_SCORES, tightens_after={2, 4})
    context = _reviews_context(hinted, 12)
    assert _rows(context.sql(text)) == TOP_K_ROWS, where
    # without the LIMIT the sort holds every row and hands no threshold
    every = Oracle(**TOP_K_SCORES)
    whole = _reviews_context(every, 12)
    assert _rows(whole.sql(text.replace(" LIMIT 3", "")))[:3] == TOP_K_ROWS, where
    assert _scored_rows(every) == 12, where
    # r03 may still score above r01's 0.3; once r00, r02, and r04 hold
    # 1.0, a later review can only tie at 1.0 with a larger id, which the
    # top 3 does not take
    assert _scored_rows(hinted) == 5, where
    context.session.close()
    whole.session.close()


def test_a_join_with_a_small_build_side_skips_probe_rows_whose_key_is_absent():
    text = ("SELECT r.id FROM picked p JOIN reviews r ON r.id = p.review "
            "WHERE AI_FILTER(PROMPT('q1: {0}', r.body))")
    every = Oracle(filters=lambda alias, document, position: True)
    context = _reviews_context(every, 12)
    picked = pa.table({"id": ["p0", "p1"], "review": ["r03", "r07"]})
    context.register("picked", quail.DocumentProvider.from_table(picked, id_col="id"))
    plan = context._native.explain(text)
    assert "HashJoinExec" in plan and "ModelRegionExec: hint=[DynamicFilter" in plan
    assert sorted(_rows(context.sql(text))) == [("r03",), ("r07",)]
    # the join's keys are complete before the region starts
    assert sorted(every.asked) == [("r", 3), ("r", 7)]
    context.session.close()


def test_a_top_k_over_a_join_skips_the_anchors_past_its_threshold():
    # the tables' id columns have different names: a test above the
    # region reads a column by its name
    text = ("SELECT r.id, t.tid FROM reviews r JOIN topics t "
            "ON AI_FILTER(PROMPT('q4: {0} {1}', r.body, t.topic)) "
            "ORDER BY t.tid LIMIT 1")
    asked = []

    def joins(node, assignment):
        asked.append(assignment["t"])
        return True

    # the join's anchor is the topic; once t0's pairs are in, the sort
    # holds its one row and tightens
    oracle = Oracle(joins=joins, tightens_after={0})
    context = _reviews_context(oracle, 12)
    topics = pa.table({"tid": ["t0", "t1", "t2"],
                       "topic": ["ending", "pacing", "cast"]})
    context.register("topics", quail.DocumentProvider.from_table(topics, id_col="tid"))
    assert "ModelRegionExec: hint=[DynamicFilter" in context._native.explain(text)
    assert [row[1] for row in _rows(context.sql(text))] == ["t0"]
    # t1 and t2 are past the top row
    assert sorted(set(asked)) == [0]
    context.session.close()


def test_a_region_that_reads_another_regions_rows_runs_after_it(context):
    # the CTE's LIMIT runs in DataFusion between the two regions, so the
    # second region reads the first one's rows; both run on the thread
    # that called DataFusion, the first to its end before the second
    text = ("WITH kept AS (SELECT r.id, r.body FROM reviews r "
            "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) ORDER BY r.id LIMIT 2) "
            "SELECT r.id FROM kept r WHERE AI_FILTER(PROMPT('q2: {0}', r.body)) "
            "ORDER BY r.id")
    assert len(context.region_plans(text)) == 2
    # q1 keeps r0, r2, and r3, and the LIMIT r0 and r2; the oracle answers
    # q2 by each document's position in the second region's input, 1 and 1
    assert _rows(context.sql(text)) == [("r0",), ("r2",)]


def test_a_table_name_with_underscores_keeps_its_rows_in_a_region():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    try:
        ORACLE.catalog = session.catalog
        context = quail.SessionContext(session,
                                       physical_executor=oracle_executor(ORACLE))
        context.register("reviews__archive",
                         quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
        # a plain test cuts the table's rows into a region input
        text = ("SELECT r.id FROM reviews__archive r WHERE r.stars > 1 "
                "AND AI_FILTER(PROMPT('q1: {0}', r.body)) ORDER BY r.id")
        assert _rows(context.sql(text)) == [("r0",), ("r2",)]
        with pytest.raises(quail.common.expressions.CompileError, match="'@'"):
            context.register("reviews@r",
                             quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    finally:
        session.close()


def _yes_context(tables):
    """Return a session and a context whose model answers yes to everything."""
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    oracle = Oracle(filters=lambda *_: True, joins=lambda *_: True)
    oracle.catalog = session.catalog
    context = quail.SessionContext(
        session, physical_executor=oracle_executor(oracle))
    for name, table in tables.items():
        context.register(name, quail.DocumentProvider.from_table(table, id_col="id"))
    return session, context


def _sorted_rows(table):
    return sorted(zip(*(column.to_pylist() for column in table.columns)))


def test_a_join_above_a_region_of_two_ids_keeps_every_pair(monkeypatch):
    monkeypatch.setenv("DATAFUSION_OPTIMIZER_ENABLE_DYNAMIC_FILTER_PUSHDOWN", "true")
    session, context = _yes_context({
        "r": pa.table({"id": ["r0", "r1"], "body": ["a", "b"]}),
        "t": pa.table({"id": ["t0", "t1"], "body": ["c", "d"]}),
        "v": pa.table({"id": ["v0", "v1"], "rid": ["r0", "r1"]}),
    })
    # the region returns r.id and t.id; the join's filter of r.id must
    # not reach it reading t.id
    rows = _sorted_rows(context.sql(
        "SELECT r.id, t.id, v.id FROM r CROSS JOIN t JOIN v ON v.rid = r.id "
        "WHERE AI.IF(PROMPT('q: {0}', concat(r.body, ' ', t.body)))"))
    session.close()
    assert rows == [("r0", "t0", "v0"), ("r0", "t1", "v0"),
                    ("r1", "t0", "v1"), ("r1", "t1", "v1")]


def test_a_join_filter_keeps_a_document_while_it_keeps_one_of_its_copies(
        monkeypatch):
    monkeypatch.setenv("DATAFUSION_OPTIMIZER_ENABLE_DYNAMIC_FILTER_PUSHDOWN", "true")
    session, context = _yes_context({
        "r": pa.table({"id": ["r0", "r1", "r2", "r3"], "stars": [1, 2, 3, 4]}),
        # d0 and d2 hold the same text, so one answer serves both
        "d": pa.table({"id": ["d0", "d1", "d2", "d3"],
                       "name": ["a", "b", "a", "d"],
                       "rid": ["r0", "r1", "r2", "r3"]}),
    })
    rows = _sorted_rows(context.sql(
        "SELECT r.id, d.id FROM r JOIN d ON r.id = d.rid "
        "WHERE r.stars >= 2 AND AI.IF(PROMPT('ok? {0}', d.name))"))
    session.close()
    assert rows == [("r1", "d1"), ("r2", "d2"), ("r3", "d3")]


def test_an_ai_call_of_one_table_in_a_joins_on_runs_below_the_join():
    session, context = _yes_context({
        "r": pa.table({"id": ["r0", "r1"], "body": ["a", "b"]}),
        "v": pa.table({"id": ["v0", "v1"], "rid": ["r0", "r1"]}),
    })
    cases = {
        "a test of one table":
            "SELECT r.id, v.id FROM r JOIN v ON r.id = v.rid "
            "AND AI.IF(PROMPT('q: {0}', r.body))",
        "the same test of an outer join":
            "SELECT r.id, v.id FROM r LEFT JOIN v ON r.id = v.rid "
            "AND AI.IF(PROMPT('q: {0}', r.body))",
    }
    for name, sql in cases.items():
        assert _sorted_rows(context.sql(sql)) == [("r0", "v0"), ("r1", "v1")], name
    session.close()


def test_a_join_keys_answer_is_also_returned():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    names = ["Acme", "Globex"]
    oracle = Oracle(maps=lambda spec, document: names[document])
    oracle.catalog = session.catalog
    context = quail.SessionContext(
        session, physical_executor=oracle_executor(oracle))
    context.register("r", quail.DocumentProvider.from_table(
        pa.table({"id": ["r0", "r1"], "body": ["a", "b"]}), id_col="id"))
    context.register("v", quail.DocumentProvider.from_table(
        pa.table({"id": ["v0"], "name": ["Acme"]}), id_col="id"))
    call = "AI.MAP(r.body, 'Name the vendor.', 16)"
    rows = _sorted_rows(context.sql(
        f"SELECT r.id, v.id, {call} AS m FROM r JOIN v ON v.name = {call}"))
    session.close()
    assert rows == [("r0", "v0", "Acme")]
