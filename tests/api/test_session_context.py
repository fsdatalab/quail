"""quail.SessionContext: DataFusion binds the query, Quail runs each model region.

Each transcript test's transcript shows the plan, the
questions the model answered, and the rows (helpers/transcript.py).
"""

import threading

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER, check_table
from helpers.transcript import Transcripts, assert_transcript

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


# a forward pass size that holds about one document and its question
SMALL_PASS = 256

# per operator family: a query of that operator alone, and the oracle
# truth it is answered from; `answered` records each answer
FAMILIES = {
    "filter": ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
               lambda answered: Oracle(filters=lambda alias, document, position:
                                       answered(document % 2 == 0))),
    "classify": (QUERIES["IMDB-11 classify"],
                 lambda answered: Oracle(labels=lambda spec, ids:
                                         answered(LABELS[ids[0] % 4]))),
    "score documents": (
        "SELECT r.id, AI.SCORE(PROMPT('q7: {0}', r.body)) AS score FROM reviews r",
        lambda answered: Oracle(scores=lambda spec, ids: answered(ids[0] / 10))),
    "score pairs": (
        "SELECT r.id, a.id, AI.SCORE(PROMPT('q9: {0} {1}', r.body, a.aspect)) "
        "AS fit FROM reviews r CROSS JOIN aspects a",
        lambda answered: Oracle(scores=lambda spec, ids: answered(sum(ids) / 10))),
    "extract": (QUERIES["CUAD-1 extract"],
                lambda answered: Oracle(answers=lambda spec, document: answered(
                    ["2027", "2030", None][document % 3]))),
    "map": ("SELECT r.id, AI.MAP(r.body, 'q8: Summarize.', 16) AS summary "
            "FROM reviews r",
            lambda answered: Oracle(maps=lambda spec, document: answered(
                f"s{document}"))),
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
    arrived = threading.Event()

    def answered(value):
        # rows cross to the region while the model goes on: an answer
        # after the first gives them a moment to arrive
        if "answered" in events:
            arrived.wait(0.2)
        events.append("answered")
        return value

    oracle = truth(answered)
    # a forward pass holds a document or two, so the region takes several
    oracle.chunk_tokens = SMALL_PASS
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    executor = oracle_executor(oracle)

    def run(request):
        response = executor(request)
        events.append("finished")
        return response

    context = quail.SessionContext(session, physical_executor=run)
    # the model writes a map's or an extraction's answer over several
    # passes: with a few more rows, the first rows' answers go on while
    # it has later rows still to read
    reviews = pa.table({
        "id": [f"r{index}" for index in range(8)],
        "body": [f"{body}{index}" for index, body in enumerate(
            REVIEWS.column("body").to_pylist() * 2)],
        "stars": REVIEWS.column("stars").to_pylist() * 2})
    contracts = pa.table({
        "id": [f"c{index}" for index in range(6)],
        "text": [f"{text}Page {index}." for index, text in enumerate(
            CONTRACTS.column("text").to_pylist() * 2)]})
    for name, table in (("reviews", reviews), ("aspects", ASPECTS),
                        ("contracts", contracts)):
        context.register(name, quail.DocumentProvider.from_table(table, id_col="id"))
    ((_, planned),) = context._native.plans(text)
    batches = []

    def on_rows(table):
        events.append("rows")
        batches.append(table)
        arrived.set()

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
                   tightens_after={1}, chunk_tokens=SMALL_PASS)
    context = _reviews_context(every, 12)
    assert "ModelRegionExec: hint=[DynamicFilter" in context._native.explain(text)
    table = context.sql(text)
    assert _rows(table) == [("r00",), ("r01",)]
    # every later id is past the top two; the pass that asks r02 was
    # packed before r01's answer was read
    assert every.asked == [("r", 0), ("r", 1), ("r", 2)]
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
    hinted = Oracle(**TOP_K_SCORES, tightens_after={2, 4},
                    chunk_tokens=SMALL_PASS)
    context = _reviews_context(hinted, 12)
    assert _rows(context.sql(text)) == TOP_K_ROWS, where
    # without the LIMIT the sort holds every row and hands no threshold
    every = Oracle(**TOP_K_SCORES)
    whole = _reviews_context(every, 12)
    assert _rows(whole.sql(text.replace(" LIMIT 3", "")))[:3] == TOP_K_ROWS, where
    assert _scored_rows(every) == 12, where
    # r03 may still score above r01's 0.3; once r00, r02, and r04 hold
    # 1.0, a later review can only tie at 1.0 with a larger id, which the
    # top 3 does not take. The model reads reviews that share a prefix
    # together, r00, r10, r11, then r01 to r04; and it reads r05, whose
    # pass was packed before r04's score was read
    assert hinted.calls == [("score", [[0], [10], [11], [1], [2], [3], [4], [5]])], \
        where
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
    # a pass holds one topic and its twelve reviews
    oracle = Oracle(joins=joins, tightens_after={0}, chunk_tokens=512)
    context = _reviews_context(oracle, 12)
    topics = pa.table({"tid": ["t0", "t1", "t2"],
                       "topic": ["ending", "pacing", "cast"]})
    context.register("topics", quail.DocumentProvider.from_table(topics, id_col="tid"))
    assert "ModelRegionExec: hint=[DynamicFilter" in context._native.explain(text)
    assert [row[1] for row in _rows(context.sql(text))] == ["t0"]
    # t1 and t2 are past the top row
    assert sorted(set(asked)) == [0]
    context.session.close()




def _answer(kind, prompt, rows):
    """ORACLE's answers, by row id; 'q:' and 'ok?' pass all; a map names vendors."""
    if prompt.startswith(("q:", "ok?")):
        return True
    index = int(rows[0][1:])
    if kind == "extract":
        return ["2027", "2030", None][index]
    if kind == "label":
        return LABELS[index]
    if kind == "map":
        return ["Acme", "Globex"][index]
    if len(rows) == 2:
        return (index + int(rows[1][1:])) % 2 == 0
    if prompt.startswith("q3:"):
        return rows[0] in ("c0", "c1")
    return rows[0] in {"q1": ("r0", "r2", "r3"), "q2": ("r0", "r1", "r3")}.get(
        prompt[:2], rows)


def _transcript(tables, *queries) -> str:
    """Transcripts of queries, each SQL or (SQL, keyword arguments)."""
    with Transcripts(tables, _answer) as runs:
        return "\n".join(runs.query(query[0], **query[1])
                         if isinstance(query, tuple) else runs.query(query)
                         for query in queries)


def test_sql_around_a_region_runs_in_datafusion():
    tables = {"reviews": REVIEWS, "aspects": ASPECTS, "contracts": CONTRACTS,
              "reviews__archive": REVIEWS}
    fields = "c.text, {'end': 'q6: When does it end?'}"
    assert_transcript(_transcript(
        tables,
        # a union's column belongs to no one table
        ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) "
         "UNION ALL SELECT a.id FROM aspects a ORDER BY id",
         {"ordered": True, "columns": True}),
        # the CTE's LIMIT runs in DataFusion between the two regions, so the
        # second region reads the first one's rows
        ("WITH kept AS (SELECT r.id, r.body FROM reviews r "
         "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) ORDER BY r.id LIMIT 2) "
         "SELECT r.id FROM kept r WHERE AI_FILTER(PROMPT('q2: {0}', r.body)) "
         "ORDER BY r.id", {"ordered": True}),
        # a plain test cuts the table's rows into a region input
        "SELECT r.id FROM reviews__archive r WHERE r.stars > 1 "
        "AND AI_FILTER(PROMPT('q1: {0}', r.body))",
        # an extraction returns and filters on its answer positions
        (f"SELECT c.id, AI.EXTRACT({fields}) AS lease, "
         f"AI.EXTRACT_SPAN({fields}) AS lease_span FROM contracts c "
         f"WHERE AI.EXTRACT_SPAN({fields})['end']['start'] < 15",
         {"columns": True})), """\
--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) UNION ALL
    SELECT a.id FROM aspects a ORDER BY id ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.body), filter: __quail_answer_1, returns: []
asked 4 of 4
  'q1: {0}': r0 TRUE, r1 FALSE, r2 TRUE, r3 TRUE
columns id string
rows 5, in order
  a0
  a1
  r0
  r2
  r3

--- WITH kept AS (SELECT r.id, r.body FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}',
    r.body)) ORDER BY r.id LIMIT 2) SELECT r.id FROM kept r WHERE AI_FILTER(PROMPT('q2:
    {0}', r.body)) ORDER BY r.id ---
plan
  __quail_answer_2 = ai_filter('q2: {0}', r.column_1), filter: __quail_answer_2,
      returns: []
  __quail_answer_1 = ai_filter('q1: {0}', r.body), filter: __quail_answer_1, returns: []
asked 6 of 6
  'q1: {0}': r0 TRUE, r1 FALSE, r2 TRUE, r3 TRUE
  'q2: {0}': r0 TRUE, r2 FALSE
rows 1, in order
  r0

--- SELECT r.id FROM reviews__archive r WHERE r.stars > 1 AND AI_FILTER(PROMPT('q1:
    {0}', r.body)) ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.body), filter: __quail_answer_1, returns: []
asked 3 of 3
  'q1: {0}': r0 TRUE, r1 FALSE, r2 TRUE
rows 2
  r0
  r2

--- SELECT c.id, AI.EXTRACT(c.text, {'end': 'q6: When does it end?'}) AS lease,
    AI.EXTRACT_SPAN(c.text, {'end': 'q6: When does it end?'}) AS lease_span FROM
    contracts c WHERE AI.EXTRACT_SPAN(c.text, {'end': 'q6: When does it
    end?'})['end']['start'] < 15 ---
plan
  lease = ai_extract(c.text, named_struct('end', 'q6: When does it end?')), filter:
      get_field(lease_span, 'end', 'start') < Int32(15)
asked 3 of 3
  'q6: When does it end?': c0 '2027', c1 '2030', c2 NULL
columns c.id string, lease struct<end: string>, lease_span struct<end: struct<start:
    int32, end: int32>>
rows 2
  c0, {'end': '2027'}, {'end': {'start': 14, 'end': 18}}
  c1, {'end': '2030'}, {'end': {'start': 14, 'end': 18}}
""")


def test_a_table_name_with_an_at_sign_is_refused():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    try:
        context = quail.SessionContext(session)
        with pytest.raises(quail.common.expressions.CompileError, match="'@'"):
            context.register("reviews@r",
                             quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    finally:
        session.close()


def test_an_ai_call_beside_or_inside_a_join_is_answered_once_per_row(monkeypatch):
    monkeypatch.setenv("DATAFUSION_OPTIMIZER_ENABLE_DYNAMIC_FILTER_PUSHDOWN", "true")
    tables = {
        "r": pa.table({"id": ["r0", "r1"], "body": ["a", "b"]}),
        "t": pa.table({"id": ["t0", "t1"], "body": ["c", "d"]}),
        "v": pa.table({"id": ["v0", "v1", "v2"], "rid": ["r0", "r1", "r9"]}),
        "vendors": pa.table({"id": ["n0"], "name": ["Acme"]}),
        "names": pa.table({"id": ["x0"], "name": ["x"]}),
    }
    test = "r.id = v.rid AND AI.IF(PROMPT('q: {0}', r.body))"
    call = "AI.MAP(r.body, 'Name the vendor.', 16)"
    assert_transcript(_transcript(
        tables,
        # the region returns r.id and t.id; the join's filter of r.id must
        # not reach it reading t.id
        "SELECT r.id, t.id, v.id FROM r CROSS JOIN t JOIN v ON v.rid = r.id "
        "WHERE AI.IF(PROMPT('q: {0}', concat(r.body, ' ', t.body)))",
        # an AI call of one table in a join's ON runs below the join
        f"SELECT r.id, v.id FROM r JOIN v ON {test}",
        f"SELECT r.id, v.id FROM r LEFT JOIN v ON {test}",
        # a join key's answer is also returned
        f"SELECT r.id, v.id, {call} AS m FROM r JOIN vendors v ON v.name = {call}",
        # the AI.IF's answer is a column of r, which the join must pass on
        "SELECT r.id, AI.IF(PROMPT('q: {0}', r.body)) AS k FROM r "
        "JOIN names v ON AI.MAP(r.body, 'Name it.', 8) <> v.name",
        # an AI call in a correlated subquery
        f"SELECT v.id FROM v WHERE EXISTS (SELECT 1 FROM r WHERE {test})",
        f"SELECT v.id FROM v WHERE NOT EXISTS (SELECT 1 FROM r WHERE {test})",
        f"SELECT v.id, (SELECT COUNT(*) FROM r WHERE {test}) AS n FROM v"), """\
--- SELECT r.id, t.id, v.id FROM r CROSS JOIN t JOIN v ON v.rid = r.id WHERE
    AI.IF(PROMPT('q: {0}', concat(r.body, ' ', t.body))) ---
plan
  __quail_answer_1 = ai_filter('q: {0}', __quail_rows_0.column_4)
asked 4 of 4
  'q: {0}': 'a c' TRUE, 'a d' TRUE, 'b c' TRUE, 'b d' TRUE
rows 4
  r0, t0, v0
  r0, t1, v0
  r1, t0, v1
  r1, t1, v1

--- SELECT r.id, v.id FROM r JOIN v ON r.id = v.rid AND AI.IF(PROMPT('q: {0}', r.body))
    ---
plan
  __quail_call_1 = ai_filter('q: {0}', r.body), filter: __quail_call_1, returns: []
asked 2 of 2
  'q: {0}': r0 TRUE, r1 TRUE
rows 2
  r0, v0
  r1, v1

--- SELECT r.id, v.id FROM r LEFT JOIN v ON r.id = v.rid AND AI.IF(PROMPT('q: {0}',
    r.body)) ---
plan
  __quail_call_1 = ai_filter('q: {0}', r.body)
asked 2 of 2
  'q: {0}': r0 TRUE, r1 TRUE
rows 2
  r0, v0
  r1, v1

--- SELECT r.id, v.id, AI.MAP(r.body, 'Name the vendor.', 16) AS m FROM r JOIN vendors v
    ON v.name = AI.MAP(r.body, 'Name the vendor.', 16) ---
plan
  m = ai_map(r.body, 'Name the vendor.', Int64(16))
asked 2 of 2
  'Name the vendor.': r0 'Acme', r1 'Globex'
rows 1
  r0, n0, 'Acme'

--- SELECT r.id, AI.IF(PROMPT('q: {0}', r.body)) AS k FROM r JOIN names v ON
    AI.MAP(r.body, 'Name it.', 8) <> v.name ---
plan
  __quail_call_1 = ai_map(r.body, 'Name it.', Int64(8))
  k = ai_filter('q: {0}', r.body)
asked 4 of 4
  'q: {0}': r0 TRUE, r1 TRUE
  'Name it.': r0 'Acme', r1 'Globex'
rows 2
  r0, TRUE
  r1, TRUE

--- SELECT v.id FROM v WHERE EXISTS (SELECT 1 FROM r WHERE r.id = v.rid AND
    AI.IF(PROMPT('q: {0}', r.body))) ---
plan
  __quail_answer_1 = ai_filter('q: {0}', r.body), filter: __quail_answer_1, fetch: 1 per
      [id], returns: []
asked 2 of 2
  'q: {0}': r0 TRUE, r1 TRUE
rows 2
  v0
  v1

--- SELECT v.id FROM v WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.id = v.rid AND
    AI.IF(PROMPT('q: {0}', r.body))) ---
plan
  __quail_answer_1 = ai_filter('q: {0}', r.body), filter: __quail_answer_1, returns: []
asked 2 of 2
  'q: {0}': r0 TRUE, r1 TRUE
rows 1
  v2

--- SELECT v.id, (SELECT COUNT(*) FROM r WHERE r.id = v.rid AND AI.IF(PROMPT('q: {0}',
    r.body))) AS n FROM v ---
plan
  __quail_answer_1 = ai_filter('q: {0}', r.body), filter: __quail_answer_1, returns: []
asked 2 of 2
  'q: {0}': r0 TRUE, r1 TRUE
rows 3
  v0, 1
  v1, 1
  v2, 0
""")


def test_a_join_filter_keeps_a_document_while_it_keeps_one_of_its_copies(
        monkeypatch):
    monkeypatch.setenv("DATAFUSION_OPTIMIZER_ENABLE_DYNAMIC_FILTER_PUSHDOWN", "true")
    tables = {
        "r": pa.table({"id": ["r0", "r1", "r2", "r3"], "stars": [1, 2, 3, 4]}),
        # d0 and d2 hold the same text, so one answer serves both
        "d": pa.table({"id": ["d0", "d1", "d2", "d3"],
                       "name": ["a", "b", "a", "d"],
                       "rid": ["r0", "r1", "r2", "r3"]}),
    }
    assert_transcript(_transcript(
        tables,
        "SELECT r.id, d.id FROM r JOIN d ON r.id = d.rid "
        "WHERE r.stars >= 2 AND AI.IF(PROMPT('ok? {0}', d.name))"), """\
--- SELECT r.id, d.id FROM r JOIN d ON r.id = d.rid WHERE r.stars >= 2 AND
    AI.IF(PROMPT('ok? {0}', d.name)) ---
plan
  __quail_answer_1 = ai_filter('ok? {0}', d.name), filter: __quail_answer_1, returns: []
asked 3 of 3
  'ok? {0}': d0/d2 TRUE, d1 TRUE, d3 TRUE
rows 3
  r1, d1
  r2, d2
  r3, d3
""")


NULL_DOCUMENT_CALL = "AI.CLASSIFY(PROMPT('q: {0}', r.body), ['a', 'b'])"


def _null_document_transcript(*queries) -> str:
    """The transcripts of queries over r, whose r1 has a NULL body, and v."""
    tables = {"r": pa.table({"id": ["r0", "r1"], "body": ["x", None]}),
              "v": pa.table({"id": ["v0", "v1"], "name": ["a", "b"]})}

    def answer(kind, prompt, rows):
        return {"label": "a", "score": 0.5}.get(kind, True)

    with Transcripts(tables, answer) as runs:
        return "\n".join(runs.query(sql) for sql in queries)


def test_a_test_of_a_null_documents_label_keeps_the_null_row():
    assert_transcript(_null_document_transcript(
        f"SELECT r.id FROM r WHERE {NULL_DOCUMENT_CALL} IS NULL"), """\
--- SELECT r.id FROM r WHERE AI.CLASSIFY(PROMPT('q: {0}', r.body), ['a', 'b']) IS NULL
    ---
plan
  __quail_column_1 = ai_classify('q: {0}', r.body, make_array('a', 'b')), filter:
      __quail_column_1 IS NULL
asked 1 of 1
  'q: {0}': r0 'a'
rows 1
  r1
""")


def test_an_ai_call_of_a_null_document_is_null_and_asks_no_model():
    call = NULL_DOCUMENT_CALL
    assert_transcript(_null_document_transcript(
        f"SELECT r.id, {call} AS c FROM r",
        "SELECT r.id FROM r WHERE AI.IF(PROMPT('q: {0}', r.body))",
        "SELECT r.id FROM r WHERE NOT AI.IF(PROMPT('q: {0}', r.body))",
        # a pair with a NULL document has no score
        "SELECT r.id, v.id, AI.SCORE(PROMPT('q: {0} {1}', r.body, v.name)) "
        "AS s FROM r CROSS JOIN v",
        # a NULL written without a type is a NULL document too
        "SELECT AI.IF(PROMPT('q: {0}', NULL)) AS k",
        # v1 has no partner; its row has no document, so no answer
        f"SELECT v.id, {call} AS c FROM r RIGHT JOIN v ON {call} = v.name"), """\
--- SELECT r.id, AI.CLASSIFY(PROMPT('q: {0}', r.body), ['a', 'b']) AS c FROM r ---
plan
  c = ai_classify('q: {0}', r.body, make_array('a', 'b'))
asked 1 of 1
  'q: {0}': r0 'a'
rows 2
  r0, 'a'
  r1, NULL

--- SELECT r.id FROM r WHERE AI.IF(PROMPT('q: {0}', r.body)) ---
plan
  __quail_answer_1 = ai_filter('q: {0}', r.body), filter: __quail_answer_1, returns: []
asked 1 of 1
  'q: {0}': r0 TRUE
rows 1
  r0

--- SELECT r.id FROM r WHERE NOT AI.IF(PROMPT('q: {0}', r.body)) ---
plan
  __quail_answer_1 = ai_filter('q: {0}', r.body), filter: NOT __quail_answer_1, returns:
      []
asked 1 of 1
  'q: {0}': r0 TRUE
rows 0

--- SELECT r.id, v.id, AI.SCORE(PROMPT('q: {0} {1}', r.body, v.name)) AS s FROM r CROSS
    JOIN v ---
plan
  s = ai_score('q: {0} {1}', r.body, v.name)
asked 2 of 2
  'q: {0} {1}': r0+v0 0.5, r0+v1 0.5
rows 4
  r0, v0, 0.5
  r0, v1, 0.5
  r1, v0, NULL
  r1, v1, NULL

--- SELECT AI.IF(PROMPT('q: {0}', NULL)) AS k ---
plan
  k = ai_filter('q: {0}', __quail_rows_1.column_0)
asked 0 of 0
rows 1
  NULL

--- SELECT v.id, AI.CLASSIFY(PROMPT('q: {0}', r.body), ['a', 'b']) AS c FROM r RIGHT
    JOIN v ON AI.CLASSIFY(PROMPT('q: {0}', r.body), ['a', 'b']) = v.name ---
plan
  c = ai_classify('q: {0}', r.body, make_array('a', 'b'))
asked 1 of 1
  'q: {0}': r0 'a'
rows 2
  v0, 'a'
  v1, NULL
""")
