"""Each scan loads the columns its region reads, and a join's placements."""


import pyarrow as pa
import pyarrow.parquet as pq
from helpers.oracle import Oracle, finish
from helpers.planning import placements
from helpers.sessions import CONFIG, TOKENIZER, _run, make_executor

import quail
from quail.api.config import EngineConfig
from quail.builtins import LOGICAL_RULES, built_in_registry, logical_rules_without


def _session(tmp_path, tokenizer=TOKENIZER):
    session = quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="quail",
                     device="h100-sxm"),
        tokenizer=tokenizer,
    )
    path = tmp_path / "reviews.parquet"
    pq.write_table(pa.table({
        "id": ["r0", "r1", "r2"],
        "review": ["good one", "bad one", "fine one"],
        "stars": [5, 1, 3],
        "wide": ["x" * 50, "y" * 50, "z" * 50],
    }), str(path))
    session.register("reviews", quail.DocumentProvider.from_parquet(
        str(path), id_col="id"))
    return session


def _token_buffer(scan):
    """The address of the first token id a scan's store holds."""
    return scan.tokens.arrow_array.chunk(0).values.buffers()[1].address


def test_projected_results_token_reuse_and_join_projection(tmp_path):
    session = _session(tmp_path)
    truth = {"r": {"q:": [1, 0, 1]}}
    query = session.sql(
        "SELECT r.stars, r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))")

    result = _run(query, make_executor(truth))

    assert sorted(result.to_rows()) == [(3, "r2"), (5, "r0")]
    # the region reads the id; DataFusion joins stars back on it
    (scan,) = query.facts()["scans"]
    assert scan["columns"] == ["id"]
    store = query.token_inputs()["r"]
    # the id, then the document, which locates answers
    assert store.projected_columns == ("id", "review")
    # a later query reads the same token store instead of tokenizing again
    tokenized = _token_buffer(store)

    later = session.sql(
        "SELECT r.stars, r.review FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))")
    second = _run(later, make_executor(truth))
    assert sorted(second.to_rows()) == [(3, "fine one"), (5, "good one")]
    assert _token_buffer(later.token_inputs()["r"]) == tokenized

    truth = {"r": {"q:": [0, 1, 0]}}

    text = _run(session.sql(
        "SELECT r.review FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))"), make_executor(truth))
    assert text.to_rows() == [("bad one",)]

    everything = session.sql(
        "SELECT * FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))")
    star = _run(everything, make_executor(truth))
    assert star.columns == ["r.id", "r.review", "r.stars", "r.wide"]
    assert star.to_rows() == [("r1", "bad one", 1, "y" * 50)]
    assert _token_buffer(everything.token_inputs()["r"]) == tokenized
    session.close()

    session = _session(tmp_path)
    session.register("products", quail.DocumentProvider.from_table(pa.table({
        "asin": ["p0", "p1"], "description": ["product 0", "product 1"],
    }), id_col="asin"))
    truth = {"r": {"q:": [1, 0, 1]}}
    # the planner picks the anchor; the rule is symmetric so either works
    even_sum = lambda a, b: (a + b) % 2 == 0  # noqa: E731
    join_truth = {("r", "p"): even_sum, ("p", "r"): even_sum}
    query = session.sql(
        "SELECT r.id, p.asin FROM reviews r JOIN products p ON "
        "AI_FILTER(PROMPT('m {0} {1}', r.review, p.description)) "
        "WHERE AI_FILTER(PROMPT('q: {0}', r.review))")
    result = _run(query, make_executor(truth, join_truth))
    assert sorted(result.to_rows()) == [("r0", "p0"), ("r2", "p0")]
    assert result.count() == 2
    session.close()


def _pair_session():
    session = quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=TOKENIZER)
    session.register("reviews", quail.DocumentProvider.from_table(
        pa.table({"id": ["r0", "r1"], "review": ["good one", "bad one"],
                  "stars": [5, 1]}), id_col="id"))
    session.register("products", quail.DocumentProvider.from_table(
        pa.table({"asin": ["p0"], "description": ["a thing"]}),
        id_col="asin"))
    return session


JOIN_SQL = """
    SELECT r.id, p.asin FROM reviews r
    JOIN products p ON AI_FILTER(PROMPT('same {0} {1}', r.review, p.description))
"""

LABELLED_JOIN_SQL = """
    SELECT r.id, p.asin,
           AI.CLASSIFY(PROMPT('topic of {0}', r.review), ['a', 'b']) AS topic
    FROM reviews r
    JOIN products p ON AI_FILTER(PROMPT('same {0} {1}', r.review, p.description))
    WHERE AI.CLASSIFY(PROMPT('topic of {0}', r.review), ['a', 'b']) IN ('a')
"""


def test_a_full_join_offers_its_classification_and_test_above_the_join():
    session = _pair_session()
    # a join with no column operator has one placement
    candidate, = placements(session.sql(JOIN_SQL).operators)
    assert candidate.rules == ()

    labelled = session.sql(LABELLED_JOIN_SQL).operators
    written, lifted = placements(labelled)
    assert [column.above_joins for column in written.found.classifies] == [False]
    assert lifted.rules == ("should_lift_column_operators",)
    assert [column.above_joins for column in lifted.found.classifies] == [True]
    ((alias, (test,)),) = lifted.found.column_filters
    assert (alias, test.name, test.before_joins) == ("r", "topic", False)
    session.close()



# R5, π_Q(AI(R; P)) ⇒ π_Q(AI(R; P = Q[P])): an AI operator returns the
# added columns the SELECT list above it reads
PROJECTION_RULE = "derive.projection"
EXTRACTED = pa.table({"id": ["r0", "r1", "r2"], "body": ["ab x", "cd", "ab y"]})


def _extract_session(rules) -> quail.Session:
    session = quail.Session(CONFIG, tokenizer=TOKENIZER,
                            registry=built_in_registry(logical_rules=rules))
    session.register("reviews",
                     quail.DocumentProvider.from_table(EXTRACTED, id_col="id"))
    return session


def test_an_extraction_under_a_select_list_returns_only_the_columns_it_reads():
    sql = ("SELECT x.id, x.e FROM (SELECT r.id, AI_EXTRACT(r.body, {'w': 'which?'}) "
           "AS e FROM reviews r) x")
    found = []
    for rules in (LOGICAL_RULES,
                  logical_rules_without(PROJECTION_RULE)):
        session = _extract_session(rules)
        query = session.sql(sql)
        node = next(line for line in query.logical.splitlines() if "AiNode" in line)
        oracle = Oracle(answers=lambda spec, document: None if document == 1 else "ab")
        rows = finish(query, oracle).collect().to_pylist()
        found.append((node.endswith("returns: [e]"),
                      sorted((row["x.id"], row["x.e"]["w"]) for row in rows)))
        session.close()
    # with the rule the node returns its answer without its spans
    assert found[0] == (True, [("r0", "ab"), ("r1", None), ("r2", "ab")])
    assert found[1] == (False, found[0][1])


def test_an_extraction_the_query_returns_keeps_its_spans():
    session = _extract_session(LOGICAL_RULES)
    query = session.sql(
        "SELECT r.id, AI_EXTRACT(r.body, {'w': 'which?'}) AS e, "
        "AI_EXTRACT_SPAN(r.body, {'w': 'which?'}) AS e_span FROM reviews r")
    assert "returns:" not in query.logical
    rows = finish(query, Oracle(answers=lambda spec, document: "ab")).collect()
    assert "e_span" in rows.column_names
    session.close()


def test_an_extraction_asks_only_the_fields_the_query_reads():
    questions = "{'w': 'which?', 'v': 'what?', 'u': 'who?'}"
    sql = (f"SELECT x.id, x.e.w AS w FROM (SELECT r.id, "
           f"AI_EXTRACT(r.body, {questions}) AS e FROM reviews r) x")
    found = []
    for rules in (LOGICAL_RULES,
                  logical_rules_without(PROJECTION_RULE)):
        session = _extract_session(rules)
        asked = []

        def answer(spec, document, asked=asked):
            asked.append(spec.name)
            return None if document == 1 else f"ab {spec.name}"

        rows = finish(session.sql(sql), Oracle(answers=answer)).collect()
        found.append((sorted(asked), sorted(
            (row["x.id"], row["w"]) for row in rows.to_pylist())))
        session.close()
    # with the rule each document is asked the one field the query reads,
    # instead of all three, and that field's answers do not change
    assert found[0][0] == ["e.w"] * 3
    assert found[1][0] == sorted(["e.u", "e.v", "e.w"] * 3)
    assert found[0][1] == [("r0", "ab e.w"), ("r1", None), ("r2", "ab e.w")]
    assert found[1][1] == found[0][1]
