"""Projection pushdown rule tests."""


import pyarrow as pa
import pyarrow.parquet as pq
from helpers.logical import kinds, nodes, read_plan
from helpers.planning import placements
from helpers.sessions import TOKENIZER, _run, make_executor

import quail
from quail.api.config import EngineConfig
from quail.physical import Recombine


def _scans(plan):
    return {node.alias: node for node in nodes(plan, "Scan")}


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


def test_projected_results_token_reuse_and_join_projection(tmp_path):
    session = _session(tmp_path)
    truth = {"r": {"q:": [1, 0, 1]}}
    query = session.sql(
        "SELECT r.stars, r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))")

    result = _run(query, make_executor(truth))

    assert sorted(result.to_rows()) == [(3, "r2"), (5, "r0")]
    scan = _scans(query.logical)["r"]
    assert scan.columns == ("stars", "id")
    store = query.token_inputs()["r"]
    # the projected columns, then the document, which locates answers
    assert store.projected_columns == ("stars", "id", "review")
    # a later query reads the same token store instead of tokenizing again
    tokenized = store.tokens.path

    later = session.sql(
        "SELECT r.stars, r.review FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))")
    second = _run(later, make_executor(truth))
    assert sorted(second.to_rows()) == [(3, "fine one"), (5, "good one")]
    assert later.token_inputs()["r"].tokens.path == tokenized

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
    assert everything.token_inputs()["r"].tokens.path == tokenized
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
    assert not [n for n in result.plan.nodes if isinstance(n, Recombine)]
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
    WHERE topic IN ('a')
"""


def test_projection_pushdown_is_the_only_rewrite_of_a_join():
    session = _pair_session()
    plan = session.sql(JOIN_SQL).logical
    assert read_plan(plan).root.input.kind == "SemanticJoin"

    # only the projection moves
    candidate, = placements(plan)
    assert candidate.rules == ()
    assert read_plan(candidate.plan).root == read_plan(plan).root

    labelled = session.sql(LABELLED_JOIN_SQL).logical
    # the placement rule offers the lifted plan for a full join
    *_, placed = placements(labelled)
    assert kinds(placed.plan)[-4:] == [
        "SemanticJoin", "SemanticClassify", "Filter", "Project"]
    session.close()
