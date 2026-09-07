"""Projection pushdown rule tests."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_session import _run, fake_tok, make_executor

import quail
from quail.logical import (
    ColumnRef,
    CompileError,
    FilterPredicate,
    LogicalPlan,
    Project,
    Scan,
    SemanticFilter,
    SemanticJoin,
    bind_join_prompt,
    bind_prompt,
)
from quail.logical_optimizer import (
    LogicalPlanningContext,
    apply_logical_rules,
)
from quail.logical_rules import ProjectionPushdown, push_down_projection
from quail.planner.plan import EngineConfig

CONTEXT = LogicalPlanningContext(catalog=None, engine_config=None)


def _joined_plan(columns):
    r = Scan("reviews", "r", "review")
    p = Scan("products", "p", "description")
    filtered = SemanticFilter(r, (FilterPredicate(bind_prompt(
        "q {0}", (ColumnRef("r", "reviews", "review"),))),))
    joined = SemanticJoin(
        (filtered, p),
        bind_join_prompt("same {0} {1}", (
            ColumnRef("r", "reviews", "review"),
            ColumnRef("p", "products", "description"),
        )),
    )
    return LogicalPlan(Project(joined, tuple(columns)))


def _scans(plan):
    return {node.alias: node for node in plan.walk()
            if isinstance(node, Scan)}


def test_rule_keeps_only_returned_columns_on_each_scan():
    plan = _joined_plan([
        ColumnRef("r", "reviews", "id"),
        ColumnRef("p", "products", "asin"),
        ColumnRef("r", "reviews", "stars"),
    ])

    optimized, changed = apply_logical_rules(
        plan, (ProjectionPushdown(),), CONTEXT)

    assert changed == ("projection_pushdown",)
    scans = _scans(optimized)
    assert scans["r"].columns == ("id", "stars")
    assert scans["p"].columns == ("asin",)
    # the prompt reads the document column as tokens, so it is not a
    # stored value
    assert scans["r"].column == "review"
    assert scans["r"].output_schema() == (
        ColumnRef("r", "reviews", "review"),
        ColumnRef("r", "reviews", "id"),
        ColumnRef("r", "reviews", "stars"),
    )
    assert optimized.root.input.output_schema() == (
        ColumnRef("r", "reviews", "review"),
        ColumnRef("r", "reviews", "id"),
        ColumnRef("r", "reviews", "stars"),
        ColumnRef("p", "products", "description"),
        ColumnRef("p", "products", "asin"),
    )


def test_rule_keeps_the_document_column_when_the_query_returns_it():
    plan = _joined_plan([
        ColumnRef("r", "reviews", "review"),
        ColumnRef("p", "products", "asin"),
    ])

    optimized, _ = apply_logical_rules(
        plan, (ProjectionPushdown(),), CONTEXT)

    scans = _scans(optimized)
    assert scans["r"].columns == ("review",)
    assert scans["p"].columns == ("asin",)
    assert scans["r"].output_schema() == (
        ColumnRef("r", "reviews", "review"),)


def test_rule_is_idempotent_and_leaves_other_nodes_alone():
    plan = _joined_plan([ColumnRef("r", "reviews", "id")])
    once = push_down_projection(plan.root)
    twice = push_down_projection(once)

    assert twice == once
    assert twice is once
    assert ProjectionPushdown().rewrite(plan.root.input, CONTEXT) is None
    assert ProjectionPushdown().rewrite(once, CONTEXT) == once
    _, changed = apply_logical_rules(
        LogicalPlan(once), (ProjectionPushdown(),), CONTEXT)
    assert changed == ()


def test_scan_explains_its_columns_and_rejects_duplicates():
    scan = Scan("reviews", "r", "review", columns=("id", "stars"))
    assert scan.explain_fields()["columns"] == ["id", "stars"]
    with pytest.raises(CompileError, match="twice"):
        Scan("reviews", "r", "review", columns=("id", "id")).validate()


def _session(tmp_path):
    session = quail.Session(EngineConfig(gpus=1), tokenizer=fake_tok)
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


def test_session_loads_only_the_columns_the_query_returns(tmp_path):
    session = _session(tmp_path)
    truth = {"r": {"q:": [1, 0, 1]}}
    query = session.sql(
        "SELECT r.stars, r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))")

    result = _run(query, make_executor(truth))

    assert sorted(result.to_rows()) == [(3, "r2"), (5, "r0")]
    scan = _scans(query.logical)["r"]
    assert scan.columns == ("stars", "id")
    store = query._token_inputs["r"]
    assert store.projected_columns == ("stars", "id")


def test_session_returns_the_document_text_and_star(tmp_path):
    session = _session(tmp_path)
    truth = {"r": {"q:": [0, 1, 0]}}

    text = _run(session.sql(
        "SELECT r.review FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))"), make_executor(truth))
    assert text.to_rows() == [("bad one",)]

    star = _run(session.sql(
        "SELECT * FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))"), make_executor(truth))
    assert star.columns == ["r.id", "r.review", "r.stars", "r.wide"]
    assert star.to_rows() == [("r1", "bad one", 1, "y" * 50)]


def test_second_query_with_other_columns_does_not_tokenize_again(tmp_path):
    reviews = ["good one", "bad one", "fine one"]
    calls = []

    def counting_tok(text):
        # prompt parts are tokenized on every query; count documents only
        if text in reviews:
            calls.append(text)
        return fake_tok(text)

    session = quail.Session(EngineConfig(gpus=1), tokenizer=counting_tok)
    path = tmp_path / "reviews.parquet"
    pq.write_table(pa.table({
        "id": ["r0", "r1", "r2"],
        "review": reviews,
        "stars": [5, 1, 3],
    }), str(path))
    session.register("reviews", quail.DocumentProvider.from_parquet(
        str(path), id_col="id"))
    truth = {"r": {"q:": [1, 0, 1]}}

    first = _run(session.sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))"), make_executor(truth))
    # the first query tokenizes the 25 document sample once to pick the
    # token type and once more when writing the file
    tokenized = len(calls)
    assert tokenized > 0
    second = _run(session.sql(
        "SELECT r.stars, r.review FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))"), make_executor(truth))

    assert sorted(first.to_rows()) == [("r0",), ("r2",)]
    assert sorted(second.to_rows()) == [(3, "fine one"), (5, "good one")]
    # the tokenizer sample and the token file are written once; the
    # second query only copies the two new value columns
    assert len(calls) == tokenized
    assert len(session._token_stores) == 1
    assert sorted(name for _, name in session._column_stores) == [
        "id", "review", "stars"]
