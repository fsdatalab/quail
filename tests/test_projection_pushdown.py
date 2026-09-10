"""Projection pushdown rule tests."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_session import _run, fake_tok, make_executor

import quail
from quail.logical import (
    ColumnRef,
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


def test_projection_rule_preserves_schema_and_is_idempotent():
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


def _session(tmp_path, tokenizer=fake_tok):
    session = quail.Session(EngineConfig(gpus=1), tokenizer=tokenizer)
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


def test_projected_results_and_token_reuse(tmp_path):
    reviews = ["good one", "bad one", "fine one"]
    calls = []

    def counting_tok(text):
        # Prompts are tokenized on every query; count documents only.
        if text in reviews:
            calls.append(text)
        return fake_tok(text)

    session = _session(tmp_path, tokenizer=counting_tok)
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
    tokenized = len(calls)
    assert tokenized > 0

    second = _run(session.sql(
        "SELECT r.stars, r.review FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))"), make_executor(truth))
    assert sorted(second.to_rows()) == [(3, "fine one"), (5, "good one")]
    assert len(calls) == tokenized
    assert len(session._token_stores) == 1
    assert sorted(name for _, name in session._column_stores) == [
        "id", "review", "stars"]

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
    assert len(calls) == tokenized
    session.close()


class _UnstableProvider:
    """A provider whose row count changes between scans."""

    id_col = "id"
    columns = ("id", "body", "stars")

    def __init__(self):
        self.scans = 0

    def schema(self):
        return pa.schema({"id": pa.string(), "body": pa.string(),
                          "stars": pa.int64()})

    def content_identity(self):
        return "unstable"

    def statistics(self):
        from quail.catalog import TableStatistics
        return TableStatistics(row_count=3)

    def scan(self, request):
        self.scans += 1
        rows = 3 if self.scans == 1 else 2
        table = pa.table({
            "id": [f"r{i}" for i in range(rows)],
            "body": ["one two"] * rows,
            "stars": list(range(rows)),
        }).select(list(request.columns))
        return table.to_reader()


def test_failed_scans_and_tokenization_leave_no_partial_cache(tmp_path):
    session = quail.Session(EngineConfig(gpus=1), tokenizer=fake_tok)
    provider = _UnstableProvider()
    session.register("docs", provider)
    session.tokenize("docs", "body", ("id",))

    with pytest.raises(RuntimeError, match="stable order"):
        session.tokenize("docs", "body", ("stars",))
    with pytest.raises(RuntimeError, match="stable order"):
        session.tokenize("docs", "body", ("stars",))

    assert ("unstable", "stars") not in session._column_stores
    assert len(list(Path(session._token_directory.name).iterdir())) == 2
    session.close()

    def broken(text):
        raise ValueError("tokenizer failed")

    session = _session(tmp_path)
    session._tok = broken

    with pytest.raises(ValueError, match="tokenizer failed"):
        session.tokenize("reviews", "review", ("id",))

    assert session._token_stores == {}
    assert session._column_stores == {}
    assert session._token_directory is None or not list(
        Path(session._token_directory.name).iterdir())
    session.close()
