"""ORDER BY, OFFSET, and DISTINCT: the Sort node, its runtime, and the limit rule."""

import pyarrow as pa
import pytest
from test_planner import _optimize, _plan, node_kinds, tok

from quail.catalog import Catalog, DocumentProvider
from quail.execution.result import QueryResult
from quail.execution.runner import NodeResult, SortRuntime
from quail.frontend.builder import col, count, docs, prompt
from quail.logical import Result, Scan, SemanticFilter
from quail.physical import AiFilter, Project, Sort
from quail.physical.base import input_ports
from quail.planner import plan_query
from quail.planner.decide import explain
from quail.planner.logical_optimizer import (
    LogicalPlanningContext,
    apply_logical_rules,
)
from quail.planner.logical_rules import built_in_logical_rules
from quail.specs import H100_SXM, QWEN3_4B_FP8


@pytest.fixture()
def catalog():
    cat = Catalog()
    cat.register("reviews", DocumentProvider.from_table(
        pa.table({"id": ["x", "y"], "review": ["x", "y"],
                  "author": ["a", "a"]}), id_col="id"))
    return cat


def _optimize_with(catalog, logical, tokens):
    """Run the built in logical rules with the catalog's id columns."""
    context = LogicalPlanningContext(
        catalog, None, model=QWEN3_4B_FP8, device=H100_SXM,
        document_tokens=tokens, tokenizer=tok)
    return apply_logical_rules(logical, built_in_logical_rules(), context)


def _sort(**fields):
    return Sort(node_id="sort", inputs=input_ports(()), **fields)


def _rows(node, table):
    result = SortRuntime().execute(node, {"rows": table}, None)
    assert isinstance(result, NodeResult)
    value = result.outputs["rows"]
    assert isinstance(value, QueryResult)
    return value.to_rows(), result.metrics


def test_sort_runtime_orders_bounds_and_deduplicates_rows():
    table = pa.table({
        "r.id": ["a", "b", "c", "d", "e"],
        "score": [0.2, None, 0.9, 0.2, 0.5],
        "label": pa.array(["x", "y", "x", "x", "y"]).dictionary_encode(),
    })
    rows, metrics = _rows(_sort(keys=(("score", True, False),),
                                columns=("r.id",)), table)
    assert rows == [("c",), ("e",), ("a",), ("d",), ("b",)]
    assert (metrics.input_rows, metrics.output_rows) == (5, 5)
    rows, _ = _rows(_sort(keys=(("score", True, True),),
                          columns=("r.id", "score"), offset=1, fetch=2), table)
    assert rows == [("c", 0.9), ("e", 0.5)]
    rows, _ = _rows(_sort(keys=(("score", False, False), ("r.id", True, False)),
                          columns=("r.id",)), table)
    assert rows == [("d",), ("a",), ("e",), ("c",), ("b",)]
    rows, _ = _rows(_sort(keys=(("label", False, False), ("score", True, True)),
                          columns=("r.id",)), table)
    assert rows == [("c",), ("a",), ("d",), ("b",), ("e",)]
    rows, metrics = _rows(_sort(keys=(("label", True, False),),
                                columns=("label",), distinct=True), table)
    assert rows == [("y",), ("x",)]
    assert (metrics.input_rows, metrics.output_rows) == (5, 2)
    rows, _ = _rows(_sort(columns=("label", "score"), distinct=True), table)
    assert rows == [("x", 0.2), ("y", None), ("x", 0.9), ("y", 0.5)]
    rows, _ = _rows(_sort(columns=("r.id",), offset=4), table)
    assert rows == [("e",)]
    materialized = QueryResult.from_table(table)
    result = SortRuntime().execute(
        _sort(keys=(("r.id", True, False),), columns=("r.id",), fetch=1),
        {"rows": materialized}, None)
    assert result.outputs["rows"].to_rows() == [("e",)]


def _filtered(catalog):
    return (docs(catalog, "reviews", tok).alias("r")
            .ai_filter(prompt("flag: {0}", col("r.review")), selectivity=0.5))


def test_order_by_ends_the_plan_in_a_sort_and_keeps_every_survivor(catalog):
    tokens = {"r": [400] * 100}
    plain = _plan(_filtered(catalog).limit(3).select("r.id"), tokens)
    assert node_kinds(plain)[-2:] == ["Project", "Limit"]
    assert plain.settings["filter_limit"] == 3

    logical = (_filtered(catalog).order_by(col("r.id").desc()).limit(3)
               .select("r.review"))
    optimized, _ = _optimize(logical, tokens)
    scan = next(node for node in optimized.walk() if isinstance(node, Scan))
    assert scan.columns == ("id", "review")
    plan = _plan(logical, tokens)
    assert node_kinds(plan)[-2:] == ["Project", "Sort"]
    assert "filter_limit" not in plan.settings
    project = next(node for node in plan.nodes if isinstance(node, Project))
    sort = next(node for node in plan.nodes if isinstance(node, Sort))
    assert project.columns == ("r.review", "r.id")
    assert sort.columns == ("r.review",)
    assert (sort.keys, sort.fetch) == ((("r.id", True, True),), 3)
    text = explain(logical, plan)
    assert "Sort: r.id DESC NULLS FIRST" in text and "Limit: 3" in text
    assert "Sort: r.id DESC" in text.split("physical:", 1)[1]
    assert "Sort: r.id DESC NULLS FIRST" in text.split("physical:", 1)[1]
    assert "fetch=3" in text

    distinct = _plan(_filtered(catalog).distinct().offset(2).select("r.id"),
                     tokens)
    sort = next(node for node in distinct.nodes if isinstance(node, Sort))
    assert (sort.keys, sort.distinct, sort.offset, sort.fetch) == (
        (), True, 2, None)
    assert "filter_limit" not in distinct.settings


def test_distinct_is_dropped_when_every_tables_id_is_selected(catalog):
    tokens = {"r": [400] * 100}
    optimized, changed = _optimize_with(
        catalog, _filtered(catalog).distinct().limit(3).select("r.id"), tokens)
    assert "distinct_elimination" in changed
    assert not isinstance(optimized.root, Result) or not optimized.root.distinct
    plan = plan_query(optimized, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=tokens)
    assert node_kinds(plan)[-2:] == ["Project", "Limit"]
    assert plan.settings["filter_limit"] == 3
    # a group's keys name it once, so DISTINCT over them is redundant too
    grouped = (_filtered(catalog).group_by("r.author").agg(n=count())
               .distinct().select("r.author", "n"))
    optimized, changed = _optimize_with(catalog, grouped, tokens)
    assert "distinct_elimination" in changed and not optimized.root.distinct
    # without the id the rows can repeat, so the DISTINCT stays
    optimized, changed = _optimize_with(
        catalog, _filtered(catalog).distinct().select("r.author"), tokens)
    assert "distinct_elimination" not in changed and optimized.root.distinct
    # without a catalog the rule cannot know the id columns
    optimized, changed = _optimize(
        _filtered(catalog).distinct().select("r.id"), tokens)
    assert "distinct_elimination" not in changed and optimized.root.distinct


def test_distinct_over_one_tables_columns_stops_its_filter_per_key(catalog):
    tokens = {"r": [400] * 100}
    logical = (_filtered(catalog).distinct().order_by("r.author").limit(2)
               .select("r.author"))
    optimized, changed = _optimize_with(catalog, logical, tokens)
    assert "per_key_stop" in changed
    node = next(node for node in optimized.walk()
                if isinstance(node, SemanticFilter))
    assert node.stop_key == ("author",)
    plan = plan_query(optimized, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=tokens)
    ai_filter = next(node for node in plan.nodes if isinstance(node, AiFilter))
    assert ai_filter.stop_key == ("author",)
    text = explain(optimized, plan)
    assert "SemanticFilter stop_key=author" in text
    assert "stop per key: author" in text
    # a second model call, or a column the filter's rows do not determine,
    # leaves the filter alone
    for logical in (
            _filtered(catalog).ai_classify(
                prompt("tone: {0}", col("r.review")), ["pos", "neg"],
                name="tone").distinct().select("r.author", "tone"),
            _filtered(catalog).distinct().select("r.id", "r.author"),
            _filtered(catalog).select("r.author")):
        optimized, changed = _optimize_with(catalog, logical, tokens)
        assert "per_key_stop" not in changed
        assert all(not node.stop_key for node in optimized.walk()
                   if isinstance(node, SemanticFilter))


def test_a_stop_key_filter_reads_one_document_per_key_until_one_passes():
    from test_session import CONFIG, _run, fake_tok, make_executor

    import quail

    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("snapshots", quail.DocumentProvider.from_table(
            pa.table({
                "snapshot_id": [f"s{i}" for i in range(6)],
                "text": [f"snap {i} " + "pad " * 10 for i in range(6)],
                "trajectory_id": ["t1", "t1", "t1", "t2", "t2", "t3"],
            }), id_col="snapshot_id"))
        query = session.sql(
            "SELECT DISTINCT s.trajectory_id FROM snapshots s "
            "WHERE AI.IF(PROMPT('fix: {0}', s.text))", dialect="bq")
        truth = {"s": {"fix": [False, True, True, True, True, False]}}
        result = _run(query, make_executor(truth))
    assert result.to_rows() == [("t1",), ("t2",)]
    answers = result.answer_tables["filters"][("s", 0)]
    assert answers.column("s").to_pylist() == [0, 1, 3, 5]
    assert answers.column("answer").to_pylist() == [False, True, True, False]
