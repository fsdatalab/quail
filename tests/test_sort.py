"""ORDER BY, OFFSET, and DISTINCT: the Sort node, its runtime, and the limit rule."""

import pyarrow as pa
import pytest
from test_planner import _optimize, _plan, node_kinds, tok

from quail.catalog import Catalog, DocumentProvider
from quail.execution.result import QueryResult
from quail.execution.runner import NodeResult, SortRuntime
from quail.frontend.builder import col, docs, prompt
from quail.logical import Scan
from quail.physical import Project, Sort
from quail.physical.base import input_ports
from quail.planner.decide import explain


@pytest.fixture()
def catalog():
    cat = Catalog()
    cat.register("reviews", DocumentProvider.from_table(
        pa.table({"id": ["x", "y"], "review": ["x", "y"]}), id_col="id"))
    return cat


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
    assert scan.columns == ("review", "id")
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
