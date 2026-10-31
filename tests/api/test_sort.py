"""ORDER BY, OFFSET, and DISTINCT: the Sort node, its runtime, and the limit rule."""

import pyarrow as pa

from quail.execution.metrics import NodeResult
from quail.execution.result import QueryResult
from quail.execution.runtimes import SortRuntime
from quail.physical import Sort
from quail.physical.base import input_ports


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
