"""Sessions for the end-to-end tests: a fake tokenizer and a planted executor."""

import pyarrow as pa

from helpers.oracle import Oracle, oracle_executor
from quail.planner.plan import EngineConfig

CONFIG = EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")

TRUTH = {"r": {"q1:": [1, 1, 0, 1, 1, 0], "q2:": [1, 0, 1, 1, 0, 1]}}


def fake_tok(text):
    return text.split()


def _run(query, execute):
    from quail.api.execute import execute_query

    return execute_query(query, physical_executor=execute)


def make_executor(filter_truth, join_truth=None):
    """Build a physical executor answering from filter and join truth tables."""
    return oracle_executor(Oracle(filters=filter_truth, joins=join_truth))


def check_table(columns) -> pa.Table:
    """Four rows in a check query table's schema, given as Substrait type names."""
    values = {"string": lambda index: f"document {index} " + "pad " * (10 + index),
              "i64": lambda index: index, "i32": lambda index: index,
              "fp64": float, "bool": lambda index: index % 2 == 0}
    return pa.table({
        column: [values.get(kind, values["string"])(index) for index in range(4)]
        for column, kind in columns})
