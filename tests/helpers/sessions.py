"""Sessions for the end-to-end tests: the test tokenizer and a planted executor."""

from functools import cache
from pathlib import Path

import pyarrow as pa

from helpers.oracle import Oracle, oracle_executor
from quail.api.config import EngineConfig

CONFIG = EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")

TRUTH = {"r": {"q1:": [1, 1, 0, 1, 1, 0], "q2:": [1, 0, 1, 1, 0, 1]}}


# the tests' tokenizer: one token per byte, and one per space and one or
# two letters; see make_tokenizer.py
TOKENIZER = Path(__file__).resolve().parents[1] / "fixtures" / "tokenizer"


@cache
def encoder():
    """Return an encoder over the tests' tokenizer, as a session loads it."""
    from quail.common.tokens import GigatokenEncoder

    return GigatokenEncoder(TOKENIZER)


def _run(query, execute):
    from quail.api.execute import execute_query

    oracle = getattr(execute, "oracle", None)
    if oracle is not None:
        oracle.catalog = query.session.catalog
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
