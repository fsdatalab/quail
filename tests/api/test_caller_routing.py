"""Reentrant SQL must fail without clearing the outer query's caller route."""

import threading

import _quail_native
import pyarrow as pa
import pytest

from quail.specs import QWEN3_4B_FP8


def test_reentrant_query_is_rejected_and_outer_route_survives():
    class Engine:
        turn = tuple(QWEN3_4B_FP8.turn)
        layout = QWEN3_4B_FP8.prompt_layout
        rules = None
        calls = 0

        def run_region(self, *_args):
            assert threading.get_ident() == caller
            with pytest.raises(RuntimeError, match="reentrant query"):
                session.sql("SELECT 1")
            self.calls += 1
            return None

    engine = Engine()
    session = _quail_native.NativeSession(engine)
    table = pa.table({"id": [0], "body": ["review"]})
    schema = pa.BufferOutputStream()
    with pa.ipc.new_stream(schema, table.schema):
        pass
    session.register("reviews", schema.getvalue().to_pybytes(), table.select)
    caller = threading.get_ident()
    query = "SELECT r.id FROM reviews r WHERE AI.IF(PROMPT('q: {0}', r.body))"
    session.sql(query)
    session.sql(query)
    assert engine.calls == 2
    assert session.sql("SELECT 1 AS n").column("n").to_pylist() == [1]
