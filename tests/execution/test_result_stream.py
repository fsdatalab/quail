"""Result streams stop reading when their requested rows are available."""

import subprocess
import sys
from types import SimpleNamespace

import pyarrow as pa
import pytest

from quail.execution.result import QueryResult, bounded_reader
from quail.execution.runtimes.relational import LimitRuntime
from quail.physical import Limit
from quail.physical.base import input_ports


@pytest.mark.parametrize("value", [
    pa.table({"id": [1, 2, 3]}), pa.record_batch({"id": [1, 2, 3]}),
    pa.array([1, 2, 3]), [1, 2, 3],
])
def test_limit_bounds_are_shared_by_tables_batches_and_sequences(value):
    node = Limit(node_id="limit", inputs=input_ports(()), count=2**80)
    result = LimitRuntime().execute(node, {"rows": value}, None)
    rows = result.outputs["rows"]
    assert rows == value if isinstance(value, list) else rows.equals(value)
    assert result.metrics.input_rows == result.metrics.output_rows == 3


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("batch_rows", [1, 3])
def test_result_limit_does_not_read_another_input_batch(limit, batch_rows):
    schema = pa.schema([("r", pa.int32())])

    def batches():
        for offset in range(0, limit, 2):
            yield pa.record_batch([[]], schema=schema)
            yield pa.record_batch([[offset, offset + 1]], schema=schema)
        raise RuntimeError("read beyond the requested limit")

    declaration = SimpleNamespace(to_reader=lambda **_:
                                  pa.RecordBatchReader.from_batches(schema, batches()))
    result = QueryResult(
        columns=["text"], declaration=declaration, document_index_schema=schema,
        output_schema=pa.schema([("text", pa.string())]),
        projection=[("r", pa.array(["a", "b", "c", "d"]))], report={})

    with result.execute_stream(limit=limit, batch_rows=batch_rows) as reader:
        received = list(reader)
    assert all(len(batch) <= batch_rows for batch in received)
    assert pa.Table.from_batches(received, schema=result.schema).to_pydict() == {
        "text": ["a", "b", "c", "d"][:limit]}


def test_acero_streams_close_on_limits_early_exit_and_errors():
    reader_text = """
import pyarrow as pa
from quail.execution.result import (
    QueryResult, build_result_declaration, document_index_table)

ids = pa.array(range(5), pa.int32())
pairs = document_index_table({"r": ids, "p": ids}, "join_answers")
declaration, schema = build_result_declaration(
    [pairs], {"r": ids, "p": ids}, "r")
result = QueryResult(
    columns=["id"], declaration=declaration, document_index_schema=schema,
    output_schema=pa.schema([("id", pa.int32())]),
    projection=[("r", ids)], report={})
for _ in range(100):
    for limit in (0, 1, 3, 5, 2**80, None):
        table = result.collect(limit=limit, batch_rows=2)
        assert len(table) == (5 if limit is None else min(5, limit))
        assert set(table["id"].to_pylist()) <= set(range(5))
assert result.count() == 5

ids = pa.array(range(200000), pa.int32())
pairs = document_index_table({"r": ids, "p": ids}, "join_answers")
declaration, schema = build_result_declaration(
    [pairs], {"r": ids, "p": ids}, "r")
for projection in ([('r', ids)], [('r', pa.array([], pa.int32()))]):
    result = QueryResult(
        columns=["id"], declaration=declaration, document_index_schema=schema,
        output_schema=pa.schema([("id", pa.int32())]),
        projection=projection, report={})
    for _ in range(40):
        with result.execute_stream(batch_rows=2):
            pass
        with result.execute_stream(batch_rows=2) as reader:
            if len(projection[0][1]):
                assert next(reader).num_rows == 2
            else:
                try:
                    next(reader)
                except pa.ArrowException as error:
                    assert "out of bounds" in str(error)
                else:
                    raise AssertionError("projection should fail")
"""
    subprocess.run([sys.executable, "-c", reader_text],
                   check=True, capture_output=True, text=True, timeout=15)


def test_materialized_result_limit_can_exceed_arrow_index_range():
    table = pa.table({"id": [1, 2, 3]})
    assert QueryResult.from_table(table).collect(limit=2**80).equals(table)


@pytest.mark.parametrize("rows", [0, 1, 5])
@pytest.mark.parametrize("limit", [None, 0, 2])
def test_python_rows_preserve_empty_projection_and_duplicate_column_names(rows, limit):
    table = pa.Table.from_arrays([pa.array(range(rows)), pa.array(range(rows))],
                                 names=["id", "id"])
    count = rows if limit is None else min(rows, limit)
    assert QueryResult.from_table(table).to_rows(limit=limit) == [
        (index, index) for index in range(count)]
    result = QueryResult.from_table(table.select([]))
    assert result.collect(limit=limit).num_rows == count
    assert result.to_rows(limit=limit) == [()] * count


@pytest.mark.parametrize("read", [
    "none", "one", "all", "limit", "context", "export", "read_error",
    "projection_error",
])
def test_result_stream_owns_and_closes_its_source(read):
    schema = pa.schema([("r", pa.int32())])

    class Source:
        def __init__(self):
            self.schema = schema
            self.closed = 0
            self.reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def __next__(self):
            if read == "read_error":
                raise OSError("source failed")
            if self.reads == 3:
                raise StopIteration
            self.reads += 1
            return pa.record_batch([[0, 1]], schema=schema)

        def close(self):
            self.closed += 1

    source = Source()
    values = pa.array([] if read == "projection_error" else ["a", "b"], pa.string())
    result = QueryResult(
        columns=["text"],
        declaration=SimpleNamespace(to_reader=lambda **_: source),
        document_index_schema=schema,
        output_schema=pa.schema([("text", pa.string())]),
        projection=[("r", values)], report={})
    reader = result.execute_stream(batch_rows=1, limit=1 if read == "limit" else None)
    assert isinstance(reader, pa.RecordBatchReader)
    if read == "none":
        reader.close()
    elif read == "one":
        assert next(reader).column(0).to_pylist() == ["a"]
        reader.close()
    elif read == "all":
        assert reader.read_all().num_rows == 6
    elif read == "limit":
        assert reader.read_all().num_rows == 1
        assert source.reads == 1
    elif read == "context":
        with reader:
            pass
    elif read == "export":
        transferred = pa.RecordBatchReader.from_stream(reader)
        del reader
        reader = transferred
        assert next(reader).num_rows == 1
        reader.close()
    elif read == "read_error":
        with pytest.raises(OSError, match="source failed"):
            reader.read_all()
    else:
        with pytest.raises(pa.ArrowException, match="Index 0 out of bounds"):
            reader.read_all()
    assert source.closed == 1
    reader.close()
    assert source.closed == 1


def test_stream_transfer_preserves_buffers_and_schema():
    values = pa.array(range(8), pa.int32())
    schema = pa.schema([pa.field("id", pa.int32(), nullable=False)],
                       metadata={b"source": b"test"})
    table = pa.Table.from_arrays([values], schema=schema)
    with bounded_reader(table.to_reader(), limit=3, batch_rows=2) as reader:
        assert reader.schema.equals(schema, check_metadata=True)
        batches = list(reader)
    assert [len(batch) for batch in batches] == [2, 1]
    assert pa.Table.from_batches(batches).column(0).to_pylist() == [0, 1, 2]
    assert all(batch.column(0).buffers()[1].address == values.buffers()[1].address
               for batch in batches)
