"""Arrow query result execution through Acero."""

import pyarrow as pa

from quail.runtime.result import (
    QueryResult,
    build_result_declaration,
    count_rows,
    document_index_table,
)


def test_acero_assembles_multiway_result_without_python_rows():
    true_join_tables = [
        document_index_table(
            {"r1": [0, 1], "a1": [0, 0]}, "join_answers"),
        document_index_table(
            {"r2": [0, 1], "a1": [0, 0]}, "join_answers"),
        document_index_table(
            {"r2": [0, 1, 1], "a2": [0, 0, 1]}, "join_answers"),
    ]
    survivors = {
        "r1": pa.array([0, 1], type=pa.int32()),
        "r2": pa.array([0, 1], type=pa.int32()),
        "a1": pa.array([0], type=pa.int32()),
        "a2": pa.array([0, 1], type=pa.int32()),
    }
    declaration, _ = build_result_declaration(
        true_join_tables, survivors, "r1")

    assert count_rows(declaration) == 6


def test_result_streams_bounded_arrow_batches():
    relation = document_index_table(
        {"r": list(range(5))}, "filter_survivors")
    survivors = {"r": pa.array(range(5), type=pa.int32())}
    declaration, index_schema = build_result_declaration([], survivors, "r")
    values = pa.array([f"r{i}" for i in range(5)])
    schema = pa.schema(
        [pa.field("r.id", pa.string(), nullable=False)],
        metadata={b"quail.kind": b"query_result"},
    )
    result = QueryResult(
        columns=["r.id"],
        declaration=declaration,
        document_index_schema=index_schema,
        output_schema=schema,
        projection=[("r", values)],
        report={},
        survivor_indices=survivors,
        true_join_tables={},
    )

    reader = result.execute_stream(batch_rows=2)
    batches = list(reader)

    assert isinstance(reader, pa.RecordBatchReader)
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert pa.Table.from_batches(batches).column("r.id").to_pylist() == [
        "r0", "r1", "r2", "r3", "r4"]
    assert result.count() == 5
    assert isinstance(result.collect(limit=3), pa.Table)
    assert result.collect(limit=3).column("r.id").to_pylist() == [
        "r0", "r1", "r2"]
    assert not hasattr(result, "rows")
    assert relation.schema.metadata[b"quail.kind"] == b"filter_survivors"


def test_ipc_result_limit_preserves_the_known_row_count(tmp_path):
    table = pa.table({"id": ["a", "b", "c"]})
    path = tmp_path / "result.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    result = QueryResult.from_ipc_file(
        str(path), table.schema, len(table)
    ).with_limit(2)

    assert result.count() == 2
    assert result.collect().to_pydict() == {"id": ["a", "b"]}
