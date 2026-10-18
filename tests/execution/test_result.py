"""Result readers own their source and stop at the rows asked for."""

import gc

import pyarrow as pa
import pytest

from quail.execution.result import QueryResult, bounded_reader
from quail.execution.runtimes.relational import LimitRuntime, SortRuntime
from quail.physical import Limit, Sort
from quail.physical.base import input_ports

SCHEMA = pa.schema([("r", pa.int32())])


class _Source:
    """A reader of two-row batches that counts how often it is closed."""

    def __init__(self, rows=6):
        self.batches = [pa.record_batch([[i, i + 1]], schema=SCHEMA)
                        for i in range(0, rows, 2)]
        self.closed = 0
        self.schema = SCHEMA

    def __next__(self):
        if not self.batches:
            raise StopIteration
        return self.batches.pop(0)

    def close(self):
        self.closed += 1


@pytest.mark.parametrize("use", ["none", "one", "all", "limit"])
def test_a_bounded_reader_closes_its_source_once(use):
    source = _Source()
    reader = bounded_reader(source, limit=3 if use == "limit" else None)
    if use == "one":
        next(reader)
    elif use in ("all", "limit"):
        assert len(reader.read_all()) == (3 if use == "limit" else 6)
    reader.close()
    del reader
    gc.collect()
    assert source.closed == 1


def test_a_limit_never_reads_a_batch_past_itself():
    source = _Source(rows=4)
    source.batches.append(None)  # reading here would raise a TypeError
    with bounded_reader(source, limit=4, batch_rows=3) as reader:
        batches = list(reader)
    assert [len(batch) for batch in batches] == [2, 2]


def test_row_bounds_survive_integers_beyond_arrow_index_range():
    table = pa.table({"id": [1, 2, 3]})
    result = QueryResult.from_table(table)
    assert result.collect(limit=2**80).num_rows == 3
    limit = Limit(node_id="limit", inputs=input_ports(()), count=2**80)
    for value in (table, table.to_batches()[0], pa.array([1, 2, 3]), [1, 2, 3]):
        result = LimitRuntime().execute(limit, {"rows": value}, None)
        assert result.metrics.output_rows == 3
    sort = Sort(node_id="sort", inputs=input_ports(()), keys=(("id", False, False),),
                columns=("id",), offset=1, fetch=2**80)
    sorted_rows = SortRuntime().execute(sort, {"rows": table}, None).outputs["rows"]
    assert sorted_rows.to_rows() == [(2,), (3,)]


def test_rows_of_a_projection_without_columns_are_empty_tuples():
    result = QueryResult.from_table(pa.table({"id": [1, 2]}).select([]))
    assert result.to_rows() == [(), ()]
