"""Result readers own their source and stop at the rows asked for."""

import gc

import pyarrow as pa
import pytest

from quail.execution.result import bounded_reader

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


@pytest.mark.parametrize("use", ["limit"])
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
