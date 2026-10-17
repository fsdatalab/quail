"""Remote result limits stop reading once the requested rows are available."""

import io
from contextlib import nullcontext
from types import SimpleNamespace

import pyarrow as pa
import pytest

from quail.remote import artifacts
from quail.remote.client import ServerClient


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["stream", "table"])
def test_result_limit_does_not_read_another_input_batch(limit, mode, monkeypatch):
    schema = pa.schema([("id", pa.int32())])

    def get_batch(index):
        offset = 2 * index
        if offset < limit:
            return pa.record_batch([[offset, offset + 1]], schema=schema)
        raise RuntimeError("read beyond the requested limit")

    source = SimpleNamespace(schema=schema, num_record_batches=10, get_batch=get_batch)
    monkeypatch.setattr(artifacts.ipc, "open_file", lambda _: nullcontext(source))
    if mode == "stream":
        payload = b"".join(artifacts.iter_ipc_stream("unused.arrow", limit))
        with pa.ipc.open_stream(payload) as reader:
            table = reader.read_all()
    else:
        table = artifacts.read_table("unused.arrow", limit=limit)
    assert table.to_pydict() == {"id": list(range(limit))}


@pytest.mark.parametrize("read", [
    "none", "one", "all", "iterate", "metadata", "context"])
def test_response_closes_with_its_reader(read):
    table = pa.table({"id": [1, 2, 3]})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=1)
    response = io.BytesIO(sink.getvalue().to_pybytes())
    client = ServerClient("http://unused")
    client._open = lambda *_args, **_kwargs: response
    reader = client.result_batches("example")
    assert isinstance(reader, pa.RecordBatchReader)
    assert not response.closed
    if read == "none":
        reader.close()
    elif read == "one":
        assert next(reader).num_rows == 1
        reader.close()
    elif read == "all":
        assert reader.read_all().equals(table)
    elif read == "iterate":
        assert pa.Table.from_batches(list(reader)).equals(table)
    elif read == "metadata":
        batches = [batch for batch, _ in reader.iter_batches_with_custom_metadata()]
        assert pa.Table.from_batches(batches).equals(table)
    else:
        with reader:
            pass
    assert response.closed
    reader.close()


def test_invalid_stream_closes_the_response():
    response = io.BytesIO(b"not an Arrow stream")
    client = ServerClient("http://unused")
    client._open = lambda *_args, **_kwargs: response
    with pytest.raises(pa.ArrowInvalid):
        client.result_batches("example")
    assert response.closed


@pytest.mark.parametrize("read_all", [False, True])
def test_interrupted_stream_closes_the_response(read_all):
    table = pa.table({"id": [1, 2, 3]})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=1)
    response = io.BytesIO(sink.getvalue().to_pybytes()[:-12])
    client = ServerClient("http://unused")
    client._open = lambda *_args, **_kwargs: response
    reader = client.result_batches("example")
    with pytest.raises(OSError):
        reader.read_all() if read_all else list(reader)
    assert response.closed
