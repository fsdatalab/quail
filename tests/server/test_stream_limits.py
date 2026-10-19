"""A result stream reads only the rows asked for and owns its response."""

import io
from contextlib import nullcontext
from types import SimpleNamespace

import pyarrow as pa
import pytest

from quail.remote import artifacts
from quail.remote.client import ServerClient


@pytest.mark.parametrize("limit", [0, 1, 3, 4])
def test_a_saved_result_is_read_only_to_its_limit(limit, monkeypatch):
    schema = pa.schema([("id", pa.int32())])

    def get_batch(index):
        offset = 2 * index
        if offset < limit:
            return pa.record_batch([[offset, offset + 1]], schema=schema)
        raise RuntimeError("read beyond the requested limit")

    source = SimpleNamespace(schema=schema, num_record_batches=10, get_batch=get_batch)
    monkeypatch.setattr(artifacts.ipc, "open_file", lambda _: nullcontext(source))
    payload = b"".join(artifacts.iter_ipc_stream("unused.arrow", limit))
    with pa.ipc.open_stream(payload) as reader:
        assert reader.read_all().to_pydict() == {"id": list(range(limit))}
    assert artifacts.read_table("unused.arrow", limit=limit).num_rows == limit


def _client_of(payload: bytes):
    response = io.BytesIO(payload)
    client = ServerClient("http://unused")
    client._open = lambda *_args, **_kwargs: response
    return client, response


@pytest.mark.parametrize("read", ["none", "one", "all", "iterate", "context"])
def test_the_response_closes_with_its_reader(read):
    table = pa.table({"id": [1, 2, 3]})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=1)
    client, response = _client_of(sink.getvalue().to_pybytes())
    reader = client.result_batches("example")
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
    else:
        with reader:
            pass
    assert response.closed


def test_an_invalid_or_truncated_stream_closes_its_response():
    client, response = _client_of(b"not an Arrow stream")
    with pytest.raises(pa.ArrowInvalid):
        client.result_batches("example")
    assert response.closed
    table = pa.table({"id": [1, 2, 3]})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    client, response = _client_of(sink.getvalue().to_pybytes()[:-12])
    with pytest.raises(OSError):
        list(client.result_batches("example"))
    assert response.closed
