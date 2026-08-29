"""Arrow token payload checks."""

import pyarrow as pa

from quail.runtime.tokens import (
    ArrowTokenDocuments,
    chain_tokens,
    decode_token_documents,
    encode_token_documents,
)


def test_arrow_token_payload_round_trip_keeps_document_views():
    tokens = pa.chunked_array([
        pa.array([[1, 2, 3], [], [4, 5]],
                 type=pa.large_list(pa.int32()))
    ])

    encoded = encode_token_documents(tokens)
    documents = decode_token_documents(encoded)

    assert isinstance(documents, ArrowTokenDocuments)
    assert len(documents) == 3
    assert list(documents[0]) == [1, 2, 3]
    assert list(documents[1]) == []
    assert list(documents[2]) == [4, 5]

    values = documents._values.to_numpy(zero_copy_only=True)
    view = documents[2].numpy()
    assert view.__array_interface__["data"][0] == (
        values.__array_interface__["data"][0] + 3 * values.itemsize)
    assert documents[2].arrow_array.offset == 3
    assert hasattr(documents[2].arrow_array, "__dlpack__")


def test_token_chain_keeps_parts_until_chunk_packing():
    tokens = pa.chunked_array([
        pa.array([[10, 11]], type=pa.large_list(pa.int32()))
    ])
    document = decode_token_documents(
        encode_token_documents(tokens))[0]

    combined = chain_tokens([1, 2], document, [20])

    assert len(combined) == 5
    assert list(combined) == [1, 2, 10, 11, 20]
    assert combined.token_parts[1] is document
