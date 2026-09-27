"""Arrow token payloads and shared prefix credits."""

import pickle

import pyarrow as pa

from quail.execution.tokens import (
    ArrowTokenDocuments,
    TokenStore,
    chain_tokens,
    decode_token_documents,
    shared_prefix_lengths,
)
from quail.planner.prefixes import shared_prefix_tokens


def test_token_views_chains_and_shared_prefixes():
    tokens = pa.chunked_array([
        pa.array([[1, 2, 3], [], [4, 5]], type=pa.large_list(pa.int32()))
    ])
    documents = decode_token_documents(tokens)
    assert isinstance(documents, ArrowTokenDocuments)
    assert [list(document) for document in documents] == [[1, 2, 3], [], [4, 5]]

    values = documents._values.to_numpy(zero_copy_only=True)
    view = documents[2].numpy()
    assert view.__array_interface__["data"][0] == (
        values.__array_interface__["data"][0] + 3 * values.itemsize)
    assert documents[2].arrow_array.offset == 3
    assert hasattr(documents[2].arrow_array, "__dlpack__")

    first = documents[0]
    combined = chain_tokens([1, 2], first, [20])
    assert list(combined) == [1, 2, 1, 2, 3, 20]
    assert combined.token_parts[1] is first

    sequences = [[1, 2, 3, 4], [1, 2, 3, 9, 9], [1, 2], [7, 8], [7, 8]]
    credits = shared_prefix_lengths(sequences)
    # 13 tokens in total; the prefix trie has 4 + 2 + 2 = 8 nodes
    assert sum(len(sequence) for sequence in sequences) - sum(credits) == 8
    assert shared_prefix_lengths([]) == []
    assert sum(shared_prefix_lengths(sequences[::-1])) == sum(credits)
    assert shared_prefix_tokens([[1, 2, 3], [1, 2, 4], [9]]) == 2


def test_token_store_and_file_reference_transport(tmp_path):
    schema = pa.schema({"id": pa.string(), "body": pa.string()})
    batches = [
        pa.record_batch([["a", "b"], ["one two", "three"]], schema=schema),
        pa.record_batch([["c"], ["four five six"]], schema=schema),
    ]
    path = tmp_path / "tokens.arrow"
    store = TokenStore.write(str(path), batches, document_column="body",
                             tokenizer=str.split, token_type=pa.string())
    assert path.exists()
    assert list(store.lengths) == [2, 1, 3]
    assert [list(store[index]) for index in range(len(store))] == [
        ["one", "two"], ["three"], ["four", "five", "six"]]
    assert store._reader.schema.names == [
        "__quail_token_ids", "__quail_token_count"]
    store.close()

    schema = pa.schema({"body": pa.string()})
    store = TokenStore.write(
        str(path), [pa.record_batch([["word " * 1000] * 10], schema=schema)],
        document_column="body", tokenizer=str.split, token_type=pa.string())
    encoded = pickle.dumps(store.select(range(10)))
    restored = pickle.loads(encoded)
    assert len(encoded) < 1000
    assert len(restored) == 10
    assert len(restored[0]) == 1000
