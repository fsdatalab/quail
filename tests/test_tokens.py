"""Arrow token payload checks."""

import pickle

import pyarrow as pa

from quail.execution.tokens import (
    ArrowTokenDocuments,
    TokenStore,
    chain_tokens,
    decode_token_documents,
)


def test_token_views_chains_and_shared_prefixes():
    tokens = pa.chunked_array([
        pa.array([[1, 2, 3], [], [4, 5]],
                 type=pa.large_list(pa.int32()))
    ])

    documents = decode_token_documents(tokens)

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

    tokens = pa.chunked_array([
        pa.array([[10, 11]], type=pa.large_list(pa.int32()))
    ])
    document = decode_token_documents(tokens)[0]

    combined = chain_tokens([1, 2], document, [20])

    assert len(combined) == 5
    assert list(combined) == [1, 2, 10, 11, 20]
    assert combined.token_parts[1] is document

    from quail.execution.tokens import shared_prefix_lengths

    sequences = [
        [1, 2, 3, 4],       # shares [1, 2, 3] with the next
        [1, 2, 3, 9, 9],
        [1, 2],             # a prefix of both
        [7, 8],             # shares nothing
        [7, 8],             # identical to the previous
    ]
    credits = shared_prefix_lengths(sequences)

    # 13 tokens in total; the prefix trie has 4 + 2 + 2 = 8 nodes
    assert sum(len(sequence) for sequence in sequences) - sum(credits) == 8
    # the credit for a sequence never exceeds its own length
    assert all(credit <= len(sequence)
               for credit, sequence in zip(credits, sequences))
    assert shared_prefix_lengths([]) == []
    # any order gives the same saving
    assert sum(shared_prefix_lengths(sequences[::-1])) == sum(credits)


def test_token_store_and_file_reference_transport(tmp_path):
    schema = pa.schema({"id": pa.string(), "body": pa.string()})
    batches = [
        pa.record_batch(
            [["a", "b"], ["one two", "three"]],
            schema=schema,
        ),
        pa.record_batch([["c"], ["four five six"]], schema=schema),
    ]
    path = tmp_path / "tokens.arrow"

    store = TokenStore.write(
        str(path),
        batches,
        document_column="body",
        tokenizer=str.split,
        token_type=pa.string(),
    )

    assert path.exists()
    assert list(store.lengths) == [2, 1, 3]
    assert [list(store[index]) for index in range(len(store))] == [
        ["one", "two"],
        ["three"],
        ["four", "five", "six"],
    ]
    assert store._reader.schema.names == [
        "__quail_token_ids", "__quail_token_count"]
    store.close()

    schema = pa.schema({"body": pa.string()})
    path = tmp_path / "tokens.arrow"
    store = TokenStore.write(
        str(path),
        [pa.record_batch([["word " * 1000] * 10], schema=schema)],
        document_column="body",
        tokenizer=str.split,
        token_type=pa.string(),
    )

    selection = store.select(range(10))
    encoded = pickle.dumps(selection)
    restored = pickle.loads(encoded)

    assert len(encoded) < 1000
    assert len(restored) == 10
    assert len(restored[0]) == 1000
