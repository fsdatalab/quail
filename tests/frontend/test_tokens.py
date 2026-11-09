"""Arrow token payloads and shared prefix lengths."""

import pickle

import pyarrow as pa
import pytest

from quail.common.tokens import (
    ArrowTokenDocuments,
    TokenStore,
    concat_tokens,
    decode_token_documents,
    prefix_tree,
)


def _store(documents):
    """A TokenStore of token id lists, as Rust's planner hands one over."""
    return TokenStore(pa.table({
        "__quail_token_ids": pa.array(documents, pa.large_list(pa.int32())),
        "__quail_token_count": pa.array(map(len, documents), pa.int64()),
    }))


def test_token_views_store_concatenations_and_shared_prefixes(monkeypatch):
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
    combined = concat_tokens([1, 2], first, [20])
    assert list(combined) == [1, 2, 1, 2, 3, 20]
    assert combined.token_parts[1] is first

    store = _store([[1, 2, 3], [4, 5], [1, 2, 9]])
    assert list(store.lengths) == [3, 2, 3]
    index = store.prefix_index
    with monkeypatch.context() as patch:
        patch.setattr("quail.common.tokens._token_array",
                      lambda *_: pytest.fail("prefix preparation ran twice"))
        assert prefix_tree(store, 1) is index.tree(1)
        assert prefix_tree(store, 16) is index.tree(16)

    long = _store([list(range(1000))] * 10)
    assert long.prefix_index is not index
    assert prefix_tree(long, 16).shared_tokens == 9 * (999 // 16 * 16)
    restored = pickle.loads(pickle.dumps(long))
    assert len(restored) == 10
    assert list(restored[9]) == list(range(1000))
