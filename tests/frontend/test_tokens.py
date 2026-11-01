"""Arrow token payloads and shared prefix lengths."""

import pickle

import pyarrow as pa
import pytest

from quail.catalog.tokens import (
    ArrowTokenDocuments,
    TokenLengths,
    TokenStore,
    concat_tokens,
    decode_token_documents,
    prefix_tree,
)
from quail.planner.graph.shards import contiguous_shards
from quail.planner.statistics.tables import AliasStats, summarize_alias


def test_token_views_store_concatenations_and_shared_prefixes(tmp_path, monkeypatch):
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
    with monkeypatch.context() as patch:
        patch.setattr(type(store.lengths), "__getitem__",
                      lambda *_: pytest.fail("statistics read an Arrow scalar"))
        memo = {}
        summary = summarize_alias(store.lengths, 3, memo=memo)
        assert summary == AliasStats(3, 6, 14, 3, 3, ((1, 1), (2, 1)))
        assert summary.variance == pytest.approx(2 / 3)
        assert contiguous_shards(store.lengths, 1, total=summary.total) == (
            ((0, 3),), [6])
        assert contiguous_shards(store.lengths, 2) == (((0, 2), (2, 3)), [3, 3])
        assert summarize_alias(store.lengths, 3, memo=memo) is summary
        assert summarize_alias(store.lengths, memo=memo).short_counts == ()
        assert summarize_alias(TokenLengths([4]), 3, memo=memo) == AliasStats(
            1, 4, 16, 4, 3)
    assert summarize_alias(TokenLengths(pa.array([], pa.int32())), 3) == AliasStats(
        0, 0, 0, 0, 3)
    assert summarize_alias(TokenLengths([])).variance == 0
    assert summarize_alias(TokenLengths([100_000_000, 100_000_001])).variance == 0.25
    assert summarize_alias(
        TokenLengths(pa.array([100_000], pa.int32()))).squared == 10_000_000_000
    index = store.prefix_index
    with monkeypatch.context() as patch:
        patch.setattr("quail.catalog.tokens._token_array",
                      lambda *_: pytest.fail("prefix preparation ran twice"))
        assert prefix_tree(store, 1) is index.tree(1)
        assert prefix_tree(store, 16) is index.tree(16)
    store.close()

    schema = pa.schema({"body": pa.string()})
    store = TokenStore.write(
        str(path), [pa.record_batch([["word " * 1000] * 10], schema=schema)],
        document_column="body", tokenizer=str.split, token_type=pa.string())
    assert store.prefix_index is not index
    assert prefix_tree(store, 16).shared_tokens == 9 * 992
    encoded = pickle.dumps(store.select(range(10)))
    restored = pickle.loads(encoded)
    assert len(encoded) < 1000
    assert len(restored) == 10
    assert len(restored[0]) == 1000
