"""Arrow token payloads and shared prefix lengths."""

import pickle

import pyarrow as pa
import pytest

from quail.common.shards import contiguous_shards
from quail.common.tokens import (
    ArrowTokenDocuments,
    TokenStore,
    concat_tokens,
    decode_token_documents,
    prefix_tree,
)

native = pytest.importorskip("_quail_native")


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

    from helpers.sessions import TOKENIZER

    tokenizer_file = str(TOKENIZER / "tokenizer.json")
    texts = ["one two", "three", "four five six"]
    path = tmp_path / "tokens.arrow"
    native.write_token_store(str(path), pa.table({"body": texts}), tokenizer_file)
    store = TokenStore(str(path))
    assert path.exists()
    lengths = list(store.lengths)
    assert len(lengths) == 3 and all(length > 0 for length in lengths)
    assert store._reader.schema.names == [
        "__quail_token_ids", "__quail_token_count"]
    total = sum(lengths)
    assert contiguous_shards(store.lengths, 1, total=total) == (((0, 3),), [total])
    index = store.prefix_index
    with monkeypatch.context() as patch:
        patch.setattr("quail.common.tokens._token_array",
                      lambda *_: pytest.fail("prefix preparation ran twice"))
        assert prefix_tree(store, 1) is index.tree(1)
        assert prefix_tree(store, 16) is index.tree(16)
    store.close()

    native.write_token_store(str(path), pa.table({"body": ["word " * 1000] * 10}),
                             tokenizer_file)
    store = TokenStore(str(path))
    assert store.prefix_index is not index
    length = len(store[0])
    assert prefix_tree(store, 16).shared_tokens == 9 * (length // 16 * 16)
    encoded = pickle.dumps(store.select(range(10)))
    restored = pickle.loads(encoded)
    assert len(encoded) < 1000
    assert len(restored) == 10
    assert len(restored[0]) == length


def test_a_growing_prefix_tree_fed_in_batches_equals_the_tree_of_all_documents():
    import random

    from quail.common.tokens import GrowingPrefixTree

    for seed in range(100):
        rng = random.Random(seed)
        documents = []
        for _ in range(rng.randrange(2, 30)):
            base = rng.choice(documents) if documents and rng.random() < 0.6 else []
            documents.append(base[:rng.randrange(len(base) + 1)]
                             + [rng.randrange(5) for _ in range(rng.randrange(1, 30))])
        page_tokens = rng.choice([1, 4, 16])
        whole = prefix_tree(documents, page_tokens)
        seen = rng.randrange(1, len(documents))
        tree = GrowingPrefixTree(prefix_tree(documents[:seen], page_tokens))
        while seen < len(documents):
            seen = rng.randrange(seen + 1, len(documents) + 1)
            tree.add(documents[:seen])
        assert (tree.order, tree.parent, tree.shared) == (
            list(whole.order), list(whole.parent), list(whole.shared)), seed


def test_a_growing_prefix_tree_keeps_fixed_documents_and_reports_moved_ones():
    from quail.common.tokens import GrowingPrefixTree

    documents = [[1] * 8 + [2] * 8, [1] * 8 + [3] * 8]
    tree = GrowingPrefixTree(prefix_tree(documents, 4))
    assert tree.parent == [None, 0] and tree.shared == [0, 8]
    # a document between them shares 12 tokens with the second; the
    # second is fixed, so only the newcomer borrows
    documents.append([1] * 8 + [2] * 4 + [3] * 4)
    assert tree.add(documents, fixed={1}) == []
    assert tree.order == [0, 2, 1]
    assert (tree.parent, tree.shared) == ([None, 0, 0], [0, 8, 12])
    # unfixed, the second takes a closer newcomer as its parent
    documents.append([1] * 8 + [3] * 4 + [2] * 4)
    assert tree.add(documents) == [1]
    assert tree.order == [0, 2, 3, 1]
    assert (tree.parent[1], tree.shared[1]) == (3, 12)
