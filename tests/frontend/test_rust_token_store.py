"""Rust's token stores read as Python's and give the same statistics."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from helpers.sessions import TOKENIZER, encoder

from quail.common.document_ends import last_tokens
from quail.common.tokens import TokenStore, prefix_tree

native = pytest.importorskip("_quail_native")

# shared prefixes, an exact duplicate, an empty document, and a prefix of
# another document
TEXTS = ["a b c d e f", "a b c d x", "a b c d e f", "", "a b", "z y x w",
         "a b c d e f g h", "z y"]
TOKENIZER_FILE = str(TOKENIZER / "tokenizer.json")


def python_statistics(store: TokenStore, page_tokens: int, columns=()):
    tree = prefix_tree(store, page_tokens)
    shared, count, rows = tree.reader_groups(store.lengths)
    first = (store.first_documents_with(list(columns)) if columns
             else store.first_documents)
    return {
        "order": list(tree.order),
        "parent": list(tree.parent),
        "shared": [int(value) for value in tree.shared],
        "reader_groups": sorted(zip(shared.tolist(), count.tolist(), rows.tolist())),
        "first_documents": first.to_pylist(),
        "last_tokens": last_tokens(store.arrow_array).tolist(),
    }


def rust_statistics(path, page_tokens: int, columns=None):
    found = native.prefix_statistics(str(path), page_tokens, columns)
    found["reader_groups"] = sorted(found["reader_groups"])
    return found


@pytest.mark.parametrize("page_tokens", [1, 2])
def test_a_rust_token_store_reads_in_python_with_the_same_statistics(
        tmp_path, page_tokens):
    path = tmp_path / "tokens.arrow"
    native.write_token_store(str(path), pa.table({"text": TEXTS}), TOKENIZER_FILE)
    store = TokenStore(str(path))
    try:
        assert [list(store[index]) for index in range(len(store))] == [
            encoder()(text) for text in TEXTS]
        assert rust_statistics(path, page_tokens) == python_statistics(
            store, page_tokens)
    finally:
        store.close()


def test_documents_with_equal_tokens_but_other_values_have_their_own_first(tmp_path):
    path = tmp_path / "tokens.arrow"
    native.write_token_store(str(path), pa.table({"text": TEXTS}), TOKENIZER_FILE)
    stars = pa.array([5, 4, 1, 3, 2, 1, 1, 3])
    store = TokenStore(str(path))
    try:
        found = rust_statistics(path, 1, pa.table({"stars": stars}))
        assert found["first_documents"] == python_statistics(
            store, 1, [stars])["first_documents"]
        # documents 0 and 2 have the same tokens but different stars
        assert found["first_documents"][2] == 2
    finally:
        store.close()


def test_a_model_tokenizer_in_rust_gives_gigatokens_ids_and_bytes(tmp_path):
    from gigatoken import Tokenizer as Gigatoken

    from quail.builtins import built_in_registry
    from quail.common.tokens import Tokenizer
    from quail.specs.files import resolve_model_path

    model = built_in_registry().model("qwen3-4b-fp8")
    try:
        directory = resolve_model_path(model.hf_name, model.revision,
                                       include_weights=False)
    except Exception as error:  # noqa: BLE001 - offline machines skip
        pytest.skip(f"the model's tokenizer is not available: {error}")
    texts = ["Lease ends in 2027.", "A moving ending — and a café.", "", "  spaced  "]
    path = tmp_path / "tokens.arrow"
    native.write_token_store(str(path), pa.table({"text": texts}),
                             str(Path(directory) / "tokenizer.json"))
    store = TokenStore(str(path))
    try:
        reference = Gigatoken(directory)
        found = [list(np.asarray(store[index])) for index in range(len(store))]
        assert found == [list(reference.encode(text)) for text in texts]
    finally:
        store.close()
    tokenizer = Tokenizer(directory)
    assert dict(enumerate(tokenizer.token_bytes())) == dict(reference.vocab)
    assert [tokenizer.decode(ids).decode() for ids in found] == texts
