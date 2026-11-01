"""Shared prefix tokens of a token store, for the speed of light estimate."""

from __future__ import annotations

import numpy as np

from quail.catalog.tokens import PrefixTree, prefix_tree


def key_documents(context, alias: str):
    """Which of an alias's documents the model is asked about, or None.

    With deduplication, the model reads only the first document of
    each text; its copies get that answer. None means every document is
    read: the table has no copies, or the model is a diffusion model,
    whose answers depend on more than the text. A filter's stop key can
    make a copy its own key document; this ignores that.
    """
    model = context.model
    if model is None or model.canvas_tokens:
        return None
    lengths = context.document_tokens.get(alias)
    store = None if lengths is None else lengths.store
    if store is None:
        return None
    first = store.first_documents.to_numpy()
    keys = first == np.arange(len(first))
    return None if keys.all() else keys


def document_shared_tokens(store) -> list[int]:
    """Return, per document, the prefix tokens an earlier document also has."""
    return prefix_tree(store, 1).shared


def subset_shared_tokens(store, rows) -> list[int]:
    """Return, per document, the shared prefix tokens when only rows compute.

    A document outside rows shares nothing, and a document in rows
    shares only with earlier documents in rows.
    """
    documents = store
    rows = list(rows)
    tree = prefix_tree([documents[row] for row in rows], 1)
    shared = [0] * len(documents)
    for row, tokens in zip(rows, tree.shared):
        shared[row] = int(tokens)
    return shared


def page_tree(store, page_tokens: int) -> PrefixTree:
    """Return the store's prefix tree in whole KV pages, cached on its token store."""
    return prefix_tree(store, page_tokens)
