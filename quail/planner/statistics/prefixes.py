"""Shared prefix tokens of a token store, for the speed of light estimate."""

from __future__ import annotations

from quail.catalog.tokens import PrefixTree, prefix_tree


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
