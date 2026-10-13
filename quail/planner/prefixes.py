"""Shared prefix tokens of a token store, for the speed of light estimate."""

from __future__ import annotations

from quail.cost.budgets import PAGE_TOKENS
from quail.execution.tokens import PrefixTree, prefix_tree

# store path -> its prefix tree; a corpus is measured once per process
# however many queries scan it
_document_tree_cache: dict[str, PrefixTree] = {}
_page_tree_cache: dict[str, PrefixTree] = {}


def _documents(store):
    """Return the token documents of a token store or a scan input."""
    return getattr(store, "tokens", store)


def _document_tree(store) -> PrefixTree:
    """Return the store's prefix tree in single tokens, cached per path."""
    documents = _documents(store)
    path = getattr(documents, "path", None)
    if path is not None and path in _document_tree_cache:
        return _document_tree_cache[path]
    tree = prefix_tree(documents, 1)
    if path is not None:
        _document_tree_cache[path] = tree
    return tree


def document_shared_tokens(store) -> list[int]:
    """Return, per document, the prefix tokens an earlier document also has."""
    return _document_tree(store).shared


def shared_prefix_tokens(store) -> int:
    """Return the prefix tokens a token store's documents share."""
    return sum(document_shared_tokens(store))


def document_token_order(store) -> list[int]:
    """Return the document positions in sorted token order."""
    return list(_document_tree(store).order)


def subset_shared_tokens(store, rows) -> list[int]:
    """Return, per document, the shared prefix tokens when only rows compute.

    A document outside rows shares nothing, and a document in rows
    shares only with earlier documents in rows.
    """
    documents = _documents(store)
    rows = list(rows)
    tree = prefix_tree([documents[row] for row in rows], 1)
    shared = [0] * len(documents)
    for row, tokens in zip(rows, tree.shared):
        shared[row] = int(tokens)
    return shared


def page_tree(store) -> PrefixTree:
    """Return the store's prefix tree in whole KV pages, cached per path."""
    documents = _documents(store)
    path = getattr(documents, "path", None)
    if path is not None and path in _page_tree_cache:
        return _page_tree_cache[path]
    tree = prefix_tree(documents, PAGE_TOKENS)
    if path is not None:
        _page_tree_cache[path] = tree
    return tree
