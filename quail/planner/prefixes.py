"""Shared prefix tokens of a token store, for the speed of light estimate."""

from __future__ import annotations

from quail.cost.budgets import PAGE_TOKENS
from quail.execution.tokens import PrefixTree, prefix_tree


def _documents(store):
    """Return the token documents of a token store or a scan input."""
    return getattr(store, "tokens", store)


def document_shared_tokens(store) -> list[int]:
    """Return, per document, the prefix tokens an earlier document also has."""
    return prefix_tree(_documents(store), 1).shared


def page_tree(store) -> PrefixTree:
    """Return the store's prefix tree in whole KV pages, cached on its token store."""
    return prefix_tree(_documents(store), PAGE_TOKENS)


def shared_prefix_tokens(store) -> int:
    """Return the prefix tokens a token store's documents share."""
    return sum(document_shared_tokens(store))
