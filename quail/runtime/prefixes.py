"""Shared prefix tokens and the two KV regrets of one query.

The per document regret counts a document's own prefix recomputed
after an earlier request had computed it. The distinct prefix regret
also counts tokens recomputed although another document's request had
computed the same prefix: with unlimited KV every distinct prefix in
the corpus is computed once.
"""

from __future__ import annotations

from quail.runtime.tokens import shared_prefix_lengths

# store path -> per document prefix credits; a corpus is measured once
# per process however many queries scan it
_prefix_credit_cache: dict[str, list[int]] = {}


def _documents(store):
    """Return the token documents of a token store or a scan input."""
    return getattr(store, "tokens", store)


def prefix_credits(store) -> list[int]:
    """Return, per document, the prefix tokens another document also has."""
    documents = _documents(store)
    path = getattr(documents, "path", None)
    if path is not None and path in _prefix_credit_cache:
        return _prefix_credit_cache[path]
    credits = shared_prefix_lengths([list(document) for document in documents])
    if path is not None:
        _prefix_credit_cache[path] = credits
    return credits


def shared_prefix_tokens(store) -> int:
    """Return the prefix tokens a token store's documents share."""
    return sum(prefix_credits(store))


def store_token_count(store) -> int:
    """Return the total tokens of a token store's documents."""
    return sum(len(document) for document in _documents(store))


def scanned_shared_prefix_tokens(scanned_columns, stores) -> int:
    """Return the shared prefix tokens over every scanned alias's documents.

    Args:
        scanned_columns: One (provider, column) per scanned alias; a
            column scanned under several aliases appears that many
            times.
        stores: Callable from (provider, column) to the token store.

    A column scanned under k aliases is the same prefix trie k times:
    each extra copy shares every token with the first.
    """
    counts = {}
    for key in scanned_columns:
        counts[key] = counts.get(key, 0) + 1
    total = 0
    for key, copies in counts.items():
        store = stores(*key)
        total += shared_prefix_tokens(store)
        total += (copies - 1) * store_token_count(store)
    return total


def scanned_aliases(stages) -> set[str]:
    """Return the aliases whose documents a query computed as prefixes.

    A filtered alias is scanned in full at its first stage. A join
    anchor's documents are prefixes; its partners are suffixes and
    cannot be reused.
    """
    scanned = set()
    for stage in stages:
        if stage.get("op") == "filter":
            scanned.add(stage["alias"])
        elif stage.get("op") == "join":
            scanned.add(stage["anchor"])
    return scanned


def cross_row_cached_tokens(report) -> int | None:
    """Return cached tokens another document's request computed."""
    backend_metrics = report.get("backend_metrics") or {}
    if "cross_row_cached_tokens" in backend_metrics:
        return int(backend_metrics["cross_row_cached_tokens"])
    if report.get("backend") == "quail":
        return 0
    return None


def distinct_prefix_regret(regret_tokens, shared_prefix, cross_row_cached):
    """Return the regret against one computation per distinct prefix."""
    if regret_tokens is None or cross_row_cached is None:
        return None
    return int(regret_tokens) + int(shared_prefix) - int(cross_row_cached)


def prefix_metrics(report: dict, scans, token_inputs) -> dict:
    """Return shared prefix tokens and both regrets for one report.

    Args:
        report: The execution report with its stages, backend, and
            regret_tokens.
        scans: The query's logical scans.
        token_inputs: Alias to token store, one per scan.
    """
    columns = {scan.alias: (scan.provider, scan.column) for scan in scans}
    stores = {columns[scan.alias]: token_inputs[scan.alias] for scan in scans}
    shared = scanned_shared_prefix_tokens(
        [columns[alias] for alias in scanned_aliases(report.get("stages", ()))],
        lambda provider, column: stores[(provider, column)],
    )
    cross_row = cross_row_cached_tokens(report)
    return {
        "shared_prefix_tokens": shared,
        "cross_row_cached_tokens": cross_row,
        "regret_distinct_tokens": distinct_prefix_regret(
            report.get("regret_tokens"), shared, cross_row),
    }
