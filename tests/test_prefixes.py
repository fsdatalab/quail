"""CPU checks for the shared prefix and KV regret accounting."""

from quail.runtime.prefixes import (
    cross_row_cached_tokens,
    distinct_prefix_regret,
    prefix_metrics,
    scanned_aliases,
    scanned_shared_prefix_tokens,
)


def test_prefix_metrics_account_for_repeated_columns_and_cross_row_hits():
    stages = [
        {"op": "filter", "alias": "r", "stage": 0},
        {"op": "join", "anchor": "r", "partners": ["a"]},
        {"op": "join", "anchor": "e", "partners": ["c"]},
    ]
    assert scanned_aliases(stages) == {"r", "e"}
    assert cross_row_cached_tokens({"backend": "quail"}) == 0
    assert cross_row_cached_tokens({
        "backend": "pipelined_vllm",
        "backend_metrics": {"cross_row_cached_tokens": 7},
    }) == 7
    assert cross_row_cached_tokens({"backend": "stock_vllm"}) is None
    assert distinct_prefix_regret(100, 50, 30) == 120
    assert distinct_prefix_regret(100, 50, None) is None

    class Store:
        def __init__(self, documents):
            self.documents = documents

        def __iter__(self):
            return iter(self.documents)

    stores = {
        ("reviews", "text"): Store([[1, 2, 3], [1, 2, 4], [9]]),
        ("aspects", "name"): Store([[5, 5], [6]]),
    }

    def lookup(provider, column):
        return stores[(provider, column)]

    # within reviews two documents share [1, 2]; one alias of aspects
    # shares nothing
    assert scanned_shared_prefix_tokens(
        [("reviews", "text"), ("aspects", "name")], lookup) == 2
    # a second alias of reviews is the same trie again: its 7 tokens
    # are all shared with the first alias
    assert scanned_shared_prefix_tokens(
        [("reviews", "text"), ("reviews", "text")], lookup) == 2 + 7

    class Scan:
        def __init__(self, alias, provider, column):
            self.alias, self.provider, self.column = alias, provider, column

    reviews = [[1, 2, 3], [1, 2, 4]]
    scans = [Scan("r", "reviews", "text"), Scan("s", "reviews", "text")]
    report = {
        "backend": "quail",
        "regret_tokens": 3,
        "stages": [
            {"op": "filter", "alias": "r", "stage": 0},
            {"op": "join", "anchor": "s", "partners": ["r"]},
        ],
    }
    metrics = prefix_metrics(report, scans, {"r": reviews, "s": reviews})
    # [1, 2] is shared inside the column, and the second alias repeats
    # all 6 tokens of the first
    assert metrics == {
        "shared_prefix_tokens": 2 + 6,
        "cross_row_cached_tokens": 0,
        "regret_distinct_tokens": 3 + 8,
    }
