"""Checks for the worker's KV hit accounting."""

from types import SimpleNamespace

from quail.backends.quail.graph import _join_round_kv


def test_join_round_counts_resident_anchors_as_hits():
    keys = [("r", 0), ("r", 1), ("r", 2)]
    arena = SimpleNamespace(is_resident=lambda key: key == ("r", 0))
    assert _join_round_kv(keys, arena) == dict(hits=1, misses=2)
