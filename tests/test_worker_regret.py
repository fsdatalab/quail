"""Checks for the worker's KV regret accounting."""

from quail.runtime.quail_graph import _join_round_kv


def test_join_round_classifies_hit_regret_and_first():
    keys = [("r", 0), ("r", 1), ("r", 2)]

    out = _join_round_kv(
        keys, [100, 200, 300],
        owned={("r", 0)},
        seen={("r", 0), ("r", 1)})

    # ("r", 0) is resident: a hit. ("r", 1) was computed earlier but
    # is gone: its 200 prefix tokens are regret. ("r", 2) is a first
    # computation: a miss, but no regret.
    assert out == dict(hits=1, misses=2, regret_tokens=200)


def test_join_round_with_no_anchors_reports_zeroes():
    assert _join_round_kv([], [], owned=set(), seen=set()) == dict(
        hits=0, misses=0, regret_tokens=0)
