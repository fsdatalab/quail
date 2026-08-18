"""The coordinator's split and merge arithmetic: what the container
parent runs between its GPU children. Pure CPU."""

import pytest

from quail.runtime.coordinator import (filter_round_payloads,
                                       join_round_payloads,
                                       merge_filter_round,
                                       merge_join_round)


def payload():
    return dict(
        model="qwen3-4b-fp8", kv_dtype="bf16", chunk_tokens=1000,
        yes_ids=[1], no_ids=[2],
        docs={"r": [[i] * (10 + i) for i in range(6)],
              "p": [[i] * 5 for i in range(4)]},
        filters={"r": [[7, 7]]},
        joins=[dict(anchor="r", partner="p", semantics="full",
                    pre=[1], mid=[2], tail=[3], swapped=False)],
        store=None, workers=2,
        shards={"r": ((0, 2, 4), (1, 3, 5)), "p": ((0, 1), (2, 3))})


def test_filter_round_split():
    subs = filter_round_payloads(payload(), payload()["shards"], 2)
    assert len(subs) == 2
    assert subs[0]["doc_index"]["r"] == [0, 2, 4]
    assert subs[1]["doc_index"]["r"] == [1, 3, 5]
    # documents follow their indices
    assert subs[1]["docs"]["r"][0] == [1] * 11
    # an unfiltered alias ships NO documents in round 1: they would
    # cross the pipe twice (here and in the join round) for no work
    assert "p" not in subs[0]["docs"]
    assert subs[0]["model"] == "qwen3-4b-fp8"
    assert subs[0]["filters"] == payload()["filters"]


def test_merge_filter_round():
    outs = [
        dict(filters={"r": {0: [1], 2: [0], 4: [1]}},
             survivors={"r": [0, 4]},
             fresh_tokens=100, store={"r": dict(stored_docs=3)},
             boot_s=1.0, wall_s=2.0, peak_gib=10),
        dict(filters={"r": {1: [1], 3: [1], 5: [0]}},
             survivors={"r": [1, 3]},
             fresh_tokens=50, store={"r": dict(stored_docs=3)},
             boot_s=1.5, wall_s=2.0, peak_gib=11),
    ]
    m = merge_filter_round(outs)
    assert m["filters"]["r"] == {0: [1], 1: [1], 2: [0], 3: [1],
                                 4: [1], 5: [0]}
    assert m["survivors"]["r"] == [0, 1, 3, 4]
    assert m["fresh_tokens"] == 150
    assert m["store"]["r"]["stored_docs"] == 6


def test_join_round_anchors_follow_shards():
    p = payload()
    survivors = {"r": [0, 1, 3, 4]}     # p unfiltered: survives whole
    subs = join_round_payloads(p, p["shards"], 2, survivors)
    assert subs[0]["anchor_index"] == [0, 4]     # shard 0 minus dead 2
    assert subs[1]["anchor_index"] == [1, 3]
    # every worker sees the identical full partner list
    for s in subs:
        assert s["partners"]["p"]["index"] == [0, 1, 2, 3]
        assert s["partners"]["p"]["docs"][0] == [0] * 5
    assert subs[0]["anchor_docs"][1] == [4] * 14


def test_join_round_refuses_mixed_anchors():
    p = payload()
    p["joins"].append(dict(anchor="p", partner="r", semantics="full",
                           pre=[], mid=[], tail=[], swapped=False))
    with pytest.raises(NotImplementedError):
        join_round_payloads(p, p["shards"], 2,
                            {"r": [0], "p": [0]})


def test_merge_join_round_disjoint_anchors():
    outs = [
        dict(joins=[dict(rows={0: [1, 0, 1], 1: [0, 0, 1]},
                         anchor_index=[0, 4],
                         partner_index=[1, 2, 3])],
             fresh_tokens=10, wall_s=1.0),
        dict(joins=[dict(rows={0: [0, 1, 0]},
                         anchor_index=[3],
                         partner_index=[1, 2, 3])],
             fresh_tokens=10, wall_s=1.0),
    ]
    merged = merge_join_round(outs)
    assert len(merged) == 1
    stage = merged[0]
    assert stage["anchor_index"] == [0, 4, 3]
    assert stage["partner_index"] == [1, 2, 3]
    assert stage["rows"] == {0: [1, 0, 1], 1: [0, 0, 1],
                             2: [0, 1, 0]}
