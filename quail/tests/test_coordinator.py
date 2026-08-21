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
        true_ids=[1], false_ids=[2], pre_ids=[9], limit=None,
        docs={"r": [[i] * (10 + i) for i in range(6)],
              "p": [[i] * 5 for i in range(4)]},
        filters={"r": [[7, 7]]},
        joins=[dict(anchor="r", partners=["p"], semantics="full",
                    labels={"p": [2]}, frame=[8], tail=[3])],
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


def test_join_round_carries_pre_and_store():
    # the join round needs the engine preamble (anchor prefixes are
    # pre + doc) and the store config (anchors restore and save)
    p = payload()
    p["store"] = dict(capacity_bytes=1e9, min_doc_tokens=5,
                      hashes={"r": "h"})
    subs = join_round_payloads(p, p["shards"], 2, {"r": [0, 1, 3, 4]})
    for s in subs:
        assert s["pre_ids"] == [9]
        assert s["store"]["hashes"]["r"] == "h"
        assert s["anchor_alias"] == "r"


def test_join_round_ships_every_partner_of_a_multi_table_join():
    # a 3-way join: both partner tables replicate to every worker
    p = payload()
    p["docs"]["m"] = [[9] * 3 for _ in range(3)]
    p["joins"] = [dict(anchor="r", partners=["p", "m"],
                       semantics="full",
                       labels={"p": [2], "m": [4]}, frame=[8],
                       tail=[3])]
    subs = join_round_payloads(p, p["shards"], 2, {"r": [0, 1]})
    for s in subs:
        assert sorted(s["partners"]) == ["m", "p"]
        assert s["partners"]["m"]["index"] == [0, 1, 2]


def test_join_round_refuses_mixed_anchors():
    p = payload()
    p["joins"].append(dict(anchor="p", partners=["r"],
                           semantics="exists", labels={"r": []},
                           frame=[], tail=[]))
    with pytest.raises(NotImplementedError):
        join_round_payloads(p, p["shards"], 2,
                            {"r": [0], "p": [0]})


def test_merge_filter_round_limit_truncates():
    outs = [
        dict(filters={"r": {0: [1], 2: [1], 4: [1]}},
             survivors={"r": [0, 2, 4]},
             fresh_tokens=100, store={}),
        dict(filters={"r": {1: [1], 3: [1], 5: [1]}},
             survivors={"r": [1, 3, 5]},
             fresh_tokens=50, store={}),
    ]
    m = merge_filter_round(outs, limit=4)
    assert m["survivors"]["r"] == [0, 1, 2, 3]


def test_merge_filter_round_no_limit():
    outs = [
        dict(filters={"r": {0: [1], 2: [1]}},
             survivors={"r": [0, 2]},
             fresh_tokens=10, store={}),
        dict(filters={"r": {1: [1]}},
             survivors={"r": [1]},
             fresh_tokens=10, store={}),
    ]
    m = merge_filter_round(outs, limit=None)
    assert m["survivors"]["r"] == [0, 1, 2]


def test_filter_round_carries_limit():
    p = payload()
    p["limit"] = 3
    subs = filter_round_payloads(p, p["shards"], 2)
    for s in subs:
        assert s["limit"] == 3


def test_merge_join_round_disjoint_anchors():
    # partner_index entries are index tuples (one global index per
    # partner alias), identical on every worker
    outs = [
        dict(joins=[dict(rows={0: [1, 0, 1], 1: [0, 0, 1]},
                         anchor_index=[0, 4],
                         partner_index=[[1, 0], [2, 0], [3, 1]])],
             fresh_tokens=10, wall_s=1.0),
        dict(joins=[dict(rows={0: [0, 1, 0]},
                         anchor_index=[3],
                         partner_index=[[1, 0], [2, 0], [3, 1]])],
             fresh_tokens=10, wall_s=1.0),
    ]
    merged = merge_join_round(outs)
    assert len(merged) == 1
    stage = merged[0]
    assert stage["anchor_index"] == [0, 4, 3]
    assert stage["partner_index"] == [[1, 0], [2, 0], [3, 1]]
    assert stage["rows"] == {0: [1, 0, 1], 1: [0, 0, 1],
                             2: [0, 1, 0]}
