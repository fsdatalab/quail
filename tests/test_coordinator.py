"""Tests for the coordinator's filter/join round splitting, merging, gating, and thinning."""

from quail.runtime.coordinator import (derive_plan_nodes,
                                       filter_round_limit,
                                       filter_round_payloads,
                                       gate_group,
                                       join_group_payloads,
                                       merge_filter_round,
                                       merge_join_round,
                                       stage_for_anchor,
                                       thin_survivors)


def payload():
    return dict(
        model="qwen3-4b-fp8", kv_dtype="bf16", chunk_tokens=1000,
        true_ids=[1], false_ids=[2], pre_ids=[9], limit=None,
        docs={"r": [[i] * (10 + i) for i in range(6)],
              "p": [[i] * 5 for i in range(4)]},
        filters={"r": [[7, 7]]},
        filter_arena_writes={"r": True},
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
    limited = merge_filter_round(outs, limit=3)
    assert limited["survivors"]["r"] == [0, 1, 3]


def test_join_group_anchors_follow_filter_shards():
    p = payload()
    survivors = {"r": [0, 1, 3, 4]}     # p unfiltered: survives whole
    subs = join_group_payloads(p, 2, survivors, p["joins"])
    assert subs[0]["anchor_index"] == [0, 4]     # shard 0 minus dead 2
    assert subs[1]["anchor_index"] == [1, 3]
    # every worker sees the identical full partner list
    for s in subs:
        assert s["partners"]["p"]["index"] == [0, 1, 2, 3]
        assert s["partners"]["p"]["docs"][0] == [0] * 5
    assert subs[0]["anchor_docs"][1] == [4] * 14


def test_join_group_reshards_an_anchor_without_filter_shards():
    # a hand-built payload with no shard entry at all for the anchor:
    # shards balance fresh over the live documents. Only token ids
    # move; KV is computed on the new GPU.
    p = payload()
    del p["shards"]["r"]
    survivors = {"r": [0, 1, 3, 5]}
    subs = join_group_payloads(p, 2, survivors, p["joins"])
    covered = sorted(g for s in subs for g in s["anchor_index"])
    assert covered == [0, 1, 3, 5]
    assert all(s["anchor_index"] for s in subs)


def test_join_group_reshards_an_unfiltered_anchor_over_its_live_set():
    # engine payloads ship a scan shard for EVERY alias, but an
    # anchor that ran no filter round (it was a partner before the
    # barrier) has no KV on any GPU to stay near: its scan shard is
    # ignored and shards balance fresh over the live documents - the
    # re-shard.
    p = payload()
    group = [dict(anchor="p", partners=["r"], semantics="full",
                  labels={"r": [2]}, frame=[8], tail=[3])]
    # both live p docs sit in p's first scan shard ((0, 1), (2, 3));
    # following that shard would leave worker 1 with no anchors
    survivors = {"p": [0, 1], "r": [0, 1, 3]}
    subs = join_group_payloads(p, 2, survivors, group)
    assert [s["anchor_index"] for s in subs] == [[0], [1]]
    # a filtered anchor still follows its filter shard (locality):
    # test_join_group_anchors_follow_filter_shards above


def test_join_group_carries_pre_and_store():
    # the join round needs the engine preamble (anchor prefixes are
    # pre + doc) and the store config (anchors restore and save)
    p = payload()
    p["store"] = dict(capacity_bytes=1e9, min_doc_tokens=5,
                      hashes={"r": "h"})
    subs = join_group_payloads(p, 2, {"r": [0, 1, 3, 4]}, p["joins"])
    for s in subs:
        assert s["pre_ids"] == [9]
        assert s["store"]["hashes"]["r"] == "h"
        assert s["anchor_alias"] == "r"


def test_join_group_ships_every_partner_of_a_multi_table_join():
    # a 3-way join: both partner tables replicate to every worker
    p = payload()
    p["docs"]["m"] = [[9] * 3 for _ in range(3)]
    p["joins"] = [dict(anchor="r", partners=["p", "m"],
                       semantics="full",
                       labels={"p": [2], "m": [4]}, frame=[8],
                       tail=[3])]
    subs = join_group_payloads(p, 2, {"r": [0, 1]}, p["joins"])
    for s in subs:
        assert sorted(s["partners"]) == ["m", "p"]
        assert s["partners"]["m"]["index"] == [0, 1, 2]


def test_join_group_two_same_anchor_stages():
    # two full stages sharing one anchor: one round; every worker
    # gets both stages and every partner table of either stage
    p = payload()
    p["docs"]["m"] = [[9] * 3 for _ in range(3)]
    p["joins"].append(dict(anchor="r", partners=["m"],
                           semantics="full", labels={"m": [4]},
                           frame=[8], tail=[3]))
    subs = join_group_payloads(p, 2, {"r": [0, 1, 3, 4]}, p["joins"])
    for s in subs:
        assert len(s["joins"]) == 2
        assert sorted(s["partners"]) == ["m", "p"]
        assert s["partners"]["m"]["index"] == [0, 1, 2]
        assert s["partners"]["p"]["index"] == [0, 1, 2, 3]
    assert subs[0]["anchor_index"] == [0, 4]
    assert subs[1]["anchor_index"] == [1, 3]


def test_stage_for_anchor_materializes_either_side():
    spec = dict(anchor="r", partners=["p"], aliases=["r", "p"],
                semantics="full",
                frames={"r": [70], "p": [71]},
                labels={"r": [80], "p": [81]}, tail=[3])
    r_side = stage_for_anchor(spec, "r")
    assert r_side["anchor"] == "r"
    assert r_side["frame"] == [70]
    assert r_side["labels"] == {"p": [81]}
    assert r_side["partners"] == ["p"]
    p_side = stage_for_anchor(spec, "p")
    assert p_side["anchor"] == "p"
    assert p_side["frame"] == [71]
    assert p_side["labels"] == {"r": [80]}
    assert p_side["partners"] == ["r"]
    # a hand-built spec (no per-table maps) is already materialized
    hand = dict(anchor="r", partners=["p"], labels={"p": [2]},
                frame=[8], tail=[3], semantics="full")
    assert stage_for_anchor(hand, "p") is hand


def test_derive_plan_nodes_groups_and_barriers():
    # the executor's grouping rule, reconstructed for payloads built
    # without a planner
    j1 = dict(anchor="p", semantics="full")
    j2 = dict(anchor="p", semantics="full")
    nodes = derive_plan_nodes([j1, j2])
    assert [n["op"] for n in nodes] == ["JoinGroup"]
    assert nodes[0]["stage_idxs"] == (0, 1)

    j3 = dict(anchor="r", semantics="full")
    nodes = derive_plan_nodes([j1, j3])
    assert [n["op"] for n in nodes] == ["JoinGroup", "Barrier",
                                        "JoinGroup"]
    assert nodes[1]["next_anchor"] == "r"
    assert nodes[2]["stage_idxs"] == (1,)

    # a gate runs alone (its keep rule differs from the in-call
    # gate); same anchor, so no barrier between the three groups
    anti = dict(anchor="p", semantics="anti")
    nodes = derive_plan_nodes([j1, anti, j2])
    assert [n["op"] for n in nodes] == ["JoinGroup"] * 3
    assert [n["stage_idxs"] for n in nodes] == [(0,), (1,), (2,)]


def test_gate_group_full_exists_anti():
    out = dict(rows={0: [1, 0], 1: [0, 0], 2: [0, 1]},
               anchor_index=[5, 7, 9])
    assert gate_group(out, "full") == [5, 9]
    assert gate_group(out, "exists") == [5, 9]
    assert gate_group(out, "anti") == [7]
    # an anchor gated out mid-group has no last-stage row: dropped
    partial = dict(rows={0: [1]}, anchor_index=[5, 7])
    assert gate_group(partial, "full") == [5]


def test_thin_survivors_keeps_only_surviving_pair_members():
    # stage anchored on r over partner p. r0 matched p1; r2 answered
    # all NO; r4 matched p2 but was gated later (not in survivors),
    # so its pair keeps nothing alive.
    out = dict(anchor="r", partners=["p"],
               rows={0: [0, 1, 0], 1: [0, 0, 0], 2: [0, 0, 1]},
               anchor_index=[0, 2, 4],
               partner_index=[[0], [1], [2]])
    survivors = {"r": [0, 2], "p": [0, 1, 2, 3]}
    thin_survivors([out], survivors)
    assert survivors["r"] == [0]
    assert survivors["p"] == [1]


def test_thin_survivors_intersects_across_stages():
    # two finished stages touching p: a p document must appear in a
    # surviving pair of BOTH to stay live
    s1 = dict(anchor="r", partners=["p"],
              rows={0: [1, 1, 0, 0]}, anchor_index=[0],
              partner_index=[[0], [1], [2], [3]])
    s2 = dict(anchor="g", partners=["p"],
              rows={0: [0, 1, 1, 0]}, anchor_index=[0],
              partner_index=[[0], [1], [2], [3]])
    survivors = {"r": [0], "g": [0], "p": [0, 1, 2, 3]}
    thin_survivors([s1, s2], survivors)
    assert survivors["p"] == [1]


def test_filter_round_limit_rule():
    # LIMIT counts output rows. Filter-only: one survivor is one row,
    # so the filter round may stop early. With joins, cutting
    # survivor lists drops output rows (#39), so the round gets None.
    p = payload()
    p["limit"] = 3
    assert filter_round_limit(p) is None      # payload has joins
    assert all(s["limit"] is None for s in filter_round_payloads(
        p, p["shards"], 2))
    p["joins"] = []
    assert filter_round_limit(p) == 3
    assert all(s["limit"] == 3 for s in filter_round_payloads(
        p, p["shards"], 2))
    p["limit"] = None
    assert filter_round_limit(p) is None


def test_merge_join_round_two_stages():
    # stage 2 rows exist only for anchors the stage-1 gate kept; the
    # merge keeps the stages aligned and the anchors disjoint
    outs = [
        dict(joins=[dict(rows={0: [1, 0], 1: [0, 0]},
                         anchor_index=[0, 4],
                         partner_index=[[1], [2]]),
                    dict(rows={0: [0, 1, 1]},
                         anchor_index=[0, 4],
                         partner_index=[[0], [1], [2]])],
             fresh_tokens=10, wall_s=1.0),
        dict(joins=[dict(rows={0: [1, 1]},
                         anchor_index=[3],
                         partner_index=[[1], [2]]),
                    dict(rows={0: [1, 0, 0]},
                         anchor_index=[3],
                         partner_index=[[0], [1], [2]])],
             fresh_tokens=10, wall_s=1.0),
    ]
    merged = merge_join_round(outs)
    assert len(merged) == 2
    assert merged[0]["anchor_index"] == [0, 4, 3]
    assert merged[0]["rows"] == {0: [1, 0], 1: [0, 0], 2: [1, 1]}
    assert merged[1]["anchor_index"] == [0, 4, 3]
    assert merged[1]["rows"] == {0: [0, 1, 1], 2: [1, 0, 0]}
    assert merged[1]["partner_index"] == [[0], [1], [2]]


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
