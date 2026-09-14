"""Tests for the coordinator's round splitting, merging, gating, and thinning."""

import pyarrow as pa

from quail.backends.quail.coordinator import (
    filter_node_payloads,
    gate_group,
    join_group_payloads,
    merge_filter_round,
    merge_join_round,
    thin_survivors,
)
from quail.execution.types import join_answer_cells
from quail.physical import AiFilter


def payload():
    return dict(
        model="qwen3-4b-fp8", chunk_tokens=1000,
        true_ids=[1], false_ids=[2], pre_ids=[9], filter_limit=None,
        docs={"r": [[i] * (10 + i) for i in range(6)],
              "p": [[i] * 5 for i in range(4)]},
        filters={"r": [[7, 7]]},
        filter_arena_writes={"r": True},
        physical_plan={},
        joins=[dict(anchor="r", partners=["p"], semantics="full",
                    labels={"p": [2]}, frame=[8], tail=[3])],
        workers=2,
        shards={"r": ((0, 2, 4), (1, 3, 5)), "p": ((0, 1), (2, 3))})


def filter_node():
    return AiFilter(
        node_id="filter:r", alias="r", arena_writes=True,
        keep_kv=True, question_token_ids=((7, 7),),
    )


def test_filter_round_split_merge_and_limit():
    p = payload()
    subs = filter_node_payloads(
        p, filter_node(), p["shards"], 2, has_joins=True
    )
    assert len(subs) == 2
    assert subs[0]["doc_index"]["r"] == [0, 2, 4]
    assert subs[1]["doc_index"]["r"] == [1, 3, 5]
    # documents follow their indices
    assert subs[1]["docs"]["r"][0] == [1] * 11
    # an unfiltered alias ships NO documents in round 1: they would
    # cross the pipe twice (here and in the join round) for no work
    assert "p" not in subs[0]["docs"]
    assert subs[0]["model"] == "qwen3-4b-fp8"
    assert subs[0]["physical_plan"] is p["physical_plan"]
    assert subs[0]["node_id"] == "filter:r"
    assert "filters" not in subs[0]
    assert "filter_arena_writes" not in subs[0]
    assert "retain_aliases" not in subs[0]

    outs = [
        dict(filters={"r": {0: [1], 2: [0], 4: [1]}},
             survivors={"r": [0, 4]},
             fresh_tokens=100,
             boot_s=1.0, wall_s=2.0, peak_gib=10),
        dict(filters={"r": {1: [1], 3: [1], 5: [0]}},
             survivors={"r": [1, 3]},
             fresh_tokens=50,
             boot_s=1.5, wall_s=2.0, peak_gib=11),
    ]
    m = merge_filter_round(outs)
    assert m["filters"]["r"] == {0: [1], 1: [1], 2: [0], 3: [1],
                                 4: [1], 5: [0]}
    assert m["survivors"]["r"] == [0, 1, 3, 4]
    assert m["fresh_tokens"] == 150
    limited = merge_filter_round(outs, limit=3)
    assert limited["survivors"]["r"] == [0, 1, 3]

    # LIMIT counts output rows. Filter-only: one survivor is one row,
    # so the filter round may stop early. With joins, cutting
    # survivor lists drops output rows (#39), so the round gets None.
    p = payload()
    p["filter_limit"] = 3
    assert all(s["filter_limit"] is None for s in filter_node_payloads(
        p, filter_node(), p["shards"], 2, has_joins=True))
    assert all(s["filter_limit"] == 3 for s in filter_node_payloads(
        p, filter_node(), p["shards"], 2, has_joins=False))
    p["filter_limit"] = None
    assert all(s["filter_limit"] is None for s in filter_node_payloads(
        p, filter_node(), p["shards"], 2, has_joins=False))


def test_join_distribution_and_kv_placement():
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

    # an anchor kept by an earlier group stays on the workers that
    # hold its KV, thinned to the live set, instead of re-sharding
    p = payload()
    survivors = {"r": [0, 3, 4], "p": [0, 1, 2, 3]}
    prior = {"r": [[0, 4, 5], [1, 2, 3]]}
    subs = join_group_payloads(p, 2, survivors, p["joins"],
                               prior_shards=prior)
    assert subs[0]["anchor_index"] == [0, 4]
    assert subs[1]["anchor_index"] == [3]
    # without prior shards the anchor follows its filter shards
    subs = join_group_payloads(p, 2, survivors, p["joins"])
    assert subs[0]["anchor_index"] == [0, 4]
    assert subs[1]["anchor_index"] == [3]

    p = payload()
    survivors = {"r": [0, 1, 3, 4], "p": [0, 1, 2, 3]}
    prior = {"r": [[0], [3]]}

    subs = join_group_payloads(p, 2, survivors, p["joins"],
                               prior_shards=prior)

    assert 0 in subs[0]["anchor_index"]
    assert 3 in subs[1]["anchor_index"]
    assigned = [document for sub in subs
                for document in sub["anchor_index"]]
    assert sorted(assigned) == survivors["r"]
    assert len(assigned) == len(set(assigned))


def test_join_merge_gates_and_live_rows():
    out = dict(rows={0: [1, 0], 1: [0, 0], 2: [0, 1]},
               anchor_index=[5, 7, 9])
    assert gate_group(out, "full") == [5, 9]
    assert gate_group(out, "exists") == [5, 9]
    assert gate_group(out, "anti") == [7]
    # an anchor gated out mid-group has no last-stage row: dropped
    partial = dict(rows={0: [1]}, anchor_index=[5, 7])
    assert gate_group(partial, "full") == [5]

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


def test_coordinator_ships_and_merges_per_anchor_partner_lists():
    # merged rows keep each anchor's own member list
    outs = [
        dict(joins=[dict(rows={0: [1], 1: [0, 1]}, anchor_index=[0, 4],
                         partner_index=[[1], [2], [3]],
                         anchor_partners={0: [2], 1: [0, 1]})],
             fresh_tokens=1, wall_s=1.0),
        dict(joins=[dict(rows={0: [1, 0]}, anchor_index=[3],
                         partner_index=[[1], [2], [3]],
                         anchor_partners={0: [1, 2]})],
             fresh_tokens=1, wall_s=1.0),
    ]
    merged = merge_join_round(outs)[0]
    assert merged["anchor_partners"] == {0: [2], 1: [0, 1], 2: [1, 2]}
    assert list(join_answer_cells(merged)) == [
        (0, 2, 1), (1, 0, 0), (1, 1, 1), (2, 1, 1), (2, 2, 0)]
    # thinning reads the member list: anchor 0 matched partner 3
    # (member 2), anchor 4 matched partner 2, anchor 3 matched partner 2
    merged.update(anchor="r", partners=["p"])
    survivors = {"r": [0, 3, 4], "p": [1, 2, 3]}
    thin_survivors([merged], survivors)
    assert survivors == {"r": [0, 3, 4], "p": [2, 3]}

    # the payload of a pair stage carries each worker's anchors with
    # their live partner rows only
    pairs = pa.table({"r": pa.array([0, 0, 2, 3, 5], pa.int32()),
                      "p": pa.array([0, 3, 1, 2, 3], pa.int32())})
    payload = dict(
        model="qwen3-4b-fp8", chunk_tokens=1000, true_ids=[1],
        false_ids=[2], pre_ids=[9], filter_limit=None,
        docs={"r": [[i] * (10 + i) for i in range(6)],
              "p": [[i] * 5 for i in range(4)]},
        filters={"r": [[7, 7]]}, physical_plan={},
        shards={"r": ((0, 2, 4), (1, 3, 5))})
    group = [dict(anchor="r", partners=["p"], semantics="full",
                  written_pos=0, pairs_from="hash_join:r-p",
                  labels={"p": [2]}, frame=[8], tail=[3])]
    subs = join_group_payloads(payload, 2, {"r": [0, 2, 3, 5], "p": [1, 3]},
                               group, pair_tables={0: pairs})
    assert [sub["anchor_index"] for sub in subs] == [[0, 2], [3, 5]]
    assert [sub["pairs"] for sub in subs] == [
        {0: {0: [3], 2: [1]}}, {0: {3: [], 5: [3]}}]
