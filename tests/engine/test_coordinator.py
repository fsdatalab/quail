"""Typed input partitioning and output merging across workers."""

import pyarrow as pa

from quail.backends.quail.coordinator import (
    keyed_shards,
    merge_join_answers,
    shard_input,
)
from quail.execution.rows import Rows
from quail.physical import InputPort, PortRef, ValueType


def test_shards_keep_anchor_rows_and_all_partners():
    ids = pa.array([4, 1, 3], pa.int32())
    own = InputPort("input", ValueType.DOCUMENT_IDS, PortRef("scan:r", "ids:r"))
    partner = InputPort("partner", ValueType.DOCUMENT_IDS, PortRef("scan:p", "ids:p"))
    assert shard_input(own, ids, "r", [1, 4]).to_pylist() == [4, 1]
    assert shard_input(partner, ids, "r", [1, 4]) is ids
    rows = Rows({"r": ids}, pa.table({"label": ["a", "b", "c"]}))
    port = InputPort("input", ValueType.ROWS, PortRef("classify", "rows"))
    selected = shard_input(port, rows, "r", [1, 4])
    assert selected.table().to_pylist() == [
        {"r": 4, "label": "a"}, {"r": 1, "label": "b"}]
    pairs = pa.table({"r": [4, 1, 1, 3], "p": [0, 1, 2, 0]})
    port = InputPort("pairs", ValueType.PAIRS, PortRef("hash", "pairs:0"))
    assert shard_input(port, pairs, "r", [1]).to_pylist() == [
        {"r": 1, "p": 1}, {"r": 1, "p": 2}]


def test_stop_groups_stay_on_one_worker():
    assert keyed_shards([0, 1, 0, 2, 1, 0], [10, 11, 12, 13, 14, 15], 2) == (
        (0, 2, 5), (1, 3, 4))


def test_join_merge_and_partner_lists():
    # the merge keeps stages aligned though stage 2 skips gated anchors
    outs = [
        dict(joins=[dict(rows={0: [1, 0], 1: [0, 0]}, anchor_index=[0, 4],
                         partner_index=[[1], [2]]),
                    dict(rows={0: [0, 1, 1]}, anchor_index=[0, 4],
                         partner_index=[[0], [1], [2]])],
             fresh_tokens=10, wall_s=1.0),
        dict(joins=[dict(rows={0: [1, 1]}, anchor_index=[3],
                         partner_index=[[1], [2]]),
                    dict(rows={0: [1, 0, 0]}, anchor_index=[3],
                         partner_index=[[0], [1], [2]])],
             fresh_tokens=10, wall_s=1.0),
    ]
    merged = [merge_join_answers([out["joins"][i] for out in outs])
              for i in range(len(outs[0]["joins"]))]
    assert len(merged) == 2
    assert merged[0]["anchor_index"] == [0, 4, 3]
    assert merged[0]["rows"] == {0: [1, 0], 1: [0, 0], 2: [1, 1]}
    assert merged[1]["anchor_index"] == [0, 4, 3]
    assert merged[1]["rows"] == {0: [0, 1, 1], 2: [1, 0, 0]}
    assert merged[1]["partner_index"] == [[0], [1], [2]]

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
    merged = merge_join_answers([out["joins"][0] for out in outs])
    assert merged["anchor_partners"] == {0: [2], 1: [0, 1], 2: [1, 2]}
