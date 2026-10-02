"""Test packed token rows, positions, and answer rows from flattened suffixes."""

import pytest
from fakes import cpu_staging
from test_sliding_kv import plain_arena

from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.chunk import Suffixes

torch = pytest.importorskip("torch")


def test_pack_chunk_rows_for_many_suffixes(monkeypatch):
    cpu_staging(monkeypatch)
    arena = plain_arena()
    key = ("d", 0)
    arena.activate(key, 4, capacity_tokens=8, base_tokens=4)
    sufs = Suffixes.of([[10, 11], [12], [13, 14, 15]])
    chunk = chunk_mod.pack_chunk(
        torch, arena,
        [dict(key=key, prefix=[1, 2, 3, 4], f=4, suffixes=sufs,
              read_all_rows=True)],
        attention_mode="tree")
    assert chunk.input_ids.tolist() == [1, 2, 3, 4, 10, 11, 12, 13, 14, 15]
    # each suffix's positions restart after the prefix
    assert chunk.positions.tolist() == [0, 1, 2, 3, 4, 5, 4, 4, 5, 6]
    assert chunk.final_indices.tolist() == [4, 5, 6, 7, 8, 9]
    assert chunk.rows_per_answer == (2, 1, 3)
    assert chunk.meta["cu_a"].tolist() == [0, 4, 6, 7, 10]
    assert chunk.meta["max_a"] == 4
    assert chunk.layout == [(key, 3)]
    assert chunk.fresh_keys == (key,)
    # the prefix rows are written to the key's pages
    assert chunk.meta["kv_src"].tolist() == [0, 1, 2, 3]
    # a kept document: the last row of each suffix answers, and every
    # suffix row reads the kept prefix
    chunk = chunk_mod.pack_chunk(
        torch, arena, [dict(key=key, prefix=None, f=4, suffixes=sufs)],
        attention_mode="tree")
    assert chunk.input_ids.tolist() == [10, 11, 12, 13, 14, 15]
    assert chunk.final_indices.tolist() == [1, 2, 5]
    assert chunk.rows_per_answer == ()
    assert chunk.meta["cu_a"].tolist() == [0, 2, 3, 6]
    reads = chunk.meta["reads"]
    assert reads["rows"].tolist() == [0, 1, 2, 3, 4, 5]
    assert reads["cu_q"].tolist() == [0, 6]
    assert reads["used"].tolist() == [4]
    assert reads["source"].tolist() == [0, 1, 2, 3, 4, 5]


def test_a_fresh_single_group_is_one_causal_segment_under_tree(monkeypatch):
    cpu_staging(monkeypatch)
    arena = plain_arena()
    key = ("d", 2)
    arena.activate(key, 3, capacity_tokens=8, base_tokens=3)
    group = dict(key=key, prefix=[7, 8, 9], f=3, suffixes=[[20, 21, 22]],
                 write_suffix_tokens=2, single=True)
    chunk = chunk_mod.pack_chunk(torch, arena, [group], attention_mode="tree")
    # no segment boundary after the prefix and no read of the pages
    assert chunk.meta["cu_a"].tolist() == [0, 6]
    assert chunk.meta["reads"] is None
    assert chunk.meta["kv_src"].tolist() == [0, 1, 2, 3, 4]
    assert chunk.final_indices.tolist() == [5]
    # a kept document's single suffix still reads its pages
    chunk = chunk_mod.pack_chunk(
        torch, arena, [dict(key=key, prefix=None, f=5, suffixes=[[30]],
                            single=True)], attention_mode="tree")
    assert chunk.meta["reads"]["used"].tolist() == [5]


@pytest.mark.parametrize("mode", ["tree", "unified"])
def test_decode_rounds_keep_each_fed_token_after_the_frame(monkeypatch, mode):
    cpu_staging(monkeypatch)
    arena = plain_arena()
    key = ("d", 3)
    # a three-token document, a two-token frame, the cue, and room for
    # two decoded tokens: 3 + 2 + 1 + 2 = 8 rows reserved
    arena.activate(key, 3, capacity_tokens=8, base_tokens=3)
    rows = arena.capacity_rows(key)

    def written(chunk):
        if mode == "tree":
            return chunk.meta["kv_dst"].tolist()
        return chunk.meta["unified"]["dst"].tolist()

    # round 0 packs the document, frame, and cue and keeps all six rows
    chunk = chunk_mod.pack_chunk(
        torch, arena, [dict(key=key, prefix=[7, 8, 9], f=3,
                            suffixes=[[91, 92, 93]], write_suffix_tokens=3,
                            single=True)], attention_mode=mode)
    assert written(chunk) == rows[:6].tolist()
    # round 1 feeds one token at position 6, reads the six kept rows,
    # and keeps it at row 6
    chunk = chunk_mod.pack_chunk(
        torch, arena, [dict(key=key, prefix=None, f=6, suffixes=[[40]],
                            write_suffix_tokens=1, single=True)],
        attention_mode=mode)
    assert chunk.positions.tolist() == [6]
    assert written(chunk) == rows[6:7].tolist()
    # a kept token past the key's one 16-row page is refused, not dropped
    with pytest.raises((AssertionError, ValueError)):
        chunk_mod.pack_chunk(
            torch, arena, [dict(key=key, prefix=None, f=16, suffixes=[[41]],
                                write_suffix_tokens=1, single=True)],
            attention_mode=mode)
