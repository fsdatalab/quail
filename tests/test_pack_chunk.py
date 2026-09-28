"""The chunk packer's rows, positions, and readout rows from flat suffix arrays."""

import pytest
from fakes import cpu_staging
from test_sliding_kv import plain_arena

from quail.backends.quail.executor import loop
from quail.backends.quail.executor.loop import Suffixes

torch = pytest.importorskip("torch")


def test_suffixes_take_gathers_in_order():
    sufs = Suffixes.of([[1, 2], [3], [4, 5, 6], []])
    assert [list(s) for s in sufs] == [[1, 2], [3], [4, 5, 6], []]
    assert len(sufs) == 4 and sufs.offsets.tolist() == [0, 2, 3, 6, 6]
    picked = sufs.take([2, 0, 3])
    assert picked.ids.tolist() == [4, 5, 6, 1, 2]
    assert picked.lengths.tolist() == [3, 2, 0]
    assert sufs.take(range(1, 3)).ids.tolist() == [3, 4, 5, 6]
    assert sufs.lengths_at(range(0, 4)).tolist() == [2, 1, 3, 0]
    assert sufs.lengths_at([3, 1]).tolist() == [0, 1]
    with pytest.raises(ValueError):
        Suffixes([1, 2, 3], [2])


def test_pack_chunk_rows_for_many_suffixes(monkeypatch):
    cpu_staging(monkeypatch)
    arena = plain_arena()
    key = ("d", 0)
    arena.activate(key, 4, capacity_tokens=8, base_tokens=4)
    sufs = Suffixes.of([[10, 11], [12], [13, 14, 15]])
    chunk = loop.pack_chunk(
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
    chunk = loop.pack_chunk(
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


def test_pack_chunk_accepts_suffix_lists_and_a_frame_write(monkeypatch):
    cpu_staging(monkeypatch)
    arena = plain_arena()
    key = ("d", 1)
    arena.activate(key, 3, capacity_tokens=8, base_tokens=3)
    chunk = loop.pack_chunk(
        torch, arena,
        [dict(key=key, prefix=[7, 8, 9], f=3, suffixes=[[20, 21, 22]],
              write_suffix_tokens=2)],
        attention_mode="tree")
    assert chunk.input_ids.tolist() == [7, 8, 9, 20, 21, 22]
    assert chunk.positions.tolist() == [0, 1, 2, 3, 4, 5]
    assert chunk.final_indices.tolist() == [5]
    # the prefix and the two frame rows land in the key's pages
    assert chunk.meta["kv_src"].tolist() == [0, 1, 2, 3, 4]
    assert chunk.meta["kv_dst"].tolist() == arena.capacity_rows(key)[:5].tolist()
