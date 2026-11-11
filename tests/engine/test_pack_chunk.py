"""Test packed token rows, positions, and answer rows from flattened suffixes."""

import numpy as np
import pytest
from helpers.fakes import cpu_staging, plain_arena
from helpers.rust_cost import classify_estimate, cost_model
from test_kv_reads import named

from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.chunk import Suffixes
from quail.specs import H100_SXM, QWEN3_4B_FP8

torch = pytest.importorskip("torch")


def test_pack_chunk_rows_for_many_suffixes_and_copy_drafts(monkeypatch):
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
    assert chunk.rows_per_answer.tolist() == [2, 1, 3]
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
    assert chunk.rows_per_answer is None
    assert chunk.meta["cu_a"].tolist() == [0, 2, 3, 6]
    reads = chunk.meta["reads"]
    assert reads["rows"].tolist() == [0, 1, 2, 3, 4, 5]
    assert reads["cu_q"].tolist() == [0, 6]
    assert reads["used"].tolist() == [4]
    assert reads["source"].tolist() == [0, 1, 2, 3, 4, 5]

    for mode, shared in [("tree", True), ("unified", False)]:
        named(
            f"copy drafts mode={mode} shared={shared}",
            check_copy_drafts_pack_as_causal_sequences_without_combining_slots,
            monkeypatch=monkeypatch, mode=mode, shared=shared)


@pytest.mark.parametrize("mode", ["tree"])
@pytest.mark.parametrize("shared", [16])
@pytest.mark.parametrize("run", [2])
def test_decode_cost_matches_packed_retained_kv(monkeypatch, mode, shared, run):
    cpu_staging(monkeypatch)
    arena = plain_arena()
    parent, key = ("d", 0), ("d", 1)
    arena.activate(parent, 32, capacity_tokens=40, base_tokens=32)
    arena.activate(key, 32, capacity_tokens=40, base_tokens=32,
                   borrow=(parent, shared) if shared else None)
    retained_reads = fresh_tokens = 0
    for round_ in range(2):
        first = round_ == 0
        chunk = chunk_mod.pack_chunk(
            torch, arena, [dict(
                key=key, prefix=list(range(shared, 32)) if first else None,
                f=32 if first else 35, start=shared if first else 0,
                read_key=parent if first and shared else None,
                suffixes=[[91, 92, 93]] if first else [[40, 41][:run]],
                write_suffix_tokens=3 if first else run, single=True)],
            attention_mode=mode)
        if mode == "tree":
            reads = chunk.meta["reads"]
            retained = 0 if reads is None else int(reads["used"].sum())
        else:
            pool = chunk.meta["unified"]
            retained = int((pool["used"] - pool["cu_q"].diff()).sum())
        assert retained == (shared if first else 35)
        retained_reads += retained
        fresh_tokens += len(chunk.input_ids)
        # the one label decides at the cue; the four decide at a second
        # choice that a run of one or two fed tokens reaches
        labels = ([(10,)] if first else
                  [(10, *range(11, 10 + run), 1), (10, *range(11, 10 + run), 2),
                   (20, *range(21, 20 + run), 1), (20, *range(21, 20 + run), 2)])
        cost = classify_estimate(
            cost_model(), "trie_decode", 1, 0, 2, labels, lengths=(32,),
            shared=(shared,), chunk=100,
            model=QWEN3_4B_FP8.name, device=H100_SXM.name)
        assert cost.work["tokens"] == fresh_tokens
        assert cost.work["kv_read"] == retained_reads
def check_copy_drafts_pack_as_causal_sequences_without_combining_slots(
        monkeypatch, mode, shared):
    cpu_staging(monkeypatch)
    arena = plain_arena()
    keys = [("d", 0), ("d", 1)]
    for key in keys:
        arena.activate(key, 32, capacity_tokens=48, base_tokens=32,
                       borrow=(keys[0], 16) if shared and key == keys[1] else None)
    slots = np.array([0, 0, 0, 1, 1, 1])
    positions = np.array([35, 36, 37, 33, 34, 37])
    tables = np.array([arena.owned_pages(keys[slot]) for slot in slots])
    decode = chunk_mod.DecodeRows(
        slots=slots, positions=positions, page_table=tables,
        write_rows=np.array([int(arena.capacity_rows(keys[slot])[position])
                             for slot, position in zip(slots, positions)]))
    if shared:
        decode.own_table = tables[:, 1:]
        decode.own_lengths = positions + 1 - 16
        decode.shared_rows = np.arange(len(slots))
        decode.shared_cu = np.array([0, len(slots)])
        decode.shared_table = tables[:1, :1]
        decode.shared_used = np.array([16])
    chunk = chunk_mod.pack_chunk(
        torch, arena, [], attention_mode=mode, decode=decode)
    # A gap in a slot's positions starts another causal sequence.
    assert chunk.meta["cu_a"].tolist() == [0, 3, 5, 6]
    assert chunk.final_indices.tolist() == list(range(6))
    assert chunk.rows_per_answer is None
    if mode == "tree":
        reads = chunk.meta["reads"]
        assert reads["cu_q"].tolist() == [0, 3, 5, 6]
        assert reads["used"].tolist() == [35, 33, 37]
        assert reads["table"].tolist() == tables[[0, 3, 5]].tolist()
        assert chunk.meta["kv_dst"].tolist() == decode.write_rows.tolist()
    else:
        reads = chunk.meta["unified"]
        assert reads["cu_q"].tolist() == [0, 3, 5, 6]
        assert reads["used"].tolist() == [n - (16 if shared else 0)
                                          for n in [38, 35, 38]]
        expected = tables[[0, 3, 5], 1:] if shared else tables[[0, 3, 5]]
        assert reads["table"].tolist() == expected.tolist()
        assert reads["dst"].tolist() == decode.write_rows.tolist()
        assert (chunk.meta["shared"] is not None) == shared
    for key in reversed(keys):
        arena.free_key(key)
    assert not arena.accounting.owned
