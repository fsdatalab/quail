"""The per-chunk state plan: segments, slot claims, and waves."""

import pytest
from fakes import cpu_staging
from test_state_pool import state_arena

from quail.backends.quail.executor import chunk as chunk_mod

torch = pytest.importorskip("torch")


def _tokens(n, base=1000):
    return list(range(base, base + n))


def _segments(chunk):
    return [tuple(seg) for seg in chunk.meta["state"]["segments"]]


def test_two_stage_filter_with_same_chunk_borrowing(monkeypatch):
    cpu_staging(monkeypatch)
    arena = state_arena(pages=64, slots=8)
    a, b = ("d", 0), ("d", 1)
    # A is 100 tokens; B shares A's first 64 and is 90 tokens; both
    # get a 5-token frame and a 3-token question in the same chunk
    assert arena.activate(a, 100, capacity_tokens=108, base_tokens=100, slots=3)
    assert arena.activate(b, 90, capacity_tokens=98, base_tokens=90,
                          borrow=(a, 64), slots=2)
    chunk = chunk_mod.pack_chunk(torch, arena, [
        dict(key=a, prefix=_tokens(100), start=0, f=100,
             suffixes=[_tokens(8, 5000)], write_suffix_tokens=5, single=True,
             save_at=(64, 100, 105)),
        dict(key=b, prefix=_tokens(26, 1064), start=64, f=90,
             suffixes=[_tokens(8, 5000)], write_suffix_tokens=5, single=True,
             save_at=(90, 95), borrow_parent=a),
    ], attention_mode="unified")
    a_share, a_base, a_kept = 1, 2, 3
    b_base, b_kept = 4, 5
    # the shared 64 rows and the frames are short, so the segments
    # that start from their slots rerun them as leads in the same
    # wave instead of waiting for the slots
    assert _segments(chunk) == [
        (a, 0, 64, 0, 0, a_share, 0, None),
        (a, 64, 100, 64, 0, a_base, 0, (0, 64)),
        (a, 100, 105, 100, a_base, a_kept, 1, None),
        (a, 105, 108, 105, a_base, 0, 1, (100, 105)),
        (b, 108, 134, 64, 0, b_base, 0, (0, 64)),
        (b, 134, 139, 90, b_base, b_kept, 1, None),
        (b, 139, 142, 95, b_base, 0, 1, (134, 139)),
    ]
    assert arena.state_slot_at(a, 64) == a_share
    assert arena.state_slot_at(a, 100) == a_base
    assert arena.state_slot_at(a, 105) == a_kept
    assert arena.state_slot_at(b, 95) == b_kept
    plan = chunk.meta["state"]
    assert plan["max_sequences"] == 4
    waves = plan["waves"]
    assert len(waves) == 2
    shared = list(range(64))
    assert waves[0]["rows"].tolist() == (
        shared + shared + list(range(64, 100)) + shared + list(range(108, 134)))
    assert waves[0]["cu_host"] == [0, 64, 164, 254]
    assert waves[0]["lead_host"] == [0, 64, 64]
    assert waves[0]["init_host"] == [0, 0, 0]
    assert waves[0]["save_host"] == [a_share, a_base, b_base]
    assert waves[0]["has_init"].tolist() == [False, False, False]
    assert waves[0]["save_index"].tolist() == [0, 1, 2]
    assert waves[0]["save_slots"].tolist() == [a_share, a_base, b_base]
    # the outputs of the reruns are dropped: the wave lands rows 0-63
    # once, then 64-99 and 108-133
    assert waves[0]["scatter"].tolist() == (
        shared + list(range(128, 164)) + list(range(228, 254)))
    assert waves[0]["out_rows"].tolist() == (
        shared + list(range(64, 100)) + list(range(108, 134)))
    frame_a, frame_b = list(range(100, 105)), list(range(134, 139))
    assert waves[1]["rows"].tolist() == (
        frame_a + frame_a + [105, 106, 107] + frame_b + frame_b + [139, 140, 141])
    assert waves[1]["cu_host"] == [0, 5, 13, 18, 26]
    assert waves[1]["lead_host"] == [0, 5, 0, 5]
    assert waves[1]["init_host"] == [a_base, a_base, b_base, b_base]
    assert waves[1]["save_host"] == [a_kept, 0, b_kept, 0]
    assert waves[1]["has_init"].tolist() == [True] * 4
    assert waves[1]["save_index"].tolist() == [0, 2]
    assert waves[1]["save_slots"].tolist() == [a_kept, b_kept]
    assert waves[1]["out_rows"].tolist() == (
        frame_a + [105, 106, 107] + frame_b + [139, 140, 141])
    assert waves[1]["n"] == 4 and waves[1]["max_len"] == 8

    # the second stage reads the kept state in one wave
    chunk = chunk_mod.pack_chunk(torch, arena, [
        dict(key=a, prefix=None, f=105, suffixes=[_tokens(4, 6000)]),
        dict(key=b, prefix=None, f=95, suffixes=[_tokens(4, 6000)]),
    ], attention_mode="unified")
    assert _segments(chunk) == [(a, 0, 4, 105, a_kept, 0, 0, None),
                                (b, 4, 8, 95, b_kept, 0, 0, None)]
    assert len(chunk.meta["state"]["waves"]) == 1
    assert chunk.meta["state"]["waves"][0]["scatter"] is None
    # a survivor rewound to its document keeps the base state only
    arena.retain(a, 100)
    assert arena.state_slot_at(a, 100) == a_base
    with pytest.raises(ValueError, match="no state saved at 105"):
        chunk_mod.pack_chunk(torch, arena, [
            dict(key=a, prefix=None, f=105, suffixes=[_tokens(4)])],
            attention_mode="unified")


def test_join_frame_and_partners_in_one_chunk(monkeypatch):
    cpu_staging(monkeypatch)
    arena = state_arena(pages=64, slots=6)
    x = ("d", 0)
    # the anchor's state is read after its frame only, so the prefix
    # and the frame run as one sequence into the one kept slot
    assert arena.activate(x, 100, capacity_tokens=120, base_tokens=100, slots=1)
    chunk = chunk_mod.pack_chunk(torch, arena, [
        dict(key=x, prefix=_tokens(100), f=100, suffixes=[_tokens(5, 5000)],
             write_suffix_tokens=5, save_at=(105,)),
        dict(key=x, prefix=None, f=105,
             suffixes=[_tokens(7, 6000), _tokens(8, 7000), _tokens(9, 8000)]),
    ], attention_mode="unified")
    kept = 1
    # the 105-row sequence is too long to rerun ahead of the partners
    assert _segments(chunk) == [
        (x, 0, 105, 0, 0, kept, 0, None),
        (x, 105, 112, 105, kept, 0, 1, None),
        (x, 112, 120, 105, kept, 0, 1, None),
        (x, 120, 129, 105, kept, 0, 1, None),
    ]
    assert arena.state_slot_at(x, 100) is None
    waves = chunk.meta["state"]["waves"]
    assert [w["n"] for w in waves] == [1, 3]
    assert waves[1]["cu_host"] == [0, 7, 15, 24]
    assert chunk.meta["state"]["max_sequences"] == 3
    for key in chunk.temporary_keys:
        arena.free_key(key)
    # later partners of the resident anchor read the kept state
    chunk = chunk_mod.pack_chunk(torch, arena, [
        dict(key=x, prefix=None, f=105, suffixes=[_tokens(6), _tokens(6)])],
        attention_mode="unified")
    assert [seg.init for seg in chunk.meta["state"]["segments"]] == [kept, kept]
    assert all(seg.wave == 0 for seg in chunk.meta["state"]["segments"])


def test_partners_rerun_a_resident_anchors_frame_in_its_wave(monkeypatch):
    cpu_staging(monkeypatch)
    arena = state_arena(pages=64, slots=6)
    y = ("d", 0)
    assert arena.activate(y, 100, capacity_tokens=120, base_tokens=100, slots=2)
    chunk_mod.pack_chunk(torch, arena, [
        dict(key=y, prefix=_tokens(100), f=100, suffixes=[], save_at=(100,))],
        attention_mode="unified")
    base = arena.state_slot_at(y, 100)
    # as a join packs a resident anchor: the frame entry from the base
    # state, then the partners from the kept state
    chunk = chunk_mod.pack_chunk(torch, arena, [
        dict(key=y, prefix=None, f=100, suffixes=[_tokens(5, 5000)],
             write_suffix_tokens=5, save_at=(105,)),
        dict(key=y, prefix=None, f=105,
             suffixes=[_tokens(7, 6000), _tokens(8, 7000)]),
    ], attention_mode="unified")
    kept = arena.state_slot_at(y, 105)
    assert _segments(chunk) == [
        (y, 0, 5, 100, base, kept, 0, None),
        (y, 5, 12, 105, base, 0, 0, (0, 5)),
        (y, 12, 20, 105, base, 0, 0, (0, 5)),
    ]
    waves = chunk.meta["state"]["waves"]
    assert len(waves) == 1
    frame = [0, 1, 2, 3, 4]
    assert waves[0]["rows"].tolist() == (
        frame + frame + list(range(5, 12)) + frame + list(range(12, 20)))
    assert waves[0]["cu_host"] == [0, 5, 17, 30]
    assert waves[0]["scatter"].tolist() == (
        frame + list(range(10, 17)) + list(range(22, 30)))
    assert waves[0]["out_rows"].tolist() == list(range(20))
    assert waves[0]["max_len"] == 13


def test_single_pass_groups_save_nothing_and_errors(monkeypatch):
    cpu_staging(monkeypatch)
    arena = state_arena(pages=64, slots=4)
    key = ("d", 0)
    assert arena.activate(key, 20, capacity_tokens=40, base_tokens=20)
    chunk = chunk_mod.pack_chunk(torch, arena, [
        dict(key=key, prefix=_tokens(20), f=20, suffixes=[_tokens(6)])],
        attention_mode="unified")
    # one suffix and nothing to save: the prefix and suffix are one
    # sequence from the zero state
    assert _segments(chunk) == [(key, 0, 26, 0, 0, 0, 0, None)]
    assert arena.state.owned_count(key) == 0
    with pytest.raises(ValueError, match="unified attention"):
        chunk_mod.pack_chunk(torch, arena, [
            dict(key=key, prefix=None, f=20, suffixes=[_tokens(6)])],
            attention_mode="tree")
    with pytest.raises(ValueError, match="no recurrent state"):
        chunk_mod.pack_chunk(torch, arena, [
            dict(key=key, prefix=None, f=20, suffixes=[_tokens(6)])],
            attention_mode="unified", canvas=(7,))
    with pytest.raises(ValueError, match="no state saved at 20"):
        chunk_mod.pack_chunk(torch, arena, [
            dict(key=key, prefix=None, f=20, suffixes=[_tokens(6)])],
            attention_mode="unified")
    # a kept state alone cuts the one sequence where the kept rows
    # end; the tail reruns the short head in the same wave
    kept_key = ("d", 2)
    assert arena.activate(kept_key, 20, capacity_tokens=40, base_tokens=20,
                          slots=1)
    chunk = chunk_mod.pack_chunk(torch, arena, [
        dict(key=kept_key, prefix=_tokens(20), f=20, suffixes=[_tokens(6)],
             write_suffix_tokens=2, save_at=(22,))],
        attention_mode="unified")
    slot = arena.state_slot_at(kept_key, 22)
    assert _segments(chunk) == [(kept_key, 0, 22, 0, 0, slot, 0, None),
                                (kept_key, 22, 26, 22, 0, 0, 0, (0, 22))]
    # a saved base needs a reserved slot
    with pytest.raises(ValueError, match="exceeds its 0 reserved"):
        chunk_mod.pack_chunk(torch, arena, [
            dict(key=key, prefix=_tokens(20), f=20, suffixes=[_tokens(6)],
                 save_at=(20,))],
            attention_mode="unified")
    child = ("d", 1)
    assert arena.activate(child, 32, capacity_tokens=40, base_tokens=32,
                          borrow=(key, 16))
    with pytest.raises(ValueError, match="no state saved at 16 on its parent"):
        chunk_mod.pack_chunk(torch, arena, [
            dict(key=child, prefix=_tokens(16), start=16, f=32,
                 suffixes=[_tokens(6)], borrow_parent=key)],
            attention_mode="unified")
