"""The sliding-layer KV pool: window origins, trimming, and one page currency."""

import pytest
from helpers.fakes import cpu_staging, plain_arena

from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.arena import KVArena

torch = pytest.importorskip("torch")

WINDOW = 32     # tokens, two pages


def cpu_arena(pages=64, sliding_pages=16):
    # layer 0 keeps every token, layer 1 slides
    return KVArena(
        num_layers=2,
        num_pages=pages,
        page_tokens=16,
        num_kv_heads=1,
        d_head=2,
        dtype=torch.float32,
        device="cpu",
        layer_kv=[(1, 2), (1, 2)],
        sliding_layers=(1,),
        sliding_window=WINDOW,
        num_sliding_pages=sliding_pages,
    )


def test_window_origin_and_trim():
    arena = cpu_arena()
    assert arena.has_sliding
    assert arena.origin(20) == 0
    assert arena.origin(WINDOW) == 0
    assert arena.origin(100) == 64      # 100 - 32 = 68, down to a page
    key = ("d", 0)
    # a fresh 100-token document with 20 rows of room takes 8 pages per pool
    assert arena.activate(key, 100, capacity_tokens=120, base_tokens=100)
    assert len(arena.owned_pages(key)) == 8
    assert len(arena.owned_sliding_pages(key)) == 8
    assert arena.sliding_start(key) == 0
    assert arena.free_pages == min(64 - 8, int((16 - 8) * 4))
    freed = arena.trim_window(key)
    # pages before row 64 leave the sliding pool: 4 of them
    assert len(arena.owned_sliding_pages(key)) == 4
    assert arena.sliding_start(key) == 64
    assert freed == arena.free_pages - min(64 - 8, (16 - 8) * 4)
    assert arena.trim_window(key) == 0
    assert arena.capacity_rows_sliding(key).numel() == 4 * 16
    assert arena.capacity_rows(key).numel() == 8 * 16
    assert arena.activate(key, 110, capacity_tokens=140) is not None
    assert len(arena.owned_pages(key)) == 9
    assert len(arena.owned_sliding_pages(key)) == 5     # (140 - 64) / 16
    # retaining below the base would drop window rows
    with pytest.raises(ValueError, match="window"):
        arena.retain(key, 90)
    arena.retain(key, 100)
    assert len(arena.owned_pages(key)) == 7
    assert len(arena.owned_sliding_pages(key)) == 3     # (100 - 64) / 16
    arena.free_key(key)
    assert arena.free_pages == 64


def test_pack_chunk_builds_both_pools_and_reads_borrowed_pages(monkeypatch):
    cpu_staging(monkeypatch)
    arena = cpu_arena(pages=64, sliding_pages=32)
    key = ("d", 0)
    document = list(range(100))
    tail = [500, 501]
    canvas = (900, 901)
    arena.activate(key, 100, capacity_tokens=120, base_tokens=100)
    chunk = chunk_mod.pack_chunk(
        torch, arena, [dict(key=key, prefix=document, f=100, suffixes=[tail])],
        attention_mode="unified", canvas=canvas)
    assert chunk.fresh_keys == (key,)
    full = chunk.meta["unified"]
    sliding = full["sliding"]
    assert full["src"].tolist() == list(range(104))
    assert sliding["src"].tolist() == list(range(104))
    assert full["used"].tolist() == [104] and sliding["used"].tolist() == [104]
    assert full["table"].shape == (1, 8) and sliding["table"].shape == (1, 8)
    assert chunk.meta["canvas"]["sliding"]["used"].tolist() == [104]
    arena.trim_window(key)
    # a later stage reads the kept rows: the sliding pool from row 64
    chunk = chunk_mod.pack_chunk(
        torch, arena, [dict(key=key, prefix=None, f=100, suffixes=[[600]])],
        attention_mode="unified", canvas=canvas)
    assert chunk.fresh_keys == ()
    full = chunk.meta["unified"]
    sliding = full["sliding"]
    assert full["used"].tolist() == [103]
    assert sliding["used"].tolist() == [103 - 64]
    assert sliding["table"].shape[1] == 4
    # the suffix rows go to the key's own pages after row 100 in the
    # every-token pool and after row 36 in the sliding pool
    assert full["dst"].tolist() == arena.capacity_rows(key)[100:103].tolist()
    assert sliding["dst"].tolist() == \
        arena.capacity_rows_sliding(key)[36:39].tolist()
    assert chunk.meta["canvas"]["sliding"]["used"].tolist() == [39]
    # two suffixes take temporaries after a copy of each partial last page
    chunk = chunk_mod.pack_chunk(
        torch, arena, [dict(key=key, prefix=None, f=100,
                            suffixes=[[600], [601, 602]])],
        attention_mode="unified")
    full = chunk.meta["unified"]
    sliding = full["sliding"]
    assert len(chunk.temporary_keys) == 2
    assert full["used"].tolist() == [101, 102]
    assert sliding["used"].tolist() == [37, 38]
    assert full["tail_src"].numel() == 2 * (100 % 16)
    assert sliding["tail_src"].numel() == 2 * (36 % 16)
    for temp in chunk.temporary_keys:
        arena.free_key(temp)
    # each suffix may carry its own canvas
    chunk = chunk_mod.pack_chunk(
        torch, arena, [dict(key=key, prefix=None, f=100,
                            suffixes=[[600], [601, 602]],
                            canvas=[[900, 901], [910, 911]])],
        attention_mode="unified", canvas=canvas)
    assert chunk.input_ids.tolist() == [600, 900, 901, 601, 602, 910, 911]
    assert chunk.positions.tolist() == [100, 101, 102, 100, 101, 102, 103]
    assert chunk.final_indices.tolist() == [1, 5]
    assert chunk.meta["unified"]["used"].tolist() == [103, 104]
    for temp in chunk.temporary_keys:
        arena.free_key(temp)
    with pytest.raises(ValueError, match="1 canvases for 2 suffixes"):
        chunk_mod.pack_chunk(
            torch, arena, [dict(key=key, prefix=None, f=100,
                                suffixes=[[600], [601]], canvas=[[900, 901]])],
            attention_mode="unified", canvas=canvas)
    arena.free_key(key)
    assert arena.free_pages == 64

    arena = plain_arena()
    parent, child = ("d", 0), ("d", 1)
    tail = [500]
    arena.activate(parent, 100, capacity_tokens=110, base_tokens=100)
    # the child shares 64 tokens (4 pages) and adds 30 of its own
    arena.activate(child, 94, capacity_tokens=104, base_tokens=94,
                   borrow=(parent, 64))
    assert arena.owned_pages(child)[:4] == arena.owned_pages(parent)[:4]
    chunk = chunk_mod.pack_chunk(
        torch, arena,
        [dict(key=parent, prefix=document, f=100, suffixes=[tail]),
         dict(key=child, prefix=document[64:94], start=64, f=94, suffixes=[tail])],
        attention_mode="unified")
    assert chunk.fresh_keys == (parent, child)
    assert chunk.tokens == 101 + 31
    positions = chunk.positions.tolist()
    assert positions[:101] == list(range(101))
    assert positions[101:] == list(range(64, 95))
    unified = chunk.meta["unified"]
    assert unified["used"].tolist() == [101, 95]
    table = unified["table"].tolist()
    assert table[1][:4] == table[0][:4]
    # the child's rows land after the borrowed pages, in its own pages
    dst = unified["dst"].tolist()
    assert dst[101:] == arena.capacity_rows(child)[64:95].tolist()
    assert arena.capacity_rows(child)[:64].tolist() == \
        arena.capacity_rows(parent)[:64].tolist()
    # under tree attention without a read_key, the fresh rows read the
    # child's own first 64 rows (the borrowed pages) and the suffix
    # reads all 94
    chunk = chunk_mod.pack_chunk(
        torch, arena,
        [dict(key=child, prefix=document[64:94], start=64, f=94,
              suffixes=[tail, [501]])],
        attention_mode="tree")
    reads = chunk.meta["reads"]
    assert reads["used"].tolist() == [64, 94]
    assert reads["cu_q"].tolist() == [0, 30, 32]
    assert chunk.meta["cu_a"].tolist() == [0, 30, 31, 32]
    arena.free_key(parent)
    arena.free_key(child)
    assert arena.free_pages == 64

    arena = plain_arena()
    parent, a, b = ("d", 0), ("d", 1), ("d", 2)
    arena.activate(parent, 100, capacity_tokens=110, base_tokens=100)
    # two siblings share 64 tokens (4 pages) of the parent
    arena.activate(a, 74, capacity_tokens=84, base_tokens=74,
                   borrow=(parent, 64))
    arena.activate(b, 70, capacity_tokens=80, base_tokens=70,
                   borrow=(parent, 64))
    chunk = chunk_mod.pack_chunk(
        torch, arena,
        [dict(key=parent, prefix=document, f=100, suffixes=[tail]),
         dict(key=a, prefix=document[64:74], start=64, read_key=parent, f=74,
              suffixes=[tail], write_suffix_tokens=1),
         dict(key=b, prefix=document[64:70], start=64, read_key=parent, f=70,
              suffixes=[tail], write_suffix_tokens=1)],
        attention_mode="tree")
    assert chunk.fresh_keys == (parent, a, b)
    # call A: the parent's prefix, its tail, then each sibling's
    # prefix and tail as one causal segment
    assert chunk.meta["cu_a"].tolist() == [0, 100, 101, 112, 119]
    reads = chunk.meta["reads"]
    # call B: the parent's tail reads its 100 rows; both siblings' 18
    # rows read the parent's 64 shared positions in one sequence
    assert reads["cu_q"].tolist() == [0, 1, 19]
    assert reads["used"].tolist() == [100, 64]
    assert reads["max_q"] == 18
    assert reads["rows"].tolist() == [100] + list(range(101, 119))
    table = reads["table"].tolist()
    assert table[1][:4] == table[0][:4]
    # the siblings' prefixes and kept tails write their own pages after
    # the borrowed ones
    dst = chunk.meta["kv_dst"].tolist()
    assert dst[100:110] == arena.capacity_rows(a)[64:74].tolist()
    assert dst[110:111] == arena.capacity_rows(a)[74:75].tolist()
    assert dst[111:117] == arena.capacity_rows(b)[64:70].tolist()
    assert dst[117:118] == arena.capacity_rows(b)[70:71].tolist()
    for key in (parent, a, b):
        arena.free_key(key)
    assert arena.free_pages == 64
