"""Tests for the host KV store: its copies, extents, and eviction."""

import torch

from quail.backends.quail.executor.kv_store import (
    PIECE_TOKENS,
    KVStore,
    content_id,
)

TENSORS, HEADS, DIM = 4, 2, 64
RECORD_BYTES = TENSORS * HEADS * DIM * 2


def _pool(rows=256, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(0)
    return torch.randn((TENSORS, rows, HEADS, DIM), generator=generator).to(dtype)


def _store(records, **options):
    return KVStore(torch, (TENSORS, HEADS, DIM), torch.bfloat16,
                   records * RECORD_BYTES, device="cpu", **options)


def test_a_saved_prefix_restores_to_other_rows_bit_for_bit():
    pool = _pool()
    store = _store(64)
    source = torch.arange(3, 203, 5)
    target = torch.arange(210, 250)
    store.save(b"document", source)
    # a prefix can be restored once a flush has stored it
    assert not store.has(b"document", 40)
    store.flush(pool)
    assert store.has(b"document", 40) and not store.has(b"document", 39)
    store.restore(b"document", target)
    store.flush(pool)
    assert torch.equal(pool[:, source], pool[:, target])
    stats = store.stats()
    assert (stats["coder"], stats["restored_tokens"]) == ("raw", 40)
    assert stats["held_bytes"] == 40 * RECORD_BYTES


def test_a_prefix_longer_than_a_piece_restores_whole():
    tokens = 2 * PIECE_TOKENS + 100
    pool = _pool(rows=2 * tokens)
    store = _store(tokens)
    store.save(b"long", torch.arange(tokens))
    store.flush(pool)
    assert len(store.saved[b"long"].pieces) == 3
    store.restore(b"long", torch.arange(tokens, 2 * tokens))
    store.flush(pool)
    assert torch.equal(pool[:, :tokens], pool[:, tokens:])


def _full_store(pool):
    """A store of 3072 records holding a long prefix and two short ones."""
    # a pass over a prefix costs its tokens and its token pairs
    store = _store(3072, prefill_seconds=lambda tokens: tokens + tokens * tokens / 2)
    store.save(b"long", torch.arange(2048))
    store.save(b"short", torch.arange(512))
    store.save(b"other", torch.arange(512))
    store.flush(pool)
    return store


def test_a_full_store_drops_short_prefixes_before_a_long_one():
    pool = _pool(rows=4096)
    store = _full_store(pool)
    store.save(b"new", torch.arange(600))
    store.flush(pool)
    # the long prefix saves the most seconds per byte, so both short ones left
    assert set(store.saved) == {b"long", b"new"}
    assert store.stats()["dropped_tokens"] == 1024


def test_a_long_prefix_no_query_reads_leaves_after_short_ones_in_use():
    pool = _pool(rows=8192)
    store = _full_store(pool)
    # each eviction raises the clock under the prefixes restored after it
    for step in range(8):
        name = b"filler %d" % step
        store.save(name, torch.arange(512))
        store.flush(pool)
        store.restore(name, torch.arange(4096, 4608))
        store.flush(pool)
    assert b"long" not in store.saved
    assert torch.equal(pool[:, 4096:4608], pool[:, :512])


def test_a_prefix_larger_than_the_store_is_skipped():
    pool = _pool(rows=2048)
    store = _store(1024)
    store.save(b"long", torch.arange(2000))
    store.flush(pool)
    assert not store.has(b"long", 2000) and store.held_tokens == 0
    assert store.stats()["skipped_tokens"] == 2000
    # the room its first piece took is free again
    store.save(b"short", torch.arange(1024))
    store.flush(pool)
    assert store.has(b"short", 1024)


def test_a_content_id_names_the_tokens_not_their_container():
    assert content_id([5, 6, 7]) == content_id((5, 6, 7))
    assert content_id([5, 6, 7]) != content_id([5, 6])


def _arena(pages):
    from quail.backends.quail.executor.arena import KVArena

    arena = KVArena(num_layers=TENSORS // 2, num_pages=pages, page_tokens=16,
                    num_kv_heads=HEADS, d_head=DIM, dtype=torch.bfloat16,
                    device="cpu")
    arena.attach_store(_store(4096))
    return arena


def _write(arena, key, tokens, seed):
    """Fill a fresh key's prefix rows and note them as written."""
    arena.activate(key, tokens)
    rows = arena.capacity_rows(key)[:tokens]
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((TENSORS, tokens, HEADS, DIM), generator=generator)
    arena._pool[:, rows] = values.to(torch.bfloat16)
    arena.note_prefix(key, b"content %d" % seed, tokens)
    return arena._pool[:, rows].clone()


def test_a_freed_prefix_stays_in_the_arena_and_lends_its_whole_pages():
    arena = _arena(pages=16)
    written = _write(arena, ("r", 0), 40, seed=1)
    arena.free_key(("r", 0))
    # the kept prefix's 3 pages count as free: an admission may take them
    assert arena.resident_keys() == [] and arena.free_pages == 16
    kept, shared = arena.kept_prefix(b"content 1", 40)
    assert shared == 32 and arena.kept_prefix(b"content 1", 39) == (None, 0)
    # a document with the same prefix reads the kept pages and writes its own
    assert arena.reserve(("s", 5), 40, capacity_tokens=48, borrow=(kept, shared))
    rows = arena.capacity_rows(("s", 5))[:32]
    assert torch.equal(arena._pool[:, rows], written[:, :32])
    assert arena.store.stats()["saved_tokens"] == 0


def test_a_kept_prefix_computed_once_is_dropped_when_pages_run_short():
    arena = _arena(pages=8)
    _write(arena, ("r", 0), 40, seed=1)
    arena.note_computed(b"content 1")
    arena.free_key(("r", 0))
    # kept pages count as free, and a document that needs them takes them
    assert arena.free_pages == 8
    assert arena.reserve(("r", 1), 100, capacity_tokens=128)
    assert arena.kept_prefix(b"content 1", 40) == (None, 0)
    arena.flush_store()
    assert not arena.stored(b"content 1", 40)


def test_a_kept_prefix_computed_twice_leaves_for_the_host_store():
    arena = _arena(pages=8)
    written = _write(arena, ("r", 0), 40, seed=1)
    arena.note_computed(b"content 1")
    arena.note_computed(b"content 1")
    arena.free_key(("r", 0))
    assert arena.reserve(("r", 1), 100, capacity_tokens=128)
    arena.flush_store()
    assert arena.stored(b"content 1", 40)
    arena.free_key(("r", 1))
    assert arena.reserve(("r", 2), 40, capacity_tokens=48)
    arena.restore(("r", 2), b"content 1", 40)
    arena.flush_store()
    rows = arena.capacity_rows(("r", 2))[:40]
    assert torch.equal(arena._pool[:, rows], written)
