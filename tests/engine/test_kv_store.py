"""Tests for the host KV store: its codecs, extents, and eviction."""

import torch

from quail.backends.quail.executor.kv_store import KVStore, content_id

TENSORS, HEADS, DIM = 4, 2, 64


def _pool(rows=256, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(0)
    return torch.randn((TENSORS, rows, HEADS, DIM), generator=generator).to(dtype)


def test_a_saved_prefix_restores_to_other_rows_bit_for_bit():
    pool = _pool()
    store = KVStore(torch, (TENSORS, HEADS, DIM), torch.bfloat16, 1 << 22,
                    device="cpu")
    source = torch.arange(3, 203, 5)
    target = torch.arange(210, 250)
    store.save(b"document", source)
    store.flush(pool)
    assert store.has(b"document", 40) and not store.has(b"document", 39)
    store.restore(b"document", target)
    store.flush(pool)
    assert torch.equal(pool[:, source], pool[:, target])
    assert store.record_bytes == TENSORS * HEADS * DIM * 2
    assert store.stats()["restored_tokens"] == 40


def _full_store(pool):
    """A store of 3072 records holding a long prefix and two short ones."""
    store = KVStore(torch, (TENSORS, HEADS, DIM), torch.bfloat16,
                    3072 * TENSORS * HEADS * DIM * 2, device="cpu",
                    # a pass over a prefix costs its tokens and its token pairs
                    prefill_seconds=lambda tokens: tokens + tokens * tokens / 2)
    store.save(b"long", torch.arange(2048))
    store.save(b"short", torch.arange(512))
    store.save(b"other", torch.arange(512))
    store.flush(pool)
    return store


def _held(store):
    return {name for name in (b"long", b"short", b"other", b"new")
            if name in store.saved}


def test_a_full_store_drops_short_prefixes_before_a_long_one():
    pool = _pool(rows=4096)
    store = _full_store(pool)
    store.save(b"new", torch.arange(600))
    store.flush(pool)
    # the long prefix saves the most seconds per byte, so both short ones left
    assert _held(store) == {b"long", b"new"}
    assert store.stats()["dropped_tokens"] == 1024


def test_a_long_prefix_no_query_reads_leaves_after_short_ones_in_use():
    pool = _pool(rows=8192)
    store = _full_store(pool)
    # each eviction raises the clock under the prefixes restored after it
    for step in range(8):
        store.save(b"filler %d" % step, torch.arange(512))
        store.restore(b"filler %d" % step, torch.arange(4096, 4608))
        store.flush(pool)
    assert b"long" not in store.saved
    assert torch.equal(pool[:, 4096:4608], pool[:, :512])


def test_a_prefix_larger_than_the_store_is_skipped():
    store = KVStore(torch, (TENSORS, HEADS, DIM), torch.bfloat16,
                    1024 * TENSORS * HEADS * DIM * 2, device="cpu")
    store.save(b"long", torch.arange(2000))
    assert not store.has(b"long", 2000) and store.held_tokens == 0
    assert store.stats()["skipped_tokens"] == 2000


def test_a_content_id_names_the_tokens_not_their_container():
    assert content_id([5, 6, 7]) == content_id((5, 6, 7))
    assert content_id([5, 6, 7]) != content_id([5, 6])
