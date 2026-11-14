"""Tests for the host KV store: its codecs, extents, and eviction."""

import pytest
import torch

from quail.backends.quail.executor.kv_store import KVStore, content_id

TENSORS, HEADS, DIM = 4, 2, 64


def _pool(rows=256, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(0)
    return torch.randn((TENSORS, rows, HEADS, DIM), generator=generator).to(dtype)


def _store(codec, capacity_bytes=1 << 22, dtype=torch.bfloat16):
    return KVStore(torch, (TENSORS, HEADS, DIM), dtype, capacity_bytes,
                   codec=codec, device="cpu")


def _round_trip(codec):
    """Save 40 scattered rows, then restore them to 40 other rows."""
    pool = _pool()
    store = _store(codec)
    source = torch.arange(3, 203, 5)
    target = torch.arange(210, 250)
    store.save(b"document", source)
    store.flush(pool)
    assert store.has(b"document", 40) and not store.has(b"document", 39)
    store.restore(b"document", target)
    store.flush(pool)
    return pool[:, source].float(), pool[:, target].float(), store


def test_a_raw_prefix_restores_bit_for_bit():
    saved, restored, store = _round_trip("raw")
    assert torch.equal(saved, restored)
    assert store.record_bytes == TENSORS * HEADS * DIM * 2
    assert store.stats()["restored_tokens"] == 40


@pytest.mark.parametrize("codec, levels, ratio",
                         [("int8", 127, 0.52), ("int4", 7, 0.29)])
def test_a_quantized_prefix_restores_within_half_a_step(codec, levels, ratio):
    saved, restored, store = _round_trip(codec)
    group = store.group
    largest = saved.reshape(TENSORS, 40, -1, group).abs().amax(-1, keepdim=True)
    error = (saved - restored).reshape(TENSORS, 40, -1, group).abs()
    # half a step, and the bf16 rounding of the restored value
    assert (error <= largest / levels * 0.5 + largest / 128).all()
    assert store.record_bytes <= ratio * TENSORS * HEADS * DIM * 2


def test_a_full_store_drops_the_prefix_used_longest_ago():
    pool = _pool(rows=4096)
    store = _store("raw", capacity_bytes=3 * 1024 * TENSORS * HEADS * DIM * 2)
    rows = torch.arange(1024)
    for name in (b"a", b"b", b"c"):
        store.save(name, rows)
    store.flush(pool)
    store.restore(b"a", rows + 1024)       # "b" is now the oldest
    store.flush(pool)
    store.save(b"d", rows)
    store.flush(pool)
    assert [store.has(name, 1024) for name in (b"a", b"b", b"c", b"d")] == [
        True, False, True, True]
    assert store.stats()["dropped_tokens"] == 1024
    assert torch.equal(pool[:, rows + 1024], pool[:, rows])


def test_a_prefix_larger_than_the_store_is_not_saved():
    store = _store("raw", capacity_bytes=1024 * TENSORS * HEADS * DIM * 2)
    store.save(b"long", torch.arange(2000))
    assert not store.has(b"long", 2000) and store.held_tokens == 0


def test_a_content_id_names_the_tokens_not_their_container():
    assert content_id([5, 6, 7]) == content_id((5, 6, 7))
    assert content_id([5, 6, 7]) != content_id([5, 6])
