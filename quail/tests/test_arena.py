"""PageArena accounting: pages out, pages back, rows where they
should be. The tensor-backed KVArena runs only on GPU and is covered
by the milestone 1 parity probe."""

import random

import pytest

from quail.executor.arena import PageArena


def test_alloc_free_roundtrip():
    a = PageArena(n_pages=10, page_tokens=16)
    pages = a.alloc("d0", 40)      # 3 pages
    assert len(pages) == 3
    assert a.free_pages == 7
    assert a.resident_tokens == 40
    assert a.free_key("d0") == 3
    assert a.free_pages == 10
    assert a.resident_tokens == 0


def test_alloc_returns_none_when_short():
    a = PageArena(n_pages=2, page_tokens=16)
    assert a.alloc("big", 100) is None      # needs 7 pages
    assert a.free_pages == 2                # nothing leaked
    assert a.alloc("ok", 32) is not None
    assert a.alloc("more", 1) is None       # free list empty


def test_double_alloc_rejected():
    a = PageArena(n_pages=4, page_tokens=16)
    a.alloc("d", 16)
    with pytest.raises(KeyError):
        a.alloc("d", 16)


def test_row_indices_follow_pages():
    a = PageArena(n_pages=4, page_tokens=4)
    pages = a.alloc("d", 10)       # 3 pages, last partially filled
    rows = a.row_indices("d")
    assert len(rows) == 10
    expect = []
    left = 10
    for p in pages:
        take = min(left, 4)
        expect.extend(range(p * 4, p * 4 + take))
        left -= take
    assert rows == expect


def test_no_page_shared_between_documents():
    rng = random.Random(5)
    a = PageArena(n_pages=64, page_tokens=16)
    live = {}
    for step in range(500):
        if live and rng.random() < 0.45:
            key = rng.choice(sorted(live))
            a.free_key(key)
            del live[key]
        else:
            key = f"d{step}"
            got = a.alloc(key, rng.randrange(1, 200))
            if got is not None:
                live[key] = got
        held = [p for pages in live.values() for p in pages]
        assert len(held) == len(set(held)), "page double-owned"
        assert len(held) + a.free_pages == 64, "pages leaked"
