"""Tests for PageArena page allocation, freeing, and row indexing."""

import random

import pytest

from quail.executor.arena import PageArena


def test_allocation_preserves_page_ownership():
    a = PageArena(n_pages=10, page_tokens=16)
    pages = a.alloc("d0", 40)      # 3 pages
    assert len(pages) == 3
    assert a.free_pages == 7
    assert a.resident_tokens == 40
    assert a.free_key("d0") == 3
    assert a.free_pages == 10
    assert a.resident_tokens == 0

    assert a.alloc("big", 200) is None
    assert a.alloc("d", 5, capacity_tokens=10) is not None
    assert len(a.row_indices("d")) == 5
    assert len(a.row_indices("d", 10)) == 10
    with pytest.raises(KeyError):
        a.alloc("d", 5)

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

    a = PageArena(n_pages=3, page_tokens=4)
    a.alloc("doc", 4)
    assert a.grow("doc", 12) == 2
    assert a.grow("doc", 16) is None


def test_retention_rewind_and_pinning():
    a = PageArena(n_pages=10, page_tokens=4)
    a.alloc("doc", 7, capacity_tokens=11)
    assert len(a.owned["doc"]) == 3

    a.pin("doc")
    assert "doc" in a.pinned
    a.retain("doc")
    assert "doc" not in a.pinned
    assert a.retained["doc"] == 7
    assert a.retained_prefix_tokens == 7

    assert a.rewind("doc", 4) == 2
    assert len(a.owned["doc"]) == 1
    assert a.tokens["doc"] == 4
    a.free_key("doc")
    assert not a.pinned
    assert not a.retained

    a = PageArena(n_pages=4, page_tokens=16)
    a.alloc("one-page", 16)
    a.alloc("two-pages", 17)
    a.retain("one-page")
    a.retain("two-pages")

    assert a.pop_retained_victim() == ("two-pages", 2, 17)

    a = PageArena(n_pages=2, page_tokens=16)
    a.alloc("doc", 16)
    a.retain("doc")
    a.pin("doc")

    assert a.pop_retained_victim() is None
