"""The store's CPU logic: extent allocation, the length threshold,
and the admission scheduler's restore/deferred-release behavior. The
pinned pool and the transfers are GPU-side and covered by the
milestone store gate."""

import random

import pytest

from quail.executor.kvstore import ExtentAllocator, alloc_with_reclaim
from quail.executor.pack import FilterAdmission
from quail.planner.decide import store_length_threshold


# --------------------------------------------------- extent allocator

def test_alloc_free_coalesce_roundtrip():
    a = ExtentAllocator(100)
    x = a.alloc(40)
    y = a.alloc(40)
    assert (x, y) == (0, 40)
    assert a.alloc(30) is None          # only 20 left
    a.free(x, 40)
    a.free(y, 40)
    # coalesced back into one extent: an 80-row doc fits again
    assert a.alloc(80) == 0
    assert a.free_rows == 20


def test_allocator_random_invariants():
    rng = random.Random(9)
    a = ExtentAllocator(1000)
    live = {}
    for step in range(2000):
        if live and rng.random() < 0.5:
            key = rng.choice(sorted(live))
            off, n = live.pop(key)
            a.free(off, n)
        else:
            n = rng.randrange(1, 60)
            off = a.alloc(n)
            if off is not None:
                for o, ln in live.values():
                    assert off + n <= o or off >= o + ln, "overlap"
                live[step] = (off, n)
        assert a.used == sum(n for _, n in live.values())
    for off, n in live.values():
        a.free(off, n)
    assert a.free_rows == 1000
    assert a.free_list == [(0, 1000)]


# --------------------------------------- cross-dataset reclamation

def test_reclaim_evicts_idle_dataset_not_own():
    # pool of 100 rows, dataset A fills it; dataset B's query must
    # claim space from A (shortest A doc first), never from B itself
    allocs = [ExtentAllocator(100)]
    extents = {}
    for i, tk in enumerate((40, 35, 25)):       # A: 100 rows total
        slab, off = alloc_with_reclaim(allocs, extents, tk, "A")
        extents[("A", i)] = (slab, off, tk)
    assert allocs[0].free_rows == 0

    got = alloc_with_reclaim(allocs, extents, 20, keep_hash="B")
    assert got is not None
    # the shortest A extent (25) was evicted, nothing else
    assert ("A", 2) not in extents
    assert ("A", 0) in extents and ("A", 1) in extents

    # The active dataset is never evicted.
    assert alloc_with_reclaim(allocs, extents, 90, "A") is None
    assert ("A", 0) in extents and ("A", 1) in extents

    allocs = [ExtentAllocator(90)]
    extents = {}
    for i in range(3):
        slab, off = alloc_with_reclaim(allocs, extents, 30, "A")
        extents[("A", i)] = (slab, off, 30)
    assert alloc_with_reclaim(allocs, extents, 60, "B") is not None
    assert len([k for k in extents if k[0] == "A"]) == 1


# ------------------------------------------------- length threshold

def test_store_length_threshold_boundaries():
    assert store_length_threshold([100, 200, 300], capacity_bytes=1e9,
                                  kappa=1000) == 1
    t = store_length_threshold([100, 200, 300], capacity_bytes=500,
                               kappa=1)
    assert t == 200
    assert sum(x for x in [100, 200, 300] if x >= t) <= 500
    assert store_length_threshold([100, 200], capacity_bytes=50,
                                  kappa=1) == 0


# --------------------------------- admission: restore + deferred free

def test_restored_doc_costs_question_only():
    sched = FilterAdmission([100, 100], [10], chunk_budget=130,
                            arena_pages=100, page_tokens=16,
                            restored={0})
    groups = sched.next_chunk()
    # doc 0 restored (cost 10) + doc 1 fresh (cost 110) = 120 <= 130;
    # without the restore discount only one would fit
    assert groups == [(0, 0, True), (1, 0, True)]


def test_deferred_release_holds_pages_until_release():
    sched = FilterAdmission([160], [10], chunk_budget=400,
                            arena_pages=10, page_tokens=16)
    assert sched.next_chunk() == [(0, 0, True)]
    sched.report(0, 0, False, release=False)   # store copy in flight
    assert sched.free_pages == 0               # pages still held
    assert 0 in sched.resident
    assert sched.done()                        # but no work remains
    sched.release(0)
    assert sched.free_pages == 10
    assert 0 not in sched.resident
