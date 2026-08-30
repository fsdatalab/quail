import random

from quail.executor.retention import RetainedPool


def test_pool_admits_while_capacity_lasts():
    pool = RetainedPool(10)
    assert pool.offer("a", 4, 50) == (True, ())
    assert pool.offer("b", 4, 50) == (True, ())
    # 2 pages left; a 3-page doc must displace, but its prefix does
    # not contain more tokens than a victim's prefix
    assert pool.offer("c", 3, 37) == (False, ())
    assert pool.pages == 8
    assert len(pool) == 2


def test_pool_replaces_lowest_prefix_tokens_per_page_when_better():
    pool = RetainedPool(10)
    pool.offer("short-a", 5, 70)     # 14 tokens per page
    pool.offer("short-b", 5, 75)     # 15 tokens per page
    kept, victims = pool.offer("long", 5, 72)
    assert kept and victims == ("short-a",)
    assert "short-a" not in pool and "long" in pool
    assert pool.pages == 10
    assert pool.prefix_tokens == 147


def test_pool_equal_prefix_tokens_never_swaps():
    pool = RetainedPool(4)
    pool.offer("first", 4, 60)
    assert pool.offer("second", 4, 60) == (False, ())
    assert "first" in pool


def test_pool_rejects_when_victims_cost_more():
    pool = RetainedPool(6)
    pool.offer("a", 3, 40)
    pool.offer("b", 3, 42)
    # needs both victims; 40 + 42 > 80, so nothing moves
    assert pool.offer("big", 6, 80) == (False, ())
    assert pool.pages == 6 and len(pool) == 2
    # a rejected offer must not disturb later victim ordering
    kept, victims = pool.offer("better", 3, 41)
    assert kept and victims == ("a",)


def test_pool_can_take_multiple_victims():
    pool = RetainedPool(6)
    pool.offer("a", 3, 30)
    pool.offer("b", 3, 33)
    kept, victims = pool.offer("big", 6, 64)
    assert kept and set(victims) == {"a", "b"}
    assert pool.pages == 6 and pool.prefix_tokens == 64


def test_pool_rejects_oversized_and_zero_capacity():
    pool = RetainedPool(4)
    pool.offer("a", 2, 20)
    assert pool.offer("big", 5, 100) == (False, ())
    assert pool.offer("a2", 3, 40) == (True, ("a",))
    empty = RetainedPool(0)
    assert empty.offer("x", 1, 1) == (False, ())


def test_pool_discard_forgets_outside_evictions():
    pool = RetainedPool(6)
    pool.offer("a", 3, 30)
    pool.offer("b", 3, 33)
    pool.discard("a")
    assert pool.pages == 3 and "a" not in pool
    # the stale heap record for "a" must not surface as a victim
    kept, victims = pool.offer("c", 6, 64)
    assert kept and victims == ("b",)
    pool.discard("missing")    # unknown keys are a no-op


def test_pool_prefix_tokens_only_rise_under_random_offers():
    rng = random.Random(11)
    for _ in range(50):
        pool = RetainedPool(rng.randrange(1, 40))
        total = 0
        for i in range(200):
            pages = rng.randrange(1, 8)
            prefix_tokens = rng.randrange(1, pages * 16 + 1)
            kept, victims = pool.offer(i, pages, prefix_tokens)
            assert pool.pages <= pool.cap_pages
            assert pool.prefix_tokens >= total
            total = pool.prefix_tokens
            assert kept == (i in pool)
            assert all(v not in pool for v in victims)
