import random

import pytest

from quail.executor.retention import RetainedPool
from quail.planner.sol import prefix_recompute_seconds
from quail.planner.work import triangle
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_32B_FP8


def test_pool_admits_while_capacity_lasts():
    pool = RetainedPool(10)
    assert pool.offer("a", 4, 1.0) == (True, ())
    assert pool.offer("b", 4, 1.0) == (True, ())
    # 2 pages left; a 3-page doc must displace, but its value does
    # not strictly exceed a victim's
    assert pool.offer("c", 3, 1.0) == (False, ())
    assert pool.pages == 8
    assert len(pool) == 2


def test_pool_replaces_lowest_value_per_page_when_strictly_better():
    pool = RetainedPool(10)
    pool.offer("short-a", 5, 5.0)     # 1.0 per page
    pool.offer("short-b", 5, 10.0)    # 2.0 per page
    kept, victims = pool.offer("long", 5, 6.0)
    assert kept and victims == ("short-a",)
    assert "short-a" not in pool and "long" in pool
    assert pool.pages == 10
    assert pool.value == pytest.approx(16.0)


def test_pool_equal_value_never_swaps():
    pool = RetainedPool(4)
    pool.offer("first", 4, 7.0)
    assert pool.offer("second", 4, 7.0) == (False, ())
    assert "first" in pool


def test_pool_rejects_when_victims_cost_more():
    pool = RetainedPool(6)
    pool.offer("a", 3, 8.0)
    pool.offer("b", 3, 9.0)
    # needs both victims; 8 + 9 > 12, so nothing moves
    assert pool.offer("big", 6, 12.0) == (False, ())
    assert pool.pages == 6 and len(pool) == 2
    # a rejected offer must not disturb later victim ordering
    kept, victims = pool.offer("better", 3, 8.5)
    assert kept and victims == ("a",)


def test_pool_can_take_multiple_victims():
    pool = RetainedPool(6)
    pool.offer("a", 3, 1.0)
    pool.offer("b", 3, 2.0)
    kept, victims = pool.offer("big", 6, 4.0)
    assert kept and set(victims) == {"a", "b"}
    assert pool.pages == 6 and pool.value == pytest.approx(4.0)


def test_pool_rejects_oversized_and_zero_capacity():
    pool = RetainedPool(4)
    pool.offer("a", 2, 1.0)
    assert pool.offer("big", 5, 100.0) == (False, ())
    assert pool.offer("a2", 3, 100.0) == (True, ("a",))
    empty = RetainedPool(0)
    assert empty.offer("x", 1, 1.0) == (False, ())


def test_pool_discard_forgets_outside_evictions():
    pool = RetainedPool(6)
    pool.offer("a", 3, 1.0)
    pool.offer("b", 3, 2.0)
    pool.discard("a")
    assert pool.pages == 3 and "a" not in pool
    # the stale heap record for "a" must not surface as a victim
    kept, victims = pool.offer("c", 6, 5.0)
    assert kept and victims == ("b",)
    pool.discard("missing")    # unknown keys are a no-op


def test_pool_value_only_rises_under_random_offers():
    rng = random.Random(11)
    for _ in range(50):
        pool = RetainedPool(rng.randrange(1, 40))
        total = 0.0
        for i in range(200):
            pages = rng.randrange(1, 8)
            value = rng.random() * pages   # roughly value ~ pages
            kept, victims = pool.offer(i, pages, value)
            assert pool.pages <= pool.cap_pages
            assert pool.value >= total - 1e-9
            total = pool.value
            assert kept == (i in pool)
            assert all(v not in pool for v in victims)


def test_prefix_value_uses_dense_tokens_and_attention_pairs():
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        one = prefix_recompute_seconds(1, model, H100_SXM)
        two = prefix_recompute_seconds(2, model, H100_SXM)
        assert one > 0
        assert two > 2 * one
        assert triangle(2) == 3

    with pytest.raises(ValueError):
        prefix_recompute_seconds(-1, QWEN3_4B_FP8, H100_SXM)
