import pytest

from docengine.runtime.kv import (
    KVCapacityError,
    KVOwnershipError,
    KVPageAllocator,
)


def allocator(pages=8):
    return KVPageAllocator(
        total_pages=pages,
        page_size_tokens=16,
        bytes_per_token=72,
        activation_reserve_bytes=1024,
    )


def test_allocate_extend_truncate_and_free():
    pool = allocator()
    alloc = pool.allocate("doc", 17)
    assert alloc.page_ids == (0, 1)
    assert pool.free_pages == 6
    alloc = pool.extend("doc", 15)
    assert alloc.token_count == 32
    assert alloc.page_ids == (0, 1)
    pool.extend("doc", 1)
    assert pool.allocation("doc").page_ids == (0, 1, 2)
    pool.truncate("doc", 16)
    assert pool.allocation("doc").page_ids == (0,)
    pool.free("doc")
    assert pool.free_pages == 8


def test_shared_prefix_uses_reference_counts():
    pool = allocator()
    pool.allocate("document", 32)
    shared = pool.share_prefix("document", "tail", 32)
    assert shared.page_ids == (0, 1)
    assert pool.allocated_pages == 2
    pool.free("document")
    assert pool.allocated_pages == 2
    pool.free("tail")
    assert pool.allocated_pages == 0


def test_shared_prefix_must_end_on_page_boundary():
    pool = allocator()
    pool.allocate("document", 17)
    with pytest.raises(KVOwnershipError):
        pool.share_prefix("document", "tail", 17)


def test_no_implicit_eviction_on_capacity_failure():
    pool = allocator(pages=2)
    pool.allocate("a", 32)
    with pytest.raises(KVCapacityError):
        pool.allocate("b", 1)
    assert pool.allocation("a").page_ids == (0, 1)


def test_temporary_hbm_accounting():
    pool = allocator()
    pool.allocate("a", 16)
    pool.reserve_temporary(200)
    assert pool.hbm_bytes_used == 1024 + 16 * 72 + 200
    pool.release_temporary(200)
    assert pool.temporary_bytes == 0


def test_explicit_page_padding_for_fused_tails():
    pool = allocator()
    pool.allocate("document", 16)
    pool.share_prefix("document", "fused", 16)
    allocation = pool.extend_with_pages("fused", token_count=17, page_count=2)
    assert allocation.token_count == 33
    assert allocation.page_ids == (0, 1, 2)
    pool.free("fused")
    pool.free("document")
    assert pool.free_pages == 8


def test_double_free_is_rejected():
    pool = allocator()
    pool.allocate("a", 1)
    pool.free("a")
    with pytest.raises(KVOwnershipError):
        pool.free("a")
