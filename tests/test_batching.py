from docengine.runtime.batching import (
    BatchLimits,
    VariableLengthBatchPacker,
    WorkItem,
    WorkKind,
)
from docengine.runtime.kv import KVPageAllocator


def pool(pages=64):
    return KVPageAllocator(
        total_pages=pages,
        page_size_tokens=16,
        bytes_per_token=72,
    )


def work(
    name,
    tokens,
    *,
    offset=0,
    cached=0,
    temporary=0,
    kind=WorkKind.PREFILL,
    document=0,
):
    return WorkItem(
        work_id=name,
        owner=name,
        document_id=document,
        filter_start=0,
        k=1,
        kind=kind,
        token_offset=offset,
        total_new_tokens=tokens,
        cached_prefix_tokens=cached,
        temporary_bytes=temporary,
    )


def test_packs_exact_variable_length_chunks():
    packer = VariableLengthBatchPacker(BatchLimits(
        max_new_tokens=16,
        max_sequences=4,
        max_temporary_bytes=100,
    ))
    batch = packer.pack(
        [work("a", 3), work("b", 20, document=1)],
        pool(),
    )
    assert [chunk.new_tokens for chunk in batch.chunks] == [3, 13]
    assert batch.total_new_tokens == 16
    assert batch.unused_capacity_reason == "token_limit"


def test_sequence_limit_is_independent_of_token_limit():
    packer = VariableLengthBatchPacker(BatchLimits(
        max_new_tokens=100,
        max_sequences=2,
        max_temporary_bytes=100,
    ))
    batch = packer.pack(
        [work("a", 3), work("b", 4), work("c", 5)],
        pool(),
    )
    assert batch.sequence_count == 2
    assert batch.total_new_tokens == 7
    assert batch.unused_capacity_reason == "sequence_limit"


def test_hbm_limit_truncates_chunk_without_eviction():
    kv = pool(pages=2)
    packer = VariableLengthBatchPacker(BatchLimits(
        max_new_tokens=100,
        max_sequences=4,
        max_temporary_bytes=100,
    ))
    batch = packer.pack([work("a", 100)], kv)
    assert batch.total_new_tokens == 32
    assert batch.kv_pages_added == 2
    assert kv.free_pages == 2
    assert batch.unused_capacity_reason == "hbm_limit"


def test_existing_partial_page_contributes_capacity():
    kv = pool(pages=1)
    kv.allocate("a", 15)
    packer = VariableLengthBatchPacker(BatchLimits(
        max_new_tokens=4,
        max_sequences=1,
        max_temporary_bytes=0,
    ))
    item = work("a", 19, offset=15, cached=15)
    batch = packer.pack([item], kv)
    assert batch.total_new_tokens == 1
    assert batch.kv_pages_added == 0


def test_temporary_hbm_limit_skips_work_that_does_not_fit():
    packer = VariableLengthBatchPacker(BatchLimits(
        max_new_tokens=16,
        max_sequences=2,
        max_temporary_bytes=5,
    ))
    batch = packer.pack(
        [work("a", 4, temporary=6), work("b", 4, temporary=2)],
        pool(),
    )
    assert [chunk.work_id for chunk in batch.chunks] == ["b"]
    assert batch.unused_capacity_reason == "temporary_hbm_limit"


def test_completed_ready_set_reports_no_ready_work():
    packer = VariableLengthBatchPacker(BatchLimits(
        max_new_tokens=16,
        max_sequences=4,
        max_temporary_bytes=100,
    ))
    batch = packer.pack([work("a", 3)], pool())
    assert batch.unused_capacity_reason == "no_ready_useful_work"
