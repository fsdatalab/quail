"""The admission scheduler's restore/deferred-release behavior (the
pack.py logic that still exists without the CPU store)."""

import pytest

from quail.executor.pack import FilterAdmission


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


def test_default_release_unchanged():
    sched = FilterAdmission([160], [10], chunk_budget=400,
                            arena_pages=10, page_tokens=16)
    sched.next_chunk()
    sched.report(0, 0, False)
    assert sched.free_pages == 10
