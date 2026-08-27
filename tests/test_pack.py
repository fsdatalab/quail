"""Tests for pack_stream, FilterAdmission, and gate/assemble against brute force."""

import random

import pytest

from quail.executor.pack import (FilterAdmission, assemble,
                                 brute_force_triples, gate, matches,
                                 orient, pack_stream, pages_for,
                                 plan_groups)


def test_orient_prefers_longer_side():
    assert orient(1000, 8) == "left"
    assert orient(8, 1000) == "right"
    assert orient(50, 50) == "left"


def test_plan_groups_budget_respected_random():
    rng = random.Random(7)
    for _ in range(50):
        budget = rng.randrange(2_000, 30_000)
        prefix = rng.randrange(100, budget - 600)
        suffixes = [rng.randrange(20, min(600, budget - prefix))
                    for _ in range(rng.randrange(1, 400))]
        groups = plan_groups(prefix, suffixes, budget)
        seen = []
        for start, end in groups:
            assert prefix + sum(suffixes[start:end]) <= budget
            seen.extend(range(start, end))
        assert seen == list(range(len(suffixes)))


def test_pack_stream_every_suffix_once_in_order():
    rng = random.Random(11)
    for _ in range(50):
        budget = rng.randrange(3_000, 40_000)
        anchors = []
        for _ in range(rng.randrange(1, 30)):
            prefix = rng.randrange(50, budget // 3)
            room = budget - prefix
            suffixes = [rng.randrange(10, max(11, room // 2))
                        for _ in range(rng.randrange(0, 60))]
            anchors.append((prefix, suffixes))
        chunks, kv_to_cache = pack_stream(anchors, budget)
        covered = {a: [] for a in range(len(anchors))}
        carried_count = {a: 0 for a in range(len(anchors))}
        for chunk in chunks:
            assert chunk
            tokens = 0
            for a, start, end, carried in chunk:
                prefix, suffixes = anchors[a]
                tokens += (prefix if carried else 0) \
                    + sum(suffixes[start:end])
                covered[a].extend(range(start, end))
                carried_count[a] += carried
            assert tokens <= budget
        for a, (prefix, suffixes) in enumerate(anchors):
            assert covered[a] == list(range(len(suffixes)))
            # the computed-exactly-once invariant
            assert carried_count[a] == (1 if suffixes else 0)


def test_pack_stream_cut_stream_continues_without_prefix():
    chunks, kv_to_cache = pack_stream([(100, [400, 400, 400])], 600)
    assert chunks == [[(0, 0, 1, True)],
                      [(0, 1, 2, False)],
                      [(0, 2, 3, False)]]
    assert kv_to_cache == {0}


def test_pack_stream_keep_and_already_kept():
    chunks, kv = pack_stream([(100, [50, 50])], 1000, keep={0})
    assert chunks == [[(0, 0, 2, True)]]
    assert kv == {0}
    chunks, kv = pack_stream([(100, [400, 400])], 600,
                             already_kept={0})
    assert chunks == [[(0, 0, 1, False)], [(0, 1, 2, False)]]
    assert kv == set()


def test_pack_stream_atomicity_error():
    with pytest.raises(ValueError):
        pack_stream([(100, [950])], 1000)


def test_gate_matches_assemble_vs_brute_force():
    rng = random.Random(3)
    for _ in range(30):
        nA, nB, nC = (rng.randrange(1, 12) for _ in range(3))
        ans1 = {b: [1 if rng.random() < 0.4 else 0 for _ in range(nA)]
                for b in range(nB)}
        survivors = gate(ans1)
        ans2 = {b: [1 if rng.random() < 0.3 else 0 for _ in range(nC)]
                for b in survivors}
        assert assemble(ans1, ans2) == brute_force_triples(ans1, ans2)
    rows = {0: [0, 0], 1: [0, 1]}
    assert gate(rows) == [1]
    assert matches(rows) == {0: [], 1: [1]}


# ------------------------------------------- the admission simulator

def _drive(sched, truth, deliver_lag=1, rng=None):
    """Drive the scheduler to completion, delivering answers with the given lag. Returns per-chunk group lists."""
    chunks, outstanding = [], []
    idle = 0
    while not sched.done():
        groups = sched.next_chunk()
        if groups:
            chunks.append(groups)
            outstanding.append(groups)
            idle = 0
        else:
            idle += 1
            assert sched.in_flight or outstanding, \
                "no chunk buildable and nothing in flight: stuck"
            assert idle < 3, "scheduler stopped making progress"
        while len(outstanding) > (deliver_lag if groups else 0):
            for doc, stage, _fresh in outstanding.pop(0):
                sched.report(doc, stage, truth[doc][stage])
    return chunks


def _check_invariants(sched, chunks, truth, doc_tokens, stage_tokens,
                      budget, arena_pages, page_tokens):
    fresh_count = {}
    for groups in chunks:
        tokens = 0
        for doc, stage, fresh in groups:
            tokens += stage_tokens[stage] + (doc_tokens[doc] if fresh
                                             else 0)
            if fresh:
                fresh_count[doc] = fresh_count.get(doc, 0) + 1
                assert stage == 0
        assert tokens <= budget, "over-budget chunk"
    # every document was admitted exactly once (computed once, ever)
    assert fresh_count == {d: 1 for d in range(len(doc_tokens))}
    # answers match the planted truth up to the first FALSE
    for d, row in enumerate(truth):
        expect = []
        for j, v in enumerate(row):
            expect.append(v)
            if not v:
                break
        assert sched.answers[d] == expect, f"doc {d} gating wrong"
    # all pages returned
    assert sched.free_pages == arena_pages
    assert sched.survivors() == [d for d, row in enumerate(truth)
                                 if all(row)]


def test_admission_simulator_random_shapes():
    rng = random.Random(23)
    for trial in range(20):
        n_docs = rng.randrange(5, 60)
        n_stages = rng.randrange(1, 6)
        page_tokens = 16
        doc_tokens = [rng.randrange(20, 900) for _ in range(n_docs)]
        stage_tokens = [rng.randrange(10, 60) for _ in range(n_stages)]
        budget = max(doc_tokens) + max(stage_tokens) \
            + rng.randrange(0, 2000)
        # arena sometimes tight (forces waiting), never below one doc
        arena_pages = max(pages_for(max(doc_tokens), page_tokens),
                          rng.randrange(4, 200))
        truth = [[1 if rng.random() < 0.7 else 0
                  for _ in range(n_stages)] for _ in range(n_docs)]
        sched = FilterAdmission(doc_tokens, stage_tokens, budget,
                                arena_pages, page_tokens)
        chunks = _drive(sched, truth, deliver_lag=rng.choice((0, 1)),
                        rng=rng)
        _check_invariants(sched, chunks, truth, doc_tokens,
                          stage_tokens, budget, arena_pages,
                          page_tokens)


def test_admission_survivor_priority():
    # one resident survivor's next suffix packs before fresh docs
    sched = FilterAdmission([100, 100], [10, 10], 200,
                            arena_pages=100, page_tokens=16)
    first = sched.next_chunk()
    assert first == [(0, 0, True)]     # doc 1 skipped: no room (110+110)
    sched.report(0, 0, True)
    second = sched.next_chunk()
    assert second[0] == (0, 1, False)  # survivor suffix leads
    assert (1, 0, True) in second


def test_admission_pages_block_in_order():
    # arena holds one big doc; the second waits for pages even though
    # it would fit the chunk
    sched = FilterAdmission([160, 160], [10], 400,
                            arena_pages=10, page_tokens=16)
    first = sched.next_chunk()
    assert first == [(0, 0, True)]
    assert sched.next_chunk() == []    # pages blocked, answer in flight
    sched.report(0, 0, False)          # FALSE frees the pages
    assert sched.next_chunk() == [(1, 0, True)]


def test_admission_starts_with_only_currently_free_pages():
    sched = FilterAdmission([80], [10], 200,
                            arena_pages=10, page_tokens=16,
                            available_pages=2)

    assert sched.next_chunk() == []
    assert sched.blocked_pages == 3
    sched.add_free_pages(3)
    assert sched.next_chunk() == [(0, 0, True)]


def test_admission_can_leave_a_passing_document_resident():
    sched = FilterAdmission([80], [10], 200,
                            arena_pages=10, page_tokens=16)
    assert sched.next_chunk() == [(0, 0, True)]

    assert sched.report(0, 0, True, release=False) == ()
    assert sched.free_pages == 5


def test_admission_refuses_impossible_shapes():
    with pytest.raises(ValueError):
        FilterAdmission([1000], [10], 500, 100, 16)     # over chunk
    with pytest.raises(ValueError):
        FilterAdmission([1000], [10], 2000, 2, 16)      # over arena


def test_admission_limit_drains_in_flight():
    # limit=1 with 2 docs admitted in the same chunk: the second
    # answer still lands (in_flight drains) even though the limit is
    # already met
    sched = FilterAdmission([50, 50], [10], 200,
                            arena_pages=100, page_tokens=16, limit=1)
    groups = sched.next_chunk()
    assert len(groups) == 2
    sched.report(0, 0, True)
    assert sched._survivor_count == 1
    # done() is False because doc 1 is still in_flight
    assert not sched.done()
    sched.report(1, 0, True)
    assert sched.done()
    assert sched._survivor_count == 2


def test_admission_limit_drain_ready_returns_stranded():
    # 3 docs, 2 stages, limit=1. All pass stage 0. The stage-1 suffix
    # is 200 tokens, so the next chunk holds only doc 0's; it passes
    # and meets the limit while docs 1 and 2 still sit in ready.
    # drain_ready returns them and restores their pages.
    sched = FilterAdmission([50, 50, 50], [10, 200], 250,
                            arena_pages=100, page_tokens=16, limit=1)
    for doc, _, _ in sched.next_chunk():        # all fresh, stage 0
        sched.report(doc, 0, True)
    groups = sched.next_chunk()
    assert groups == [(0, 1, False)]            # room for one suffix
    sched.report(0, 1, True)                    # limit reached
    assert sched.done()
    assert sched.drain_ready() == [1, 2]
    assert sched.free_pages == 100
    assert not sched.resident


def test_admission_limit_reduces_work():
    """Verify that a limit reduces admitted documents and chunks compared to an unlimited run."""
    rng = random.Random(42)
    n_docs = 80
    n_stages = 3
    doc_tokens = [rng.randrange(30, 200) for _ in range(n_docs)]
    stage_tokens = [rng.randrange(10, 40) for _ in range(n_stages)]
    budget = max(doc_tokens) + sum(stage_tokens) + 300
    arena_pages = max(pages_for(max(doc_tokens), 16), 60)
    truth = [[1 if rng.random() < 0.8 else 0
              for _ in range(n_stages)] for _ in range(n_docs)]
    limit = 5

    # unlimited run
    sched_all = FilterAdmission(doc_tokens, stage_tokens, budget,
                                arena_pages, 16, limit=None)
    chunks_all = _drive(sched_all, truth, deliver_lag=0)
    admitted_all = len(sched_all.answers)

    # limited run (same truth table, same parameters)
    sched_lim = FilterAdmission(doc_tokens, stage_tokens, budget,
                                arena_pages, 16, limit=limit)
    chunks_lim = _drive(sched_lim, truth, deliver_lag=0)
    admitted_lim = len(sched_lim.answers)

    # the limited run must have found enough survivors
    assert len(sched_lim.survivors()) >= limit

    # the limited run admitted strictly fewer documents
    assert admitted_lim < admitted_all, (
        f"limit={limit} admitted {admitted_lim}, unlimited admitted "
        f"{admitted_all}; early termination did not reduce work")

    # the limited run built fewer chunks
    assert len(chunks_lim) < len(chunks_all), (
        f"limit={limit} built {len(chunks_lim)} chunks, unlimited "
        f"built {len(chunks_all)}; expected fewer chunks")


# ----------------------------- no page bin (single-stage, no store) --

def test_admission_no_page_bin_validation():
    # a document too big for any arena is admitted when there is no
    # page bin; a document too big for the chunk is still refused
    sched = FilterAdmission([1000], [10], 2000, None, 16)
    assert sched.next_chunk() == [(0, 0, True)]
    with pytest.raises(ValueError):
        FilterAdmission([1000], [10], 500, None, 16)


def test_admission_no_page_bin_random_shapes():
    rng = random.Random(29)
    for _ in range(20):
        n_docs = rng.randrange(5, 60)
        doc_tokens = [rng.randrange(20, 900) for _ in range(n_docs)]
        stage_tokens = [rng.randrange(10, 60)]
        budget = max(doc_tokens) + stage_tokens[0] \
            + rng.randrange(0, 2000)
        truth = [[1 if rng.random() < 0.7 else 0]
                 for _ in range(n_docs)]
        sched = FilterAdmission(doc_tokens, stage_tokens, budget,
                                None, 16)
        chunks = _drive(sched, truth, deliver_lag=rng.choice((0, 1)))
        fresh_count = {}
        for groups in chunks:
            tokens = 0
            for doc, stage, fresh in groups:
                assert fresh and stage == 0
                tokens += stage_tokens[0] + doc_tokens[doc]
                fresh_count[doc] = fresh_count.get(doc, 0) + 1
            assert tokens <= budget, "over-budget chunk"
        # every document admitted exactly once, no page accounting
        assert fresh_count == {d: 1 for d in range(n_docs)}
        assert sched.free_pages is None
        assert not sched.resident
        for d, row in enumerate(truth):
            assert sched.answers[d] == row
        assert sched.survivors() == [d for d, row in enumerate(truth)
                                     if all(row)]


def test_admission_no_page_bin_with_limit():
    # LIMIT composes with the fast path: admission stops at the
    # survivor target, still with no page accounting
    n_docs, limit = 40, 3
    truth = [[1]] * n_docs
    sched = FilterAdmission([100] * n_docs, [10], 230, None, 16,
                            limit=limit)
    chunks = _drive(sched, truth, deliver_lag=0)
    admitted = sum(len(g) for g in chunks)
    assert len(sched.survivors()) >= limit
    assert admitted < n_docs
    assert sched.free_pages is None and not sched.resident
