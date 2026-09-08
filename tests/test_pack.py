"""Tests for JoinAdmission, FilterAdmission, and gate/assemble against brute force."""

import random

import pytest

from quail.executor.pack import (
    FilterAdmission,
    JoinAdmission,
    assemble,
    brute_force_triples,
    gate,
    matches,
    pages_for,
)

# ------------------------------------------------ join admission


def _drive_join(sched, truth, arena_pages, resident=None,
                deliver_lag=1, rng=None):
    """Drive a JoinAdmission to completion against a simulated free list.

    truth[a][j] is the 0/1 row for anchor a at stage j.

    Returns:
        (chunks, events) in launch order.
    """
    resident = resident or {}
    extra = max(sched.frames)
    held = dict(resident)
    free = arena_pages - sum(held.values())
    chunks, outstanding, events = [], [], []
    idle = 0

    def settle(a):
        nonlocal free
        free += held.pop(a)

    while not sched.done():
        groups = sched.next_chunk(free)
        if groups:
            for a, j, start, end, carried in groups:
                if j == 0 and start == 0:
                    need = pages_for(sched.prefix[a] + extra,
                                     sched.page_tokens)
                    free -= need - held.get(a, 0)
                    assert free >= 0, "admitted past the free list"
                    held[a] = need
                    assert carried == (a not in resident)
            chunks.append(groups)
            outstanding.append(groups)
            idle = 0
        else:
            idle += 1
            assert outstanding, \
                "no chunk buildable and nothing in flight: stuck"
            assert idle < 3, "scheduler stopped making progress"
        lag = deliver_lag if groups else 0
        if rng is not None and groups:
            lag = rng.choice((0, deliver_lag))
        while len(outstanding) > lag:
            for a, j, start, end, _ in outstanding.pop(0):
                bits = truth[a][j][start:end]
                for kind, anchor in sched.report(a, j, start, end, bits):
                    events.append((kind, anchor))
                    settle(anchor)
    stranded = {a for a, st in enumerate(sched._stage) if st == -3}
    assert free == arena_pages - sum(held[a] for a in stranded) \
        - sum(held[a] for a in held if a not in stranded
              and a in resident and sched._stage[a] == -1)
    return chunks, events


def _check_join_invariants(sched, chunks, events, truth, prefix,
                           stages, frames, budget, resident=None):
    resident = resident or {}
    n, k = len(prefix), len(stages)
    covered = {(a, j): [] for a in range(n) for j in range(k)}
    carried_count = [0] * n
    for groups in chunks:
        tokens = 0
        seen = set()
        for a, j, start, end, carried in groups:
            assert a not in seen, "one anchor twice in a chunk"
            seen.add(a)
            assert end > start
            tokens += (prefix[a] if carried else 0) \
                + (frames[j] if start == 0 else 0) \
                + sum(stages[j][start:end])
            covered[(a, j)].extend(range(start, end))
            carried_count[a] += carried
        assert tokens <= budget, "over-budget chunk"
    # which stages each anchor should reach, from the planted truth
    for a in range(n):
        reach = 0
        for j in range(k):
            if not stages[j]:
                break
            reach = j + 1
            if not any(truth[a][j]):
                break
        for j in range(k):
            if j < reach:
                assert covered[(a, j)] == list(range(len(stages[j]))), \
                    f"anchor {a} stage {j} partners not streamed once"
                assert sched.answers[j][a] == truth[a][j]
            else:
                assert covered[(a, j)] == []
                assert a not in sched.answers[j]
        # the prefix is computed at most once, never when resident
        assert carried_count[a] == (0 if a in resident else 1)
        finished = reach == k
        dropped = reach < k and stages[reach - 1] and \
            not any(truth[a][reach - 1]) if reach else False
        kinds = [kind for kind, anchor in events if anchor == a]
        if finished:
            assert kinds == ["finished"]
        elif dropped:
            assert kinds == ["dropped"]
        else:
            assert kinds == []


def _random_join(rng, n_anchors=None):
    k = rng.randrange(1, 4)
    page_tokens = 16
    budget = rng.randrange(2_000, 20_000)
    frames = [rng.choice((0, 0, rng.randrange(1, 40))) for _ in range(k)]
    stages = []
    for j in range(k):
        count = rng.randrange(1, 80)
        stages.append([rng.randrange(10, max(11, budget // 4))
                       for _ in range(count)])
    top = budget - max(frames) - max(max(s) for s in stages if s)
    n = n_anchors or rng.randrange(1, 40)
    prefix = [rng.randrange(20, max(21, top)) for _ in range(n)]
    truth = [[[1 if rng.random() < 0.5 else 0 for _ in stages[j]]
              for j in range(k)] for _ in range(n)]
    return prefix, stages, frames, budget, page_tokens, truth


def test_join_admission_random_shapes():
    rng = random.Random(17)
    for _ in range(60):
        prefix, stages, frames, budget, page_tokens, truth = \
            _random_join(rng)
        need = [pages_for(p + max(frames), page_tokens) for p in prefix]
        arena_pages = max(max(need), rng.randrange(4, 400))
        # resident anchors hold some of their pages; the rest (frame
        # room) must fit the arena together, or the run cannot start
        resident = {}
        for a in range(len(prefix)):
            if rng.random() < 0.3:
                resident[a] = rng.randrange(1, need[a] + 1)
        while sum(need[a] for a in resident) > arena_pages:
            resident.popitem()
        sched = JoinAdmission(prefix, stages, budget, arena_pages,
                              page_tokens, frame_tokens=frames,
                              resident=resident)
        chunks, events = _drive_join(sched, truth, arena_pages,
                                     resident, deliver_lag=1, rng=rng)
        _check_join_invariants(sched, chunks, events, truth, prefix,
                               stages, frames, budget, resident)


def test_join_admission_cut_stream_continues_first():
    # anchor 0's stream is cut; its continuation leads the next chunk
    # and packs no prefix, ahead of fresh anchor 1
    sched = JoinAdmission([100, 50], [[400, 50]], 520,
                          arena_pages=100, page_tokens=16)
    assert sched.next_chunk(100) == [(0, 0, 0, 1, True)]
    assert sched.next_chunk(100) == [(0, 0, 1, 2, False),
                                     (1, 0, 0, 1, True)]
    assert sched.next_chunk(100) == [(1, 0, 1, 2, False)]


def test_join_admission_mixes_stages_in_one_chunk():
    # anchor 0 answers TRUE at stage 0 and its stage-1 partners lead
    # the next chunk, followed by fresh anchor 1's stage 0
    sched = JoinAdmission([100, 100], [[50], [30, 30]], 250,
                          arena_pages=100, page_tokens=16)
    assert sched.next_chunk(100) == [(0, 0, 0, 1, True)]
    assert sched.report(0, 0, 0, 1, [1]) == []
    assert sched.next_chunk(100) == [(0, 1, 0, 2, False),
                                     (1, 0, 0, 1, True)]
    assert sched.report(0, 1, 0, 2, [0, 1]) == [("finished", 0)]
    assert sched.report(1, 0, 0, 1, [0]) == [("dropped", 1)]
    assert sched.done()
    assert sched.answers == [{0: [1], 1: [0]}, {0: [0, 1]}]


def test_join_admission_advances_before_the_stream_is_answered():
    # the whole stage-0 stream is launched over two chunks; the first
    # answers TRUE, so stage 1 starts while the second is in flight
    sched = JoinAdmission([100], [[400, 400], [50]], 600,
                          arena_pages=100, page_tokens=16)
    assert sched.next_chunk(100) == [(0, 0, 0, 1, True)]
    assert sched.next_chunk(100) == [(0, 0, 1, 2, False)]
    assert sched.report(0, 0, 0, 1, [1]) == []
    assert sched.next_chunk(100) == [(0, 1, 0, 1, False)]
    assert sched.report(0, 0, 1, 2, [0]) == []
    assert sched.report(0, 1, 0, 1, [1]) == [("finished", 0)]
    assert sched.answers == [{0: [1, 0]}, {0: [1]}]


def test_join_admission_waits_for_a_true_before_advancing():
    # a cut stream whose first chunk answered FALSE does not advance
    # until a later chunk answers TRUE; all FALSE drops the anchor
    sched = JoinAdmission([100], [[400, 400], [50]], 600,
                          arena_pages=100, page_tokens=16)
    sched.next_chunk(100)
    sched.next_chunk(100)
    assert sched.report(0, 0, 0, 1, [0]) == []
    assert sched.next_chunk(100) == []
    assert sched.report(0, 0, 1, 2, [0]) == [("dropped", 0)]
    assert sched.done()


def test_join_admission_pages_block_in_order():
    # the arena holds one anchor; the second waits for pages even
    # though it would fit the chunk, and admits once the first drops
    sched = JoinAdmission([160, 160], [[10]], 400,
                          arena_pages=10, page_tokens=16)
    assert sched.next_chunk(10) == [(0, 0, 0, 1, True)]
    assert sched.blocked_pages == 10
    assert sched.next_chunk(0) == []
    assert sched.blocked_pages == 10
    assert sched.report(0, 0, 0, 1, [0]) == [("finished", 0)]
    assert sched.next_chunk(10) == [(1, 0, 0, 1, True)]


def test_join_admission_resident_anchors_pass_a_page_block():
    # anchor 1 is resident (no prefix, no pages) and packs even while
    # fresh anchor 0 waits for pages; resident anchors also go first
    sched = JoinAdmission([160, 160], [[10]], 400,
                          arena_pages=10, page_tokens=16,
                          resident={1: 10})
    assert sched.next_chunk(0) == [(1, 0, 0, 1, False)]
    assert sched.blocked_pages == 10
    assert sched.report(1, 0, 0, 1, [1]) == [("finished", 1)]
    assert sched.next_chunk(10) == [(0, 0, 0, 1, True)]


def test_join_admission_resident_growth_costs_pages():
    # a resident anchor short of frame room needs the difference
    sched = JoinAdmission([160], [[10]], 400, arena_pages=20,
                          page_tokens=16, frame_tokens=[20],
                          resident={0: 10})
    assert sched.next_chunk(1) == []
    assert sched.blocked_pages == 1
    assert sched.next_chunk(2) == [(0, 0, 0, 1, False)]


def test_join_admission_frame_counts_against_the_first_partner():
    # 100 prefix + 30 frame + 100 partner = 230: the anchor's first
    # partner alone fills the chunk; later partners fit two at a time
    sched = JoinAdmission([100], [[100, 100, 100]], 230,
                          arena_pages=100, page_tokens=16,
                          frame_tokens=[30])
    assert sched.next_chunk(100) == [(0, 0, 0, 1, True)]
    assert sched.next_chunk(100) == [(0, 0, 1, 3, False)]


def test_join_admission_empty_later_stage_strands_survivors():
    # stage 1 has no partners: a stage-0 survivor stays resident with
    # no event, and the run still finishes
    sched = JoinAdmission([100], [[10], []], 400,
                          arena_pages=100, page_tokens=16)
    sched.next_chunk(100)
    assert sched.report(0, 0, 0, 1, [1]) == []
    assert sched.done()
    assert sched.answers == [{0: [1]}, {}]


def test_join_admission_refuses_impossible_shapes():
    with pytest.raises(ValueError, match="first stage"):
        JoinAdmission([100], [[]], 400, 100, 16)
    with pytest.raises(ValueError, match="suffixes are atomic"):
        JoinAdmission([100], [[1050]], 1000, 100, 16)
    with pytest.raises(ValueError, match="no room for a partner"):
        JoinAdmission([950], [[100]], 1000, 100, 16)
    with pytest.raises(ValueError, match="anchor 0 needs 3 KV pages"):
        JoinAdmission([33], [[10]], 1000, arena_pages=2, page_tokens=16)
    # a resident anchor is never refused for pages: it already fits
    JoinAdmission([33], [[10]], 1000, arena_pages=2, page_tokens=16,
                  resident={0: 3})


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
    """Drive the scheduler to completion, delivering answers with the given lag.

    Returns:
        Per-chunk group lists.
    """
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


def test_admission_keep_credits_the_rewind_tail_pages():
    # a kept document holds pages_for(doc + kept_extra) while in
    # flight but is rewound to its own tokens: pages_for(70+30)=7
    # charged, pages_for(70)=5 kept, 2 back to admission
    sched = FilterAdmission([70, 60], [10], 200,
                            arena_pages=7, page_tokens=16,
                            kept_extra_tokens=30)
    assert sched.next_chunk() == [(0, 0, True)]
    assert sched.report(0, 0, True, release=False) == ()
    assert sched.free_pages == 2
    assert 0 not in sched.resident
    # the kept document's remaining 5 pages come back through
    # add_free_pages when the retained pool evicts it
    sched.add_free_pages(5)
    assert sched.next_chunk() == [(1, 0, True)]


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
    """Check that a limit admits fewer documents and chunks than an unlimited run."""
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
