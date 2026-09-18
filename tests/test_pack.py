"""Tests for JoinAdmission, FilterAdmission, and gate/assemble against brute force."""

import random

import pytest
from fakes import expected_filter_rows, run_streamed

from quail.backends.quail.executor.pack import (
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
        free += held.pop(a, 0)

    while not sched.done():
        for kind, anchor in sched.take_settled():
            events.append((kind, anchor))
            settle(anchor)
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
    assert free == arena_pages, "pages leaked or freed twice"
    return chunks, events


def _check_join_invariants(sched, chunks, events, truth, prefix,
                           stages, frames, budget, resident=None,
                           lists=None):
    """Check chunks, coverage, answers, and events against the truth.

    lists[a][j] is the anchor's partner index list at stage j, or
    None for the whole stage list; truth[a][j] runs over that list.
    """
    resident = resident or {}
    n, k = len(prefix), len(stages)

    def indices(a, j):
        lst = None if lists is None else lists[a][j]
        return list(range(len(stages[j]))) if lst is None else list(lst)

    covered = {(a, j): [] for a in range(n) for j in range(k)}
    carried_count = [0] * n
    for groups in chunks:
        tokens = 0
        seen = set()
        for a, j, start, end, carried in groups:
            assert a not in seen, "one anchor twice in a chunk"
            seen.add(a)
            assert end > start
            partners = sched.partner_indices(a, j, start, end)
            assert partners == indices(a, j)[start:end]
            tokens += (prefix[a] if carried else 0) \
                + (frames[j] if start == 0 else 0) \
                + sum(stages[j][i] for i in partners)
            covered[(a, j)].extend(range(start, end))
            carried_count[a] += carried
        assert tokens <= budget, "over-budget chunk"
    # which stages each anchor runs, from the planted truth: a stage
    # with no partner or no TRUE ends the anchor there
    for a in range(n):
        ran = 0
        expected = ["finished"]
        for j in range(k):
            if not indices(a, j):
                expected = ["finished"] if j == k - 1 else ["dropped"]
                break
            ran = j + 1
            if not any(truth[a][j]):
                expected = ["finished"] if j == k - 1 else ["dropped"]
                break
        for j in range(k):
            if j < ran:
                assert covered[(a, j)] == list(range(len(indices(a, j)))), \
                    f"anchor {a} stage {j} partners not streamed once"
                assert sched.answers[j][a] == truth[a][j]
            else:
                assert covered[(a, j)] == []
                assert a not in sched.answers[j]
        # the prefix is computed at most once, never when resident,
        # and never for an anchor that ran no chunk
        assert carried_count[a] == (0 if a in resident or not ran else 1)
        kinds = [kind for kind, anchor in events if anchor == a]
        assert kinds == expected, f"anchor {a}: {kinds} != {expected}"


def _random_partner_lists(rng, stages):
    """Per stage, a random index list into the stage, or None."""
    lists = []
    for lens in stages:
        roll = rng.random()
        if roll < 0.4:
            lists.append(None)
        elif roll < 0.5:
            lists.append([])
        else:
            count = rng.randrange(1, len(lens) + 1)
            lists.append(sorted(rng.sample(range(len(lens)), count)))
    return lists


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
    # some anchors stream their own partner subsets (a join over pairs)
    lists = [_random_partner_lists(rng, stages) if rng.random() < 0.5
             else [None] * k for _ in range(n)]
    truth = []
    for a in range(n):
        rows = []
        for j in range(k):
            count = len(stages[j]) if lists[a][j] is None else len(lists[a][j])
            rows.append([1 if rng.random() < 0.5 else 0
                         for _ in range(count)])
        truth.append(rows)
    return prefix, stages, frames, budget, page_tokens, truth, lists


def test_admission_invariants_over_random_shapes():
    rng = random.Random(17)
    for _ in range(80):
        prefix, stages, frames, budget, page_tokens, truth, lists = \
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
                              resident=resident,
                              anchor_partners={
                                  a: lists[a] for a in range(len(prefix))
                                  if any(lst is not None for lst in lists[a])
                              })
        chunks, events = _drive_join(sched, truth, arena_pages,
                                     resident, deliver_lag=1, rng=rng)
        _check_join_invariants(sched, chunks, events, truth, prefix,
                               stages, frames, budget, resident, lists)

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


def test_join_token_admission_and_stage_progress():
    # anchor 0's stream is cut; its continuation leads the next chunk
    # and packs no prefix, ahead of fresh anchor 1
    sched = JoinAdmission([100, 50], [[400, 50]], 520,
                          arena_pages=100, page_tokens=16)
    assert sched.next_chunk(100) == [(0, 0, 0, 1, True)]
    assert sched.next_chunk(100) == [(0, 0, 1, 2, False),
                                     (1, 0, 0, 1, True)]
    assert sched.next_chunk(100) == [(1, 0, 1, 2, False)]

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

    # 100 prefix + 30 frame + 100 partner = 230: the anchor's first
    # partner alone fills the chunk; later partners fit two at a time
    sched = JoinAdmission([100], [[100, 100, 100]], 230,
                          arena_pages=100, page_tokens=16,
                          frame_tokens=[30])
    assert sched.next_chunk(100) == [(0, 0, 0, 1, True)]
    assert sched.next_chunk(100) == [(0, 0, 1, 3, False)]

    # stage 1 has no partners: a stage-0 survivor finishes with an
    # empty last row, so the caller settles its KV
    sched = JoinAdmission([100], [[10], []], 400,
                          arena_pages=100, page_tokens=16)
    sched.next_chunk(100)
    assert sched.report(0, 0, 0, 1, [1]) == [("finished", 0)]
    assert sched.done()
    assert sched.answers == [{0: [1]}, {}]

    # a join over pairs: anchor 0 streams partners 2 and 0 of stage 0
    # and everything at stage 1; anchor 1 has no stage-0 partner and
    # is dropped before any chunk; anchor 2 has none at stage 1
    sched = JoinAdmission(
        [100, 100, 100], [[10, 20, 30], [40]], 400,
        arena_pages=100, page_tokens=16,
        anchor_partners={0: [[2, 0], None], 1: [[], None], 2: [None, []]})
    assert sched.take_settled() == [("dropped", 1)]
    assert not sched.done()
    assert sched.next_chunk(100) == [(0, 0, 0, 2, True), (2, 0, 0, 3, True)]
    assert sched.partner_indices(0, 0, 0, 2) == [2, 0]
    assert sched.partner_indices(2, 0, 1, 3) == [1, 2]
    assert sched.report(0, 0, 0, 2, [0, 1]) == []
    assert sched.report(2, 0, 0, 3, [1, 0, 0]) == [("finished", 2)]
    assert sched.next_chunk(100) == [(0, 1, 0, 1, False)]
    assert sched.report(0, 1, 0, 1, [1]) == [("finished", 0)]
    assert sched.done()
    assert sched.answers == [{0: [0, 1], 2: [1, 0, 0]}, {0: [1]}]
    # a stage 0 with no partner at all settles every anchor at once
    sched = JoinAdmission([100, 50], [[]], 400, arena_pages=100,
                          page_tokens=16)
    assert sched.take_settled() == [("finished", 0), ("finished", 1)]
    assert sched.done()


def test_join_page_capacity_and_resident_anchors():
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

    # anchor 1 is resident (no prefix, no pages) and packs even while
    # fresh anchor 0 waits for pages; resident anchors also go first
    sched = JoinAdmission([160, 160], [[10]], 400,
                          arena_pages=10, page_tokens=16,
                          resident={1: 10})
    assert sched.next_chunk(0) == [(1, 0, 0, 1, False)]
    assert sched.blocked_pages == 10
    assert sched.report(1, 0, 0, 1, [1]) == [("finished", 1)]
    assert sched.next_chunk(10) == [(0, 0, 0, 1, True)]

    # a resident anchor short of frame room needs the difference
    sched = JoinAdmission([160], [[10]], 400, arena_pages=20,
                          page_tokens=16, frame_tokens=[20],
                          resident={0: 10})
    assert sched.next_chunk(1) == []
    assert sched.blocked_pages == 1
    assert sched.next_chunk(2) == [(0, 0, 0, 1, False)]

    with pytest.raises(ValueError, match="at least one stage"):
        JoinAdmission([100], [], 400, 100, 16)
    with pytest.raises(ValueError, match="out of range"):
        JoinAdmission([100], [[10]], 400, 100, 16,
                      anchor_partners={0: [[1]]})
    with pytest.raises(ValueError, match="partner lists for 2 stages"):
        JoinAdmission([100], [[10]], 400, 100, 16,
                      anchor_partners={0: [[0], None]})
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


def test_filter_admission_capacity_and_rewind():
    # one resident survivor's next suffix packs before fresh docs
    sched = FilterAdmission([100, 100], [10, 10], 200,
                            arena_pages=100, page_tokens=16)
    first = sched.next_chunk()
    assert first == [(0, 0, True)]     # doc 1 skipped: no room (110+110)
    sched.report(0, 0, True)
    second = sched.next_chunk()
    assert second[0] == (0, 1, False)  # survivor suffix leads
    assert (1, 0, True) in second

    # arena holds one big doc; the second waits for pages even though
    # it would fit the chunk
    sched = FilterAdmission([160, 160], [10], 400,
                            arena_pages=10, page_tokens=16)
    first = sched.next_chunk()
    assert first == [(0, 0, True)]
    assert sched.next_chunk() == []    # pages blocked, answer in flight
    sched.report(0, 0, False)          # FALSE frees the pages
    assert sched.next_chunk() == [(1, 0, True)]

    sched = FilterAdmission([80], [10], 200,
                            arena_pages=10, page_tokens=16,
                            available_pages=2)

    assert sched.next_chunk() == []
    assert sched.blocked_pages == 3
    sched.add_free_pages(3)
    assert sched.next_chunk() == [(0, 0, True)]

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

    with pytest.raises(ValueError):
        FilterAdmission([1000], [10], 500, 100, 16)     # over chunk
    with pytest.raises(ValueError):
        FilterAdmission([1000], [10], 2000, 2, 16)      # over arena


def test_filter_limits_drain_work_and_reduce_admission():
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

def test_filter_admission_without_stored_kv():
    # a document too big for any arena is admitted when there is no
    # page bin; a document too big for the chunk is still refused
    sched = FilterAdmission([1000], [10], 2000, None, 16)
    assert sched.next_chunk() == [(0, 0, True)]
    with pytest.raises(ValueError):
        FilterAdmission([1000], [10], 500, None, 16)

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


# ------------------------------------------------ the chain streamed into the join


def _check_stream(out, filter_truth, pages):
    stream = out["stream"]
    assert stream.done
    assert stream.answers == expected_filter_rows(filter_truth)
    survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
    assert sorted(key[1] for key in out["anchor_keys"]) == survivors
    assert not out["arena"].accounting.owned
    assert out["arena"].accounting.free_pages == pages
    return survivors


def test_streamed_filter_feeds_the_join_and_frees_everything(monkeypatch):
    rng = random.Random(7)
    n_docs, n_partners = 60, 5
    doc_lengths = [rng.randrange(20, 90) for _ in range(n_docs)]
    filter_truth = [[1 if rng.random() < 0.8 else 0, 1 if rng.random() < 0.7 else 0]
                    for _ in range(n_docs)]
    partner_lengths = [rng.randrange(5, 20) for _ in range(n_partners)]
    join_truth = {("r", d): [1 if rng.random() < 0.5 else 0
                             for _ in range(n_partners)]
                  for d in range(n_docs)}
    # the arena holds about a dozen documents: the filter blocks on
    # pages before the corpus is through, and the join has to drain
    out = run_streamed(
        monkeypatch, doc_lengths=doc_lengths, filter_truth=filter_truth,
        partner_lengths=partner_lengths, join_truth=join_truth,
        budget=400, pages=80)
    survivors = _check_stream(out, filter_truth, 80)
    # every survivor's row answered from the planted truth, in
    # admission order
    rows = out["join_answers"][0]
    assert sorted(rows) == list(range(len(survivors)))
    for local, key in enumerate(out["anchor_keys"]):
        assert rows[local] == join_truth[key]
        assert out["settled"][key] == join_truth[key]
    # join chunks ran before the chain finished, and the chain reported
    # a page shortfall at least once
    kinds = [kind for kind, _ in out["model"].launched]
    assert "join" in kinds[:-1] and kinds[-1] == "join"
    assert kinds.index("join") < len(kinds) - 1 - kinds[::-1].index("filter")
    assert any(out["blocked"])
    # streamed anchors pack the frame and partner suffixes only
    assert out["join_tokens"] == sum(
        3 + sum(partner_lengths) for _ in survivors)


def test_streamed_loop_random_shapes(monkeypatch):
    rng = random.Random(11)
    for _ in range(25):
        n_docs = rng.randrange(1, 40)
        n_partners = rng.randrange(0, 6)
        stages = rng.randrange(1, 4)
        doc_lengths = [rng.randrange(8, 120) for _ in range(n_docs)]
        filter_truth = [[1 if rng.random() < 0.7 else 0
                         for _ in range(stages)] for _ in range(n_docs)]
        partner_lengths = [rng.randrange(4, 30) for _ in range(n_partners)]
        join_truth = {("r", d): [1 if rng.random() < 0.5 else 0
                                 for _ in range(n_partners)]
                      for d in range(n_docs)}
        frame_tokens = rng.randrange(0, 20)
        budget = max(doc_lengths) + 1 + rng.randrange(0, 400)
        budget = max(budget, frame_tokens + max(partner_lengths, default=0))
        pages = max(-(-(max(doc_lengths) + max(1, frame_tokens)) // 16),
                    rng.randrange(6, 40))
        out = run_streamed(
            monkeypatch, doc_lengths=doc_lengths, filter_truth=filter_truth,
            partner_lengths=partner_lengths, join_truth=join_truth,
            budget=budget, pages=pages, frame_tokens=frame_tokens,
            stages=stages)
        survivors = _check_stream(out, filter_truth, pages)
        if n_partners:
            rows = out["join_answers"][0]
            assert sorted(rows) == list(range(len(survivors)))
            for local, key in enumerate(out["anchor_keys"]):
                assert rows[local] == join_truth[key]
        else:
            assert out["join_answers"] == [{}]
            assert set(out["settled"]) == set(out["anchor_keys"])


def test_streamed_join_over_pairs_packs_only_allowed_partners(monkeypatch):
    rng = random.Random(11)
    n_docs, n_partners = 30, 6
    doc_lengths = [rng.randrange(8, 30) for _ in range(n_docs)]
    filter_truth = [[1 if rng.random() < 0.8 else 0,
                     1 if rng.random() < 0.7 else 0] for _ in range(n_docs)]
    partner_lengths = [rng.randrange(5, 20) for _ in range(n_partners)]
    join_truth = {("r", d): [1 if rng.random() < 0.5 else 0
                             for _ in range(n_partners)] for d in range(n_docs)}
    # document d pairs with the partners i where d + i is a multiple
    # of 4; every seventh document pairs with none
    allowed = {d: [i for i in range(n_partners)
                   if (d + i) % 4 == 0 and d % 7 != 3]
               for d in range(n_docs)}
    run = run_streamed(
        monkeypatch, doc_lengths=doc_lengths, filter_truth=filter_truth,
        partner_lengths=partner_lengths, join_truth=join_truth,
        budget=400, pages=40,
        anchor_partners=lambda key: [allowed[key[1]]])
    survivors = _check_stream(run, filter_truth, 40)
    for local, key in enumerate(run["anchor_keys"]):
        expected = [join_truth[key][i] for i in allowed[key[1]]]
        assert run["settled"][key] == expected
        assert run["join_answers"][0].get(local, []) == expected
    # every streamed anchor packs the frame plus its own partners only;
    # an anchor with no partner packs nothing and is settled at once
    assert run["join_tokens"] == sum(
        3 + sum(partner_lengths[i] for i in allowed[d])
        for d in survivors if allowed[d])
    assert any(not allowed[d] for d in survivors)


def test_join_admission_admits_incrementally_and_prices_room():
    sched = JoinAdmission([], [[10, 10, 10]], 100, 50, 16,
                          frame_tokens=[5])
    assert sched.done()
    assert sched.buildable_tokens() == 0
    first = sched.admit(40, resident_pages=3)
    assert first == 0
    assert sched.buildable_tokens() == 5 + 30
    second = sched.admit(80)
    assert second == 1
    assert sched.buildable_tokens() == 100
    groups = sched.next_chunk(free_pages=50)
    assert groups[0] == (0, 0, 0, 3, False)
    assert sched.report(0, 0, 0, 3, [0, 1, 0]) == [("finished", 0)]
    # the resident anchor packed no prefix; the fresh one still waits
    # for chunk room and counts its prefix as buildable work
    assert sched.buildable_tokens() == 100
    with pytest.raises(ValueError):
        sched.admit(5000)


def test_unified_join_accounts_for_suffix_pages():
    sched = JoinAdmission(
        [31, 31], [[1, 1, 1, 1]], 256, 8, 16,
        temporary_suffix_pages=True,
    )
    groups = sched.next_chunk(8)
    prefix_pages = sum(2 for *_, carried in groups if carried)
    suffix_pages = sum(end - start for _, _, start, end, _ in groups)
    assert prefix_pages + suffix_pages <= 8
    assert sum(end - start for _, _, start, end, _ in groups) < 8


def test_unified_join_reserves_room_for_later_larger_suffix():
    sched = JoinAdmission(
        [16, 16, 16], [[1, 63]], 256, 6, 16,
        temporary_suffix_pages=True,
    )
    free = 6
    seen = 0
    for _ in range(20):
        if sched.done():
            break
        groups = sched.next_chunk(free)
        assert groups
        new = sum(1 for *_, carried in groups if carried)
        temporary = sum(
            pages_for(sched.stages[j][i], 16)
            for _, j, start, end, _ in groups for i in range(start, end)
        )
        assert new + temporary <= free
        free -= new
        for a, j, start, end, _ in groups:
            seen += end - start
            for kind, _ in sched.report(a, j, start, end, [0.5] * (end - start)):
                if kind == "finished":
                    free += 1
    assert sched.done()
    assert seen == 6
    assert free == 6



def test_numeric_join_answers_preserve_values_across_chunks():
    import numpy as np

    sched = JoinAdmission([16], [[16, 16, 16, 16]], 48, 20, 16,
                          answer_dtype=np.float32)
    expected = np.array([0.0, 0.125, 0.5, 1.0], dtype=np.float32)
    while not sched.done():
        groups = sched.next_chunk(19)
        assert groups
        for a, j, start, end, _ in groups:
            sched.report(a, j, start, end, expected[start:end])
    assert sched.answers[0][0].dtype == np.float32
    np.testing.assert_array_equal(sched.answers[0][0], expected)
