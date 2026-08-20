"""pack.py against brute force, on randomized shapes. The pack_stream
half is the exploration's test suite carried over; the FilterAdmission
half is the brute-force simulator the design calls for."""

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


def test_plan_groups_uniform_matches_ceil():
    groups = plan_groups(1040, [58] * 3718, 25305)
    k = (25305 - 1040) // 58
    assert len(groups) == -(-3718 // k)
    assert groups[0] == (0, k)
    assert groups[-1][1] == 3718


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
    """Run the scheduler to completion the way loop.py will: build a
    chunk, then deliver answers for chunks launched `deliver_lag` ago
    (answers land while later chunks run). Returns per-chunk group
    lists for the invariant checks."""
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
    # answers match the planted truth up to the first NO
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
    sched.report(0, 0, False)          # NO frees the pages
    assert sched.next_chunk() == [(1, 0, True)]


def test_admission_refuses_impossible_shapes():
    with pytest.raises(ValueError):
        FilterAdmission([1000], [10], 500, 100, 16)     # over chunk
    with pytest.raises(ValueError):
        FilterAdmission([1000], [10], 2000, 2, 16)      # over arena


def test_admission_limit_stops_early():
    # 10 docs, all pass one stage, limit=3. With deliver_lag=0 (answers
    # land immediately), the scheduler stops admitting after the first
    # chunk's survivors reach the limit.
    doc_tokens = [50] * 10
    stage_tokens = [10]
    truth = [[1] for _ in range(10)]
    sched = FilterAdmission(doc_tokens, stage_tokens, 200,
                            arena_pages=100, page_tokens=16, limit=3)
    chunks = _drive(sched, truth, deliver_lag=0)
    assert sched._survivor_count >= 3
    assert len(sched.survivors()) >= 3
    # fewer docs admitted than the full corpus
    assert len(sched.answers) < 10


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


def test_admission_limit_none_processes_all():
    doc_tokens = [50] * 5
    truth = [[1] for _ in range(5)]
    sched = FilterAdmission(doc_tokens, [10], 500,
                            arena_pages=100, page_tokens=16, limit=None)
    _drive(sched, truth)
    assert len(sched.survivors()) == 5
