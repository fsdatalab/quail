"""Tests for JoinAdmission, FilterAdmission, and gate/assemble against brute force."""

import random

import numpy as np
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


def _drive_join(sched, truth, arena_pages, resident=None,
                deliver_lag=1, rng=None):
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
    for a in range(n):
        ran = 0
        expected = ["finished"]
        for j in range(k):
            ran = j + 1 if indices(a, j) else j
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
        assert carried_count[a] == (0 if a in resident or not ran else 1)
        kinds = [kind for kind, anchor in events if anchor == a]
        assert kinds == expected, f"anchor {a}: {kinds} != {expected}"


def _random_partner_lists(rng, stages):
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


def _random_join(rng):
    k = rng.randrange(1, 4)
    budget = rng.randrange(2_000, 20_000)
    frames = [rng.choice((0, 0, rng.randrange(1, 40))) for _ in range(k)]
    stages = []
    for j in range(k):
        count = rng.randrange(1, 80)
        stages.append([rng.randrange(10, max(11, budget // 4))
                       for _ in range(count)])
    top = budget - max(frames) - max(max(s) for s in stages if s)
    n = rng.randrange(1, 40)
    prefix = [rng.randrange(20, max(21, top)) for _ in range(n)]
    lists = [_random_partner_lists(rng, stages) if rng.random() < 0.5
             else [None] * k for _ in range(n)]
    truth = []
    for row_lists in lists:
        truth.append([[int(rng.random() < 0.5) for _ in (lens if lst is None else lst)]
                      for lens, lst in zip(stages, row_lists)])
    return prefix, stages, frames, budget, truth, lists


def test_admission_invariants_over_random_shapes():
    rng = random.Random(17)
    for _ in range(80):
        prefix, stages, frames, budget, truth, lists = _random_join(rng)
        need = [pages_for(p + max(frames), 16) for p in prefix]
        arena_pages = max(max(need), rng.randrange(4, 400))
        resident = {a: rng.randrange(1, need[a] + 1)
                    for a in range(len(prefix)) if rng.random() < 0.3}
        while sum(need[a] for a in resident) > arena_pages:
            resident.popitem()
        sched = JoinAdmission(prefix, stages, budget, arena_pages,
                              16, frame_tokens=frames,
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
    for trial in range(30):
        n_docs = rng.randrange(5, 60)
        n_stages = 1 if trial % 3 == 0 else rng.randrange(1, 6)
        doc_tokens = [rng.randrange(20, 900) for _ in range(n_docs)]
        stage_tokens = [rng.randrange(10, 60) for _ in range(n_stages)]
        budget = max(doc_tokens) + max(stage_tokens) + rng.randrange(0, 2000)
        arena_pages = max(pages_for(max(doc_tokens), 16), rng.randrange(4, 200))
        if trial % 3 == 0:
            arena_pages = None
        truth = [[1 if rng.random() < 0.7 else 0
                  for _ in range(n_stages)] for _ in range(n_docs)]
        sched = FilterAdmission(doc_tokens, stage_tokens, budget, arena_pages, 16)
        chunks = _drive(sched, truth, deliver_lag=rng.choice((0, 1)))
        _check_invariants(sched, chunks, truth, doc_tokens, stage_tokens,
                          budget, arena_pages)


def test_join_order_page_capacity_and_resident_anchors():
    # a cut stream's continuation leads the next chunk, ahead of a fresh anchor
    sched = JoinAdmission([100, 50], [[400, 50]], 520, arena_pages=100, page_tokens=16)
    assert sched.next_chunk(100) == [(0, 0, 0, 1, True)]
    assert sched.next_chunk(100) == [(0, 0, 1, 2, False), (1, 0, 0, 1, True)]

    # a resident anchor packs first, even while a fresh one waits for pages
    sched = JoinAdmission([160, 160], [[10]], 400, arena_pages=10, page_tokens=16,
                          resident={1: 10})
    assert sched.next_chunk(0) == [(1, 0, 0, 1, False)]
    assert sched.blocked_pages == 10
    assert sched.report(1, 0, 0, 1, [1]) == [("finished", 1)]
    assert sched.next_chunk(10) == [(0, 0, 0, 1, True)]

    with pytest.raises(ValueError, match="at least one stage"):
        JoinAdmission([100], [], 400, 100, 16)
    with pytest.raises(ValueError, match="out of range"):
        JoinAdmission([100], [[10]], 400, 100, 16, anchor_partners={0: [[1]]})
    with pytest.raises(ValueError, match="partner lists for 2 stages"):
        JoinAdmission([100], [[10]], 400, 100, 16,
                      anchor_partners={0: [[0], None]})
    with pytest.raises(ValueError, match="suffixes are atomic"):
        JoinAdmission([100], [[1050]], 1000, 100, 16)
    with pytest.raises(ValueError, match="no room for a partner"):
        JoinAdmission([950], [[100]], 1000, 100, 16)
    with pytest.raises(ValueError, match="anchor 0 needs 3 KV pages"):
        JoinAdmission([33], [[10]], 1000, arena_pages=2, page_tokens=16)
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


def _drive(sched, truth, deliver_lag=1):
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
                      budget, arena_pages):
    fresh = []
    for groups in chunks:
        tokens = [stage_tokens[stage] + (doc_tokens[doc] if is_fresh else 0)
                  for doc, stage, is_fresh in groups]
        assert sum(tokens) <= budget, "over-budget chunk"
        fresh += [doc for doc, stage, is_fresh in groups if is_fresh]
        assert all(stage == 0 for _, stage, is_fresh in groups if is_fresh)
    assert sorted(fresh) == list(range(len(doc_tokens)))
    for d, row in enumerate(truth):
        expect = row[:row.index(0) + 1] if 0 in row else row
        assert sched.answers[d] == expect, f"doc {d} gating wrong"
    assert sched.free_pages == arena_pages
    assert sched.survivors() == [d for d, row in enumerate(truth) if all(row)]


def test_filter_admission_order_and_rewind():
    # one resident survivor's next suffix packs before fresh docs
    sched = FilterAdmission([100, 100], [10, 10], 200,
                            arena_pages=100, page_tokens=16)
    assert sched.next_chunk() == [(0, 0, True)]     # no room for doc 1 (110+110)
    sched.report(0, 0, True)
    second = sched.next_chunk()
    assert second[0] == (0, 1, False)
    assert (1, 0, True) in second

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
    sched.add_free_pages(5)
    assert sched.next_chunk() == [(1, 0, True)]

    with pytest.raises(ValueError):
        FilterAdmission([1000], [10], 500, 100, 16)     # over chunk
    with pytest.raises(ValueError):
        FilterAdmission([1000], [10], 2000, 2, 16)      # over arena
    # without a page bin, a document too big for any arena is admitted
    sched = FilterAdmission([1000], [10], 2000, None, 16)
    assert sched.next_chunk() == [(0, 0, True)]
    assert sched.free_pages is None and not sched.resident


def test_filter_limit_drains_ready_documents():
    # the limit is met while docs 1 and 2 wait in ready; draining frees them
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


@pytest.mark.parametrize("arena_pages", [None, 400])
def test_filter_limit_stops_admission_early(arena_pages):
    sched = FilterAdmission([100] * 40, [10], 230, arena_pages, 16, limit=3)
    chunks = _drive(sched, [[1]] * 40, deliver_lag=0)
    assert len(sched.survivors()) >= 3
    assert sum(len(groups) for groups in chunks) < 40


def _check_stream(out, filter_truth, pages):
    stream = out["stream"]
    assert stream.done
    assert stream.answers == expected_filter_rows(filter_truth)
    survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
    assert sorted(key[1] for key in out["anchor_keys"]) == survivors
    assert not out["arena"].accounting.owned
    assert out["arena"].accounting.free_pages == pages
    return survivors


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


def test_join_admission_admits_incrementally_and_prices_room():
    sched = JoinAdmission([], [[10, 10, 10]], 100, 50, 16,
                          frame_tokens=[5])
    assert sched.done()
    assert sched.buildable_tokens() == 0
    assert sched.admit(40, resident_pages=3) == 0
    assert sched.buildable_tokens() == 5 + 30
    assert sched.admit(80) == 1
    assert sched.buildable_tokens() == 100
    groups = sched.next_chunk(free_pages=50)
    assert groups[0] == (0, 0, 0, 3, False)
    assert sched.report(0, 0, 0, 3, [0, 1, 0]) == [("finished", 0)]
    assert sched.buildable_tokens() == 100
    with pytest.raises(ValueError):
        sched.admit(5000)


def test_unified_join_accounts_for_suffix_pages():
    sched = JoinAdmission([31, 31], [[1, 1, 1, 1]], 256, 8, 16,
                          temporary_suffix_pages=True)
    groups = sched.next_chunk(8)
    prefix_pages = sum(2 for *_, carried in groups if carried)
    suffix_pages = sum(end - start for _, _, start, end, _ in groups)
    assert prefix_pages + suffix_pages <= 8
    assert suffix_pages < 8


def test_numeric_join_answers_preserve_values_across_chunks():
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
