"""Tests for JoinAdmission and filter runs over a prefix tree."""

import random

import pytest
from fakes import expected_filter_rows

from quail.backends.quail.executor.pack import (
    DROP,
    SKIP,
    AdmissionReport,
    JoinAdmission,
    Settlement,
    pages_for,
)
from quail.execution.tokens import PrefixTree


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
        for settlement in sched.take_settled():
            assert not settlement.survived
            events.append((settlement.kind, settlement.anchor))
            settle(settlement.anchor)
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
                report = sched.report(a, j, start, end, bits)
                for settlement in report.settlements:
                    events.append((settlement.kind, settlement.anchor))
                    settle(settlement.anchor)
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
            assert list(partners) == indices(a, j)[start:end]
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


def _check_stream(out, filter_truth, pages):
    assert out["filter_answers"] == expected_filter_rows(filter_truth)
    survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
    assert sorted(key[1] for key in out["anchor_keys"]) == survivors
    assert not out["arena"].accounting.owned
    assert out["arena"].accounting.free_pages == pages
    return survivors


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


def _tree_filter(monkeypatch, *, truth, budget, retain=(), limit=None,
                 pages=64, cap=None, order=(0, 1, 2), attention_mode=None,
                 docs=None, tree=None, stop_groups=None):
    """Run a one-question filter over three documents.

    By default, document 1 shares its first 32 tokens with document 0.
    """
    from types import SimpleNamespace

    from fakes import (
        DOC,
        QUESTION,
        FakeModel,
        cpu_arena,
        fake_pack,
        fake_pipeline,
        fake_torch,
    )

    from quail.backends.quail.executor import chunk as chunk_mod
    from quail.backends.quail.executor import loop

    # a test recording the packed specs keeps its own pack_chunk
    if not getattr(chunk_mod.pack_chunk, "recording", False):
        monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = FakeModel(truth, {})
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)
    arena = cpu_arena(pages)
    if cap is not None:
        arena.retention_cap_pages = cap
    if docs is None:
        docs = [[DOC] * 64, [DOC] * 32 + [DOC + 1] * 32, [DOC + 2] * 64]
    if tree is None:
        tree = PrefixTree(order=list(order), parent=[None, 0, None],
                          shared=[0, 32, 0])
    stats = {}
    got, _, tokens = loop.run_filter(
        fake_torch(), arena, pipeline, answers, docs, [[QUESTION]], budget,
        arena_writes=True, arena_keys=[("r", d) for d in range(len(docs))],
        prefix_tree=tree, retain_survivors=retain, limit=limit,
        attention_mode=attention_mode, stats=stats, stop_groups=stop_groups)
    return got, tokens, stats["borrowed_tokens"], arena


def test_filter_borrows_from_parents_and_stacks_siblings(monkeypatch):
    from fakes import DOC, fake_pack

    from quail.backends.quail.executor import chunk as chunk_mod

    packed = []

    def recording_pack(torch, arena, specs, **kw):
        packed.append([(s["key"], s.get("start", 0), s.get("read_key"))
                       for s in specs])
        return fake_pack(torch, arena, specs, **kw)

    recording_pack.recording = True
    monkeypatch.setattr(chunk_mod, "pack_chunk", recording_pack)

    # two siblings and another root packed in one chunk under tree
    # attention: the siblings follow their parent in one run, each
    # naming it as the read key, and the other root comes at its place
    # in tree order
    docs = [[DOC] * 64, [DOC] * 32 + [DOC + 1] * 8, [DOC + 2] * 64,
            [DOC] * 32 + [DOC + 3] * 8]
    tree = PrefixTree(order=[0, 1, 3, 2], parent=[None, 0, None, 0],
                      shared=[0, 32, 0, 32])
    got, tokens, borrowed, arena = _tree_filter(
        monkeypatch, truth={0: [1], 1: [1], 2: [1], 3: [1]}, budget=500,
        docs=docs, tree=tree, attention_mode="tree")
    assert got == {0: [1], 1: [1], 2: [1], 3: [1]}
    assert tokens == 64 + 8 + 8 + 64 + 4
    assert borrowed == 64
    assert packed == [[(("r", 0), 0, None), (("r", 1), 32, ("r", 0)),
                       (("r", 3), 32, ("r", 0)), (("r", 2), 0, None)]]
    assert arena.free_pages == 64 and not arena.accounting.owned

    # a child that is itself a parent packs after its own parent even
    # when its document index is smaller
    docs = [[DOC] * 48 + [DOC + 1] * 16, [DOC] * 48 + [DOC + 1] * 16 + [5] * 8,
            [DOC] * 48]
    tree = PrefixTree(order=[2, 0, 1], parent=[2, 0, None],
                      shared=[48, 64, 0])
    packed.clear()
    got, tokens, borrowed, arena = _tree_filter(
        monkeypatch, truth={0: [1], 1: [1], 2: [1]}, budget=500,
        docs=docs, tree=tree, attention_mode="tree")
    assert packed == [[(("r", 2), 0, None), (("r", 0), 48, ("r", 2)),
                       (("r", 1), 64, ("r", 0))]]
    assert tokens == 48 + 16 + 8 + 3
    assert arena.free_pages == 64 and not arena.accounting.owned
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)

    # answers are read after the next chunk launches, so a child right
    # behind its parent still borrows even when the parent fails
    got, tokens, borrowed, arena = _tree_filter(
        monkeypatch, truth={0: [0], 1: [1], 2: [1]}, budget=65)
    assert got == {0: [0], 1: [1], 2: [1]}
    assert tokens == 64 + 32 + 64 + 3
    assert borrowed == 32
    assert arena.free_pages == 64 and not arena.accounting.owned

    # with another document between them the parent has answered
    # before the child's turn: its pages are kept, the child borrows
    got, tokens, borrowed, arena = _tree_filter(
        monkeypatch, truth={0: [0], 1: [1], 2: [1]}, budget=65,
        order=(0, 2, 1))
    assert got == {0: [0], 1: [1], 2: [1]}
    assert tokens == 64 + 64 + 32 + 3
    assert borrowed == 32
    assert arena.free_pages == 64 and not arena.accounting.owned

    # a limit ends the run with the parent kept: its key is freed
    got, tokens, borrowed, arena = _tree_filter(
        monkeypatch, truth={0: [1], 1: [1], 2: [1]}, budget=65,
        order=(0, 2, 1), limit=1)
    assert arena.free_pages == 64 and not arena.accounting.owned

    # a retained parent under a cap of zero pages is evicted at once;
    # the child still finishes and the arena is clean
    got, tokens, borrowed, arena = _tree_filter(
        monkeypatch, truth={0: [1], 1: [1], 2: [1]}, budget=65,
        retain=True, cap=0)
    assert got == {0: [1], 1: [1], 2: [1]}
    assert arena.free_pages == 64 and arena.retained_keys() == []

    # a retained parent with room keeps its pages while the child
    # borrows them; freeing the parent leaves the child intact
    got, tokens, borrowed, arena = _tree_filter(
        monkeypatch, truth={0: [1], 1: [1], 2: [1]}, budget=500,
        retain=True, cap=64)
    assert sorted(arena.retained_keys()) == [("r", 0), ("r", 1), ("r", 2)]
    assert arena.accounting.table_pages(("r", 1))[:2] == \
        arena.accounting.table_pages(("r", 0))[:2]
    arena.free_key(("r", 0))
    assert len(arena.accounting.table_pages(("r", 1))) == 4
    for key in arena.retained_keys():
        arena.free_key(key)
    assert arena.free_pages == 64


def test_join_admission_limit_ends_the_run_and_drains_the_queue():
    # three anchors of one suffix each, a limit of one survivor: the
    # first finishes, and the others already launched drain
    sched = JoinAdmission([50, 50, 50], [[10]], 250, arena_pages=100,
                          page_tokens=16, limit=1)
    groups = sched.next_chunk(100)
    assert [a for a, *_ in groups] == [0, 1, 2]
    sched.report(0, 0, 0, 1, [1])
    assert sched.limit_reached() and sched.next_chunk(100) == []
    assert not sched.done()
    sched.report(1, 0, 0, 1, [1])
    sched.report(2, 0, 0, 1, [0])
    assert sched.done() and sched.survivors == 2
    sched = JoinAdmission([50, 50, 50], [[10], [200]], 250, arena_pages=100,
                          page_tokens=16, limit=1)
    for a, *_ in sched.next_chunk(100):
        sched.report(a, 0, 0, 1, [1])
    assert sched.next_chunk(100) == [(0, 1, 0, 1, False)]
    sched.report(0, 1, 0, 1, [1])
    assert sched.done()
    assert sched.drain() == [1, 2]


def test_stop_groups_admit_one_anchor_per_group_and_skip_the_rest():
    # groups {0, 1, 2} and {3, 4}: one anchor of each runs at a time
    sched = JoinAdmission([50] * 5, [[10]], 250, arena_pages=100,
                          page_tokens=16, stop_groups=[0, 0, 0, 1, 1])
    assert [a for a, *_ in sched.next_chunk(100)] == [0, 3]
    assert sched.next_chunk(100) == []
    sched.report(3, 0, 0, 1, [0])
    assert [a for a, *_ in sched.next_chunk(100)] == [4]
    sched.report(0, 0, 0, 1, [1])
    # the group's queued anchors are skipped without an answer
    assert sched.take_skipped() == [1, 2] and sched.take_skipped() == []
    assert sched.next_chunk(100) == [] and not sched.done()
    sched.report(4, 0, 0, 1, [1])
    assert sched.done() and sched.survivors == 2
    assert sorted(sched.answers[0]) == [0, 3, 4]
    assert sched.drain() == []
    with pytest.raises(ValueError, match="stop_groups"):
        JoinAdmission([50] * 2, [[10]], 250, arena_pages=100,
                      page_tokens=16, stop_groups=[0])
    # a wider stop admits that many anchors of a group at once
    sched = JoinAdmission([50] * 5, [[10]], 250, arena_pages=100,
                          page_tokens=16, stop_groups=[0, 0, 0, 1, 1],
                          stop_width=2)
    assert [a for a, *_ in sched.next_chunk(100)] == [0, 1, 3, 4]
    sched.report(0, 0, 0, 1, [1])
    assert sched.take_skipped() == [2] and not sched.done()


def test_filter_stop_groups_skip_a_queued_child_and_release_its_parent(
        monkeypatch):
    # document 1 borrows from 0 and passes; document 2, queued behind
    # them in the same group, never runs and the parent's hold goes
    # a 70 token budget holds one document per chunk, so one group's
    # documents run one at a time
    got, tokens, borrowed, arena = _tree_filter(
        monkeypatch, truth=[[0], [1], [1]], budget=70,
        stop_groups=[0, 0, 0])
    assert got == {0: [0], 1: [1]}
    assert borrowed == 32 and tokens == 64 + 32 + 2
    assert not arena.resident_keys() and not arena._holds
    # a parent in another group still serves its child
    got, _, borrowed, _ = _tree_filter(
        monkeypatch, truth=[[1], [1], [1]], budget=70,
        stop_groups=[0, 1, 0])
    assert got == {0: [1], 1: [1]}
    assert borrowed == 32


def test_join_anchors_borrow_a_resident_parents_pages(monkeypatch):
    from types import SimpleNamespace

    from fakes import (
        DOC,
        FRAME,
        PARTNER,
        cpu_arena,
        fake_pack,
        fake_pipeline,
        fake_torch,
    )

    from quail.backends.quail.executor import chunk as chunk_mod
    from quail.backends.quail.executor import loop

    packed = []

    def recording_pack(torch, arena, specs, **kw):
        packed.extend((s["key"], s.get("start", 0),
                       None if s["prefix"] is None else len(s["prefix"]))
                      for s in specs if s["prefix"] is not None)
        return fake_pack(torch, arena, specs, **kw)

    monkeypatch.setattr(chunk_mod, "pack_chunk", recording_pack)
    keys = [("a", d) for d in range(3)]
    # anchor 1 shares 32 tokens (2 pages) with anchor 0; anchor 2 none
    prefixes = [[DOC] * 64, [DOC] * 32 + [DOC + 1] * 32, [DOC + 2] * 64]
    tree = PrefixTree(order=[0, 1, 2], parent=[None, 0, None],
                      shared=[0, 32, 0])
    def forward(chunk):
        # partner 0 matches every anchor; frame entries answer nothing
        return [int(suffix[0] == PARTNER) if suffix[0] < FRAME else 0
                for spec in chunk.specs for suffix in spec["suffixes"]]

    pipeline = fake_pipeline(forward_chunk=forward)
    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)
    arena = cpu_arena(64)
    stats = {}
    out, _, tokens = loop.run_join(
        fake_torch(), arena, pipeline, answers, prefixes,
        [[[PARTNER + 0], [PARTNER + 1]]], 500, stage_frames=[[FRAME]],
        anchor_keys=keys, prefix_tree=tree, stats=stats)
    assert out == [{0: [1, 0], 1: [1, 0], 2: [1, 0]}]
    # the child packs its 32 own tokens from position 32
    assert packed == [(keys[0], 0, 64), (keys[1], 32, 32), (keys[2], 0, 64)]
    assert tokens == 64 + 32 + 64 + 3 * (1 + 2)
    assert stats["borrowed_tokens"] == 32
    assert arena.free_pages == 64 and not arena.accounting.owned

    # one anchor per chunk, with anchor 2 between parent and child: the
    # parent answers before the child's turn, and its pages stay held
    # until the child borrows them
    packed.clear()
    arena = cpu_arena(64)
    out, _, tokens = loop.run_join(
        fake_torch(), arena, pipeline, answers, prefixes,
        [[[PARTNER + 0], [PARTNER + 1]]], 67, stage_frames=[[FRAME]],
        anchor_keys=keys, stats=stats,
        prefix_tree=PrefixTree(order=[0, 2, 1], parent=[None, 0, None],
                               shared=[0, 32, 0]))
    assert out == [{0: [1, 0], 1: [1, 0], 2: [1, 0]}]
    assert (keys[1], 32, 32) in packed
    assert stats["borrowed_tokens"] == 32
    assert arena.free_pages == 64 and not arena.accounting.owned


def test_a_skipped_stage_passes_the_anchor_on_with_nothing_asked():
    # anchor 0 skips stage 1 of three and is asked at stage 2; anchor 1
    # skips the last stage, which counts as passed
    asks = {0: lambda j: SKIP if j == 1 else None,
            1: lambda j: SKIP if j == 2 else None}
    sched = JoinAdmission([50, 50], [[10], [10], [10]], 250,
                          arena_pages=100, page_tokens=16,
                          anchor_partners=asks)
    groups = sched.next_chunk(100)
    assert [(a, j) for a, j, *_ in groups] == [(0, 0), (1, 0)]
    assert sched.report(0, 0, 0, 1, [1]) == AdmissionReport(
        transitions=((0, 0, True),))
    assert sched.report(1, 0, 0, 1, [1]) == AdmissionReport(
        transitions=((1, 0, True),))
    assert [(a, j) for a, j, *_ in sched.next_chunk(100)] == [(0, 2), (1, 1)]
    assert sched.report(0, 2, 0, 1, [1]) == AdmissionReport(
        (Settlement("finished", 0, True),), ((0, 2, True),))
    assert sched.report(1, 1, 0, 1, [1]) == AdmissionReport(
        (Settlement("finished", 1, True),), ((1, 1, True),))
    assert sched.done() and sched.survivors == 2
    assert sched.answers[1] == {1: [1]} and sched.answers[2] == {0: [1]}
    with pytest.raises(ValueError, match="first stage cannot be skipped"):
        JoinAdmission([50], [[10]], 250, arena_pages=100, page_tokens=16,
                      anchor_partners={0: lambda j: SKIP})


@pytest.mark.parametrize("custom_decision", [False, True])
def test_stage_transition_waits_for_launches_and_respects_custom_decision(
        custom_decision):
    sched = JoinAdmission(
        [8], [[8, 8, 8, 8], [8]], 16, arena_pages=20, page_tokens=16,
        advance=(lambda a, j, row: all(row)) if custom_decision else None)
    assert sched.next_chunk(20) == [(0, 0, 0, 1, True)]
    # Even a TRUE answer cannot advance before all requests are launched.
    assert sched.report(0, 0, 0, 1, [1]) == AdmissionReport()
    assert sched.next_chunk(20) == [(0, 0, 1, 3, False)]
    assert sched.next_chunk(20) == [(0, 0, 3, 4, False)]
    partial = sched.report(0, 0, 1, 3, [0, 0])
    if custom_decision:
        # A custom decision can reject a row despite its first TRUE answer.
        assert partial == AdmissionReport()
        assert sched.next_chunk(20) == []
        assert sched.report(0, 0, 3, 4, [0]) == AdmissionReport(
            (Settlement("dropped", 0, False),), ((0, 0, False),))
    else:
        # The next stage starts before the rest of the prior row returns.
        assert partial == AdmissionReport(transitions=((0, 0, True),))
        assert sched.next_chunk(20) == [(0, 1, 0, 1, False)]
        assert sched.report(0, 0, 3, 4, [0]) == AdmissionReport()
        assert sched.report(0, 1, 0, 1, [1]) == AdmissionReport(
            (Settlement("finished", 0, True),), ((0, 1, True),))
    assert sched.done()


@pytest.mark.parametrize("selection, stages, kind", [
    (DROP, 2, "dropped"),
    ([], 2, "finished"),
    ([], 3, "dropped"),
])
def test_next_stage_settlement_preserves_the_answered_stage_decision(
        selection, stages, kind):
    sched = JoinAdmission(
        [8], [[8]] * stages, 16, arena_pages=20, page_tokens=16,
        anchor_partners={0: lambda j: None if j == 0 else selection})
    assert sched.next_chunk(20) == [(0, 0, 0, 1, True)]
    assert sched.report(0, 0, 0, 1, [1]) == AdmissionReport(
        (Settlement(kind, 0, False),), ((0, 0, True),))
    assert sched.done() and sched.survivors == 0
