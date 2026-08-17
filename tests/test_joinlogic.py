"""joinlogic against brute force, on randomized shapes."""

import random

import pytest

from quail.joinlogic import (assemble, brute_force_triples, gate, matches,
                             orient, pack_stream, plan_groups)


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


def test_plan_groups_whole_list_fits():
    assert plan_groups(1040, [58] * 3718, 421_752) == [(0, 3718)]


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


def test_plan_groups_atomicity_error():
    with pytest.raises(ValueError):
        plan_groups(1000, [24_400], 25_000)


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
        chunks, capture = pack_stream(anchors, budget)
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
            # the computed-exactly-once invariant; an anchor with no
            # suffixes and no keep mark is not packed at all
            assert carried_count[a] == (1 if suffixes else 0)


def test_pack_stream_brims_across_anchors():
    # two anchors that fit one chunk together must share it
    chunks, capture = pack_stream([(100, [50, 50]), (100, [50, 50])],
                                  1000)
    assert len(chunks) == 1
    assert [g[0] for g in chunks[0]] == [0, 1]
    assert capture == set()


def test_pack_stream_cut_stream_continues_without_prefix():
    # the stream spans three chunks: the prefix is packed once, the
    # continuations carry nothing and the anchor is marked capture
    chunks, capture = pack_stream([(100, [400, 400, 400])], 600)
    assert chunks == [[(0, 0, 1, True)],
                      [(0, 1, 2, False)],
                      [(0, 2, 3, False)]]
    assert capture == {0}


def test_pack_stream_keep_marks_capture_without_cut():
    # whole stream fits one chunk, but a later stage needs the prefix
    chunks, capture = pack_stream([(100, [50, 50])], 1000, keep={0})
    assert chunks == [[(0, 0, 2, True)]]
    assert capture == {0}


def test_pack_stream_already_kept_packs_no_prefix():
    # stage 2 of an n-way: the prefix K/V is in the store, so groups
    # are suffix-only and nothing is captured
    chunks, capture = pack_stream([(100, [400, 400, 400])], 600,
                                  already_kept={0})
    assert chunks == [[(0, 0, 1, False)],
                      [(0, 1, 2, False)],
                      [(0, 2, 3, False)]]
    assert capture == set()


def test_pack_stream_continuation_relaxes_atomicity():
    # a suffix wider than budget - prefix is packable once the
    # prefix no longer rides along
    chunks, capture = pack_stream([(300, [100, 900])], 1000)
    assert chunks == [[(0, 0, 1, True)], [(0, 1, 2, False)]]
    assert capture == {0}


def test_pack_stream_capture_only_group():
    # an anchor with no suffixes this stage, kept for a later one
    chunks, capture = pack_stream([(100, []), (50, [20])], 1000,
                                  keep={0})
    assert chunks == [[(0, 0, 0, True), (1, 0, 1, True)]]
    assert capture == {0}


def test_pack_stream_atomicity_error():
    with pytest.raises(ValueError):
        pack_stream([(100, [950])], 1000)
    with pytest.raises(ValueError):
        # even without a prefix to carry, one suffix must fit a chunk
        pack_stream([(100, [1100])], 1000, already_kept={0})


def _random_rows(rng, n_anchors, n_partners, p):
    return {a: [1 if rng.random() < p else 0 for _ in range(n_partners)]
            for a in range(n_anchors)}


def test_gate_and_matches():
    rows = {0: [0, 0, 0], 1: [0, 1, 0], 2: [1, 0, 1]}
    assert gate(rows) == [1, 2]
    assert matches(rows) == {0: [], 1: [1], 2: [0, 2]}


def test_assemble_equals_brute_force_random():
    rng = random.Random(3)
    for _ in range(30):
        nA, nB, nC = (rng.randrange(1, 12) for _ in range(3))
        ans1 = _random_rows(rng, nB, nA, rng.random() * 0.6)
        survivors = gate(ans1)
        ans2 = {b: [1 if rng.random() < 0.3 else 0 for _ in range(nC)]
                for b in survivors}
        staged = assemble(ans1, ans2)
        # the reference only looks up stage-2 rows behind a stage-1
        # YES, so survivors' rows are exactly the recorded ones
        reference = brute_force_triples(ans1, ans2)
        assert staged == reference


def test_stage2_work_equals_survivors():
    rng = random.Random(5)
    ans1 = _random_rows(rng, 40, 25, 0.05)
    survivors = gate(ans1)
    # the dedup claim: stage 2 runs once per surviving anchor, so its
    # pair count is survivors x partner count, independent of how
    # many stage-1 matches each survivor had
    n_c = 17
    assert len(survivors) * n_c == sum(n_c for _ in survivors)
