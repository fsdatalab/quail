"""The worker's CPU-side arithmetic: which join stages share one
gated run_join call. The GPU paths live in tests/gpu/."""

from quail.runtime.worker import _stage_groups


def test_consecutive_same_anchor_full_stages_share_one_group():
    # one gated run_join call: the anchor's KV is computed once and
    # the between-stage gate prunes anchors with no stage-1 match
    j1 = dict(anchor="p", semantics="full")
    j2 = dict(anchor="p", semantics="full")
    assert _stage_groups([j1, j2]) == [[j1, j2]]


def test_anchor_change_splits_groups():
    j1 = dict(anchor="p", semantics="full")
    j2 = dict(anchor="r", semantics="full")
    assert _stage_groups([j1, j2]) == [[j1], [j2]]


def test_gates_never_join_a_full_group():
    # the in-call gate keeps anchors with any TRUE; anti keeps the
    # opposite set, so it must run as its own call
    full1 = dict(anchor="p", semantics="full")
    anti = dict(anchor="p", semantics="anti")
    full2 = dict(anchor="p", semantics="full")
    assert _stage_groups([full1, anti, full2]) == \
        [[full1], [anti], [full2]]
