"""Checks for the n-filter blockwise LP and the multi-GPU wrappers."""

import numpy as np

from docengine.instance import Instance, sample_outcomes
from docengine.sched.blockwise import schedule_blockwise
from docengine.validator.check import validate

from attic.theory.cluster import partition_docs, run_builder_multi
from attic.theory.optimizer.state_actions import (build_blockwise_lp,
                                                  build_fullspec,
                                                  build_pipeline, make_types)
from attic.theory.optimizer.steady_state_lp import solve_expected_flow

from test_exact_reference import inst


def test_blockwise_lp_consistency():
    """k=1 at n=2 must match the dedicated pipeline generator; k=n must
    collapse to one cache state and match full speculation."""
    it = inst([6] * 12, [1, 2], [0.6, 0.5], kv_tokens=120)
    types = make_types(it.d, 0, rep="exact")
    lam_pipe = solve_expected_flow(build_pipeline(it, types)).lam
    lam_k1 = solve_expected_flow(build_blockwise_lp(it, types, 1)).lam
    assert abs(lam_pipe - lam_k1) <= 0.02 * lam_pipe
    mm = build_blockwise_lp(it, types, 2)
    assert len(mm.states) == 1
    lam_kn = solve_expected_flow(mm).lam
    lam_fs = solve_expected_flow(build_fullspec(it, types)).lam
    assert abs(lam_kn - lam_fs) <= 0.02 * lam_fs


def test_blockwise_lp_three_filters():
    """n=3: lookahead 2 sits between the pipeline and full speculation on a
    memory-ample instance (its extra work grows with k)."""
    it = inst([6] * 12, [1, 1, 1], [0.8, 0.8, 0.8], kv_tokens=200)
    types = make_types(it.d, 0, rep="exact")
    lam = {k: solve_expected_flow(build_blockwise_lp(it, types, k)).lam
           for k in (1, 2, 3)}
    assert lam[1] >= lam[2] - 1e-9
    assert lam[2] >= lam[3] - 1e-9


def test_partition_balance():
    rng = np.random.default_rng(0)
    d = rng.integers(50, 500, size=200).tolist()
    for G in (2, 4, 8):
        shares = partition_docs(d, G)
        loads = [sum(d[i] for i in s) for s in shares]
        assert max(loads) - min(loads) <= max(d)
        assert sorted(i for s in shares for i in s) == list(range(len(d)))


def test_multi_gpu_builder_scaling():
    rng = np.random.default_rng(5)
    d = rng.integers(3, 12, size=48).tolist()
    it = inst(d, [1, 1], [0.6, 0.5], kv_tokens=100)
    X = sample_outcomes(it, rng)

    def builder(sub, Xs):
        return schedule_blockwise(sub, Xs, 1)

    def val(sub, recs, Xs):
        return validate(sub, "pipe", recs, Xs, kmax=1)

    mk1, _ = run_builder_multi(it, X, 1, builder, validator=val)
    mk2, per = run_builder_multi(it, X, 2, builder, validator=val)
    assert mk2 <= mk1
    assert mk2 >= 0.45 * mk1        # cannot beat perfect halving by much
    assert len(per) == 2


def test_additive_repricing_orders():
    """tau_add >= tau_max on any schedule (the max cannot exceed the sum),
    and the additive class bound holds for the pipeline builder."""
    from attic.theory.reprice import reprice_records, resource_lb_additive
    rng = np.random.default_rng(11)
    d = rng.integers(4, 14, size=30).tolist()
    it = inst(d, [1, 2], [0.7, 0.5], kv_tokens=90)
    X = sample_outcomes(it, rng)
    recs = schedule_blockwise(it, X, 1)
    assert validate(it, "pipe", recs, X, kmax=1) == []
    r = reprice_records(it, recs)
    assert r["tau_add"] >= r["tau_max"] - 1e-12
    assert resource_lb_additive(it, "pipe", X, k=1) <= r["tau_add"] + 1e-9
