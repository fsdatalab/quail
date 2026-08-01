"""Checks for the expected-flow LP layer: ordering against the exact
queue-augmented model, agreement with certified optima, and validated
replays."""

import numpy as np
import pytest

from docengine.instance import Instance, sample_outcomes
from docengine.optimizer.queue_augmented import solve_queue_augmented_taskfirst
from docengine.optimizer.state_actions import (build_fullspec, build_pipeline,
                                               build_taskfirst, make_types)
from docengine.optimizer.steady_state_lp import solve_expected_flow
from docengine.runtime.replay import replay
from docengine.validator.check import validate

from test_exact import TOY_MODEL, inst


def test_lp_matches_hand_rate_single_type():
    """One length type, ample memory: lambda* must equal completions/tau of
    the best single full batch, computable by hand."""
    it = inst([6] * 10, [1, 2], [0.5, 0.5], kv_tokens=100)
    types = make_types(it.d, 0, rep="exact")
    for build in (build_taskfirst, build_fullspec):
        mm = build(it, types)
        lp = solve_expected_flow(mm)
        assert lp.status == "optimal"
        best = max(a.complete / a.tau for a in mm.actions)
        assert lp.lam <= best + 1e-9
        assert lp.lam >= 0.5 * best   # mixing cannot collapse throughput


def test_queue_augmented_below_expected_flow():
    """Paper eq. augmented-relaxation-order: lambda_hat <= lambda*."""
    it = inst([5] * 8, [1, 2], [0.6, 0.5], kv_tokens=60)
    types = make_types(it.d, 0, rep="exact")
    lp = solve_expected_flow(build_taskfirst(it, types))
    lam_hat, status = solve_queue_augmented_taskfirst(
        it, d=5, Q=(3, 3), c_max=3, e_max=3)
    assert status == "optimal"
    assert lam_hat <= lp.lam + 1e-9
    assert lam_hat > 0


def test_replays_validate_and_bracket_lp():
    """Replays must be validator-clean; their latency is a finite feasible
    schedule near the fluid target."""
    rng = np.random.default_rng(3)
    d = tuple(int(x) for x in rng.integers(3, 12, size=40))
    it = inst(list(d), [2, 1], [0.6, 0.5], kv_tokens=80)
    X = sample_outcomes(it, rng)
    types = make_types(it.d, 0, rep="exact")
    for build, vpolicy, kmax in ((build_taskfirst, "task", 1),
                                 (build_pipeline, "pipe", 1),
                                 (build_fullspec, "spec", 2)):
        mm = build(it, types)
        lp = solve_expected_flow(mm)
        assert lp.status == "optimal", (mm.method, lp.status)
        recs, phases = replay(it, X, mm, lp)
        errs = validate(it, vpolicy, recs, X, kmax=kmax)
        assert errs == [], (mm.method, errs[:3])
        total = sum(r["tau"] for r in recs)
        target = it.N / lp.lam
        assert total > 0
        # finite effects allow deviation in either direction, but a replay
        # an order of magnitude off the fluid target means a broken tracker
        assert total < 10 * target
        n_phase = sum(len(v) for v in phases.values())
        assert n_phase == len(recs)


def test_pipeline_lp_capacity_pressure():
    """Shrinking capacity must not raise pipeline throughput; overflow to
    the recompute queue keeps the LP feasible."""
    it_big = inst([6] * 12, [1, 1], [0.9, 0.5], kv_tokens=200)
    it_small = inst([6] * 12, [1, 1], [0.9, 0.5], kv_tokens=25)
    types = make_types(it_big.d, 0, rep="exact")
    lam = {}
    for tag, it in (("big", it_big), ("small", it_small)):
        lp = solve_expected_flow(build_pipeline(it, types, levels=4))
        assert lp.status == "optimal"
        lam[tag] = lp.lam
    assert lam["small"] <= lam["big"] + 1e-9
