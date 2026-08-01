"""Verification battery for the exact solvers (paper sec. 10.5 step 1 and the
sec. 11 checklist). Run with pytest."""

import math
import random

import numpy as np
import pytest

from docengine.configs import DeviceConfig, ModelConfig
from docengine.costmodel import a_pairs, dense_time, attn_time
from docengine.reference.offline import solve_offline
from docengine.reference.online import solve_online
from docengine.instance import Instance, survival
from docengine.lb import resource_lb
from docengine.manifest import emit
from docengine.validator.check import validate

# A tiny synthetic model/device so that memory pressure and weight reads are
# exercised at hand-checkable scale. kappa = 2*2*1*2*1 = 8 bytes/token.
TOY_MODEL = ModelConfig(name="toy", P=1e6, L=2, h=4, n_q=2, n_kv=1, d_h=2,
                        L_ctx=64, q_kv=1, W_mem=1_000.0, W_run=1_000.0)


def toy_device(kv_tokens: int) -> DeviceConfig:
    """Device whose free-KV capacity is exactly kv_tokens."""
    M = TOY_MODEL.W_mem + 100.0 + kv_tokens * TOY_MODEL.kappa
    return DeviceConfig(name=f"toy-{kv_tokens}", M=M, BW=1e6, R_D=1e9, R_A=1e9,
                        S=100.0)


def inst(d, p, s, kv_tokens=10_000, delta=1, cap=None, model=TOY_MODEL):
    return Instance(model=model, device=toy_device(kv_tokens), d=tuple(d),
                    p=tuple(p), s=tuple(s), delta=delta, max_new_tokens=cap)


def test_chunk_attention_invariance():
    rng = random.Random(0)
    for _ in range(200):
        d = rng.randint(1, 40)
        cuts = sorted(rng.sample(range(1, d), rng.randint(0, min(5, d - 1))))
        parts, prev = [], 0
        for c in cuts + [d]:
            parts.append(c - prev)
            prev = c
        total, c = 0, 0
        for q in parts:
            total += a_pairs(c, q)
            c += q
        assert total == d * (d + 1) // 2  # eq. (16)


def hand_tau(U, A, K_L, K_W, dev):
    D = max(2 * TOY_MODEL.P * U / dev.R_D, TOY_MODEL.W_run / dev.BW) if U else 0
    H = max(4 * TOY_MODEL.L * TOY_MODEL.attn_width * A / dev.R_A,
            TOY_MODEL.kappa * (K_L + K_W) / dev.BW)
    return D + H


def test_single_doc_single_filter_hand_check():
    """N=1, n=1, d=4, p=2, ample memory: optimum is one batch for each policy.

    Write-through ledger: task-first stores the prompt block (2) plus the doc
    interior (4-1=3, decision leaf ephemeral) -> K_W=5; pipeline stores the
    doc (4, branches consume it) plus the branch interior (2-1=1) -> K_W=5."""
    it = inst([4], [2], [0.5])
    X = np.array([[1]])
    dev = it.device
    # task-first: prompt prefill (U2, A3) + doc under prompt (U4, A=a(2,4)=18)
    v_task, sched = solve_offline(it, "task", X)
    assert len(sched) == 1
    assert math.isclose(v_task, hand_tau(6, 21, 0, 5, dev), rel_tol=1e-12)
    # pipeline: doc (U4, A10) + branch (U2, A=a(4,2)=11)
    v_pipe, sched_p = solve_offline(it, "pipe", X)
    assert len(sched_p) == 1
    assert math.isclose(v_pipe, hand_tau(6, 21, 0, 5, dev), rel_tol=1e-12)
    # full speculation with n=1 degenerates to the pipeline batch
    v_spec, _ = solve_offline(it, "spec", X, kmax=1)
    assert math.isclose(v_spec, v_pipe, rel_tol=1e-12)
    # n=1: outcomes cannot create or remove work -> online == offline
    v_on, _states = solve_online(it, "pipe")
    assert math.isclose(v_on, v_pipe, rel_tol=1e-9)


def test_chunking_dominance():
    """OPT_chunk <= OPT_atomic (eq. 31), strict when chunking removes a
    weight-read event (paper sec. 6.6): task-first, docs 3,3,3, prompt 1,
    5-new-token cap. Atomic needs 3 batches (two whole docs = 6 > 5);
    chunked packs 10 tokens into 2 full batches."""
    from dataclasses import replace
    heavy = replace(TOY_MODEL, W_run=1e5)   # weight-read-dominated regime
    X = np.array([[1], [1], [1]])
    fine = inst([3, 3, 3], [1, ], [0.5], cap=5, delta=1, model=heavy)
    coarse = inst([3, 3, 3], [1, ], [0.5], cap=5, delta=3, model=heavy)
    v_fine, sched_f = solve_offline(fine, "task", X, evict_mode="none")
    v_coarse, sched_c = solve_offline(coarse, "task", X, evict_mode="none")
    assert v_fine <= v_coarse + 1e-12
    assert v_fine < v_coarse - 1e-9   # strictly fewer weight reads
    assert len(sched_f) == 2 and len(sched_c) == 3


def test_speculation_outcome_independence():
    """eq. (41): FORCED full speculation has identical value for every X and
    equals its online value. Adaptive speculation (which contains pipeline)
    is outcome-dependent offline but never worse than fullspec."""
    it = inst([3], [2, 2], [0.5, 0.5])
    vals = []
    for x1 in (0, 1):
        for x2 in (0, 1):
            X = np.array([[x1, x2]])
            v, sched = solve_offline(it, "fullspec", X, kmax=2)
            vals.append(v)
            # the fused batch evaluates both branches at once
            assert any(rec["batch"].branches == ((0, 2),) for rec in sched)
            v_adapt, _ = solve_offline(it, "spec", X, kmax=2)
            assert v_adapt <= v + 1e-12
    assert max(vals) - min(vals) < 1e-12
    v_on, _ = solve_online(it, "fullspec", kmax=2)
    assert math.isclose(v_on, vals[0], rel_tol=1e-9)


def test_value_of_information():
    """eq. (42): E_X[OPT_off] <= OPT_on for task-first and pipeline."""
    it = inst([2, 3], [1, 1], [0.5, 0.5], kv_tokens=8)
    for policy in ("task", "pipe"):
        e_off = 0.0
        for bits in range(16):
            X = np.array([[(bits >> 0) & 1, (bits >> 1) & 1],
                          [(bits >> 2) & 1, (bits >> 3) & 1]])
            v, _ = solve_offline(it, policy, X, evict_mode="binary")
            e_off += v / 16.0
        v_on, _ = solve_online(it, policy, evict_mode="binary")
        assert e_off <= v_on + 1e-9, (policy, e_off, v_on)


def test_memory_forced_eviction():
    """Free KV capacity below sum(d): pipeline must evict and recompute; the
    solved schedule must exist, validate, and cost more than the unconstrained
    one."""
    X = np.array([[1, 1], [1, 1]])
    tight = inst([4, 4], [1, 1], [0.9, 0.9], kv_tokens=6)
    loose = inst([4, 4], [1, 1], [0.9, 0.9], kv_tokens=1000)
    v_t, sched_t = solve_offline(tight, "pipe", X, evict_mode="binary")
    v_l, _ = solve_offline(loose, "pipe", X, evict_mode="binary")
    assert v_t >= v_l - 1e-12
    errs = validate(tight, "pipe", emit(tight, "pipe", sched_t, X), X)
    assert errs == []


def test_lb_below_opt():
    X = np.array([[1, 0], [1, 1], [0, 1]])
    it = inst([2, 3, 2], [1, 1], [0.5, 0.5], kv_tokens=12)
    for policy, kmax in (("task", 1), ("pipe", 1), ("spec", 2), ("fullspec", 2)):
        v, _ = solve_offline(it, policy, X, kmax=kmax, evict_mode="binary")
        lb = resource_lb(it, policy, X)["LB"]
        assert lb <= v + 1e-12, (policy, lb, v)


def test_validator_accepts_and_rejects():
    X = np.array([[1, 1], [0, 0]])
    it = inst([3, 2], [1, 2], [0.5, 0.5], kv_tokens=20)
    for policy, kmax in (("task", 1), ("pipe", 1), ("spec", 2)):
        v, sched = solve_offline(it, policy, X, kmax=kmax, evict_mode="binary")
        recs = emit(it, policy, sched, X)
        assert validate(it, policy, recs, X, kmax=kmax) == []
        bad = [dict(r) for r in recs]
        bad[0] = dict(bad[0], U=bad[0]["U"] + 1)
        assert validate(it, policy, bad, X, kmax=kmax) != []
    # an unrevealed-outcome violation: force F2 branch into the F1 batch
    v, sched = solve_offline(it, "pipe", X, evict_mode="binary")
    recs = emit(it, "pipe", sched, X)
    for rec in recs:
        for op in rec["ops"]:
            if op["kind"] == "branch" and op["stage"] == 1:
                op2 = dict(op, stage=2)
                rec["ops"].append(op2)
                break
        else:
            continue
        break
    assert validate(it, "pipe", recs, X) != []


def test_survival_matrix():
    X = np.array([[1, 1, 0], [0, 1, 1]])
    Y = survival(X)
    assert Y.tolist() == [[1, 1, 1], [1, 0, 0]]
