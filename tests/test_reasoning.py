"""Checks for the reasoning-filter analytical layer: consistency with
the answer-only world at g=1, and the structural predictions of
notes/REASONING_MODEL.md."""

import math

from docengine.configs import DEVICES, MODELS
from docengine.reasoning.lp import lp_throughput
from docengine.reasoning.model import (RInstance, best_composition,
                                       compositions, t_pre,
                                       value_blockwise, value_taskfirst)


def inst(n, s, g, N=10000, d=313.0, calib=1.0):
    return RInstance(model=MODELS["Qwen3-4B-FP8"],
                     device=DEVICES["H100-SXM-80GB"],
                     N=N, d=d, s=(s,) * n, p=(25,) * n, g=(g,) * n,
                     calib=calib)


def test_anchor_answer_only():
    """At g=1 the pipeline's value must sit between the pure reading
    floor and 1.6x of it, bracketing the measured world's behavior."""
    it = inst(4, 0.8, 1)
    floor = t_pre(it, it.N * it.d)
    v = value_blockwise(it, (1, 1, 1, 1))
    assert floor <= v["T"] <= 1.6 * floor
    vt = value_taskfirst(it)
    assert vt["T"] > 2.5 * v["T"]          # re-reading is ruinous at n=4


def test_monotone_in_thinking():
    for K in ((1, 1, 1, 1), (2, 2), (4,)):
        prev = 0.0
        for g in (1, 33, 129, 513):
            v = value_blockwise(inst(4, 0.8, g), K)["T"]
            assert v > prev
            prev = v


def test_speculation_punished_by_thinking():
    """At g=1 full speculation is within a third of the pipeline; at
    g=513 its wasted thinking makes it clearly worse."""
    lo_p = value_blockwise(inst(4, 0.8, 1), (1, 1, 1, 1))["T"]
    lo_s = value_blockwise(inst(4, 0.8, 1), (4,))["T"]
    hi_p = value_blockwise(inst(4, 0.8, 513), (1, 1, 1, 1))["T"]
    hi_s = value_blockwise(inst(4, 0.8, 513), (4,))["T"]
    assert lo_s <= 1.35 * lo_p
    assert hi_s / hi_p > lo_s / lo_p       # ratio degrades with thinking
    assert hi_s > hi_p                     # and speculation loses outright


def test_blocks_shrink_with_thinking():
    b1 = value_blockwise(inst(4, 0.8, 1), (4,))["block_docs"]
    b2 = value_blockwise(inst(4, 0.8, 513), (4,))["block_docs"]
    assert b2 < 0.5 * b1


def test_bellman_composition_consistent():
    """The recurrence's optimum never loses to any fixed composition,
    and enumeration covers the 2^(n-1) compositions."""
    it = inst(4, 0.95, 129)
    assert len(compositions(4)) == 8
    K, v = best_composition(it)
    for K2 in compositions(4):
        assert v["T"] <= value_blockwise(it, K2)["T"] + 1e-9


def test_lp_and_recurrence_agree():
    """Steady throughput and makespan describe the same fluid: lambda
    and N/T within 40 percent of each other for the pipeline."""
    for g in (1, 129):
        it = inst(4, 0.8, g)
        lam = lp_throughput(it, (1, 1, 1, 1))["lam"]
        T = value_blockwise(it, (1, 1, 1, 1))["T"]
        ratio = lam / (it.N / T)
        assert 0.6 < ratio < 1.67, (g, lam, it.N / T)


def test_task_lp_below_pipeline_lp():
    for g in (1, 129):
        it = inst(4, 0.8, g)
        lam_t = lp_throughput(it, None)["lam"]
        lam_p = lp_throughput(it, (1, 1, 1, 1))["lam"]
        assert lam_t < lam_p


def test_generation_becomes_the_bottleneck():
    """Above modest thinking lengths the decode share dominates and the
    policy gap compresses (prediction one of the note)."""
    lo = inst(4, 0.95, 1)
    hi = inst(4, 0.95, 513)
    gap_lo = value_taskfirst(lo)["T"] / value_blockwise(lo, (1,) * 4)["T"]
    gap_hi = value_taskfirst(hi)["T"] / value_blockwise(hi, (1,) * 4)["T"]
    assert gap_hi < gap_lo
    assert not math.isclose(gap_hi, gap_lo, rel_tol=0.05)
