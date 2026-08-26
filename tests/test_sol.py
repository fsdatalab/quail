"""quail/planner/sol.py's three equations, checked against a research
note's two worked exercises (single filter over 5,000 IMDB reviews;
two filters chained). The note's own printed dense-time figures were
~1% high from a stale peak-flops constant applied twice - these tests
assert the values a clean computation against the repo's real specs
(quail.specs.qwen3_4b.QWEN3_4B_FP8, quail.specs.h100_sxm.H100_SXM)
produces, not the note's raw numbers. `context_reads` for exercise 2
isn't stated directly in the note; it's back-solved from the note's
own stated memory-time result (0.1753s) and confirmed to reproduce it
exactly here.
"""

import pytest

from quail.planner import sol
from quail.specs.h100_sxm import H100_SXM
from quail.specs.qwen3_4b import QWEN3_4B_FP8

MODEL, DEVICE = QWEN3_4B_FP8, H100_SXM


def test_exercise_1_single_filter():
    tokens, pairs, context_reads = 1_774_233, 444_634_043, 0
    assert sol.dense_seconds(MODEL, DEVICE, tokens) == \
        pytest.approx(6.455016, rel=1e-5)
    assert sol.attention_seconds(MODEL, DEVICE, pairs) == \
        pytest.approx(0.265039, rel=1e-5)
    assert sol.memory_seconds(MODEL, DEVICE, tokens, context_reads) == \
        pytest.approx(0.100932, rel=1e-5)
    assert sol.sol_seconds(MODEL, DEVICE, tokens=tokens, pairs=pairs,
                           context_reads=context_reads) == \
        pytest.approx(6.720055, rel=1e-5)


def test_exercise_2_two_filters():
    tokens = 1_968_390
    pairs = 444_634_043 + 68_895_937 + 4_659_768
    context_reads = 1_465_871
    assert sol.dense_seconds(MODEL, DEVICE, tokens) == \
        pytest.approx(7.161399, rel=1e-5)
    assert sol.attention_seconds(MODEL, DEVICE, pairs) == \
        pytest.approx(0.308884, rel=1e-5)
    assert sol.memory_seconds(MODEL, DEVICE, tokens, context_reads) == \
        pytest.approx(0.175344, rel=1e-5)
    assert sol.sol_seconds(MODEL, DEVICE, tokens=tokens, pairs=pairs,
                           context_reads=context_reads) == \
        pytest.approx(7.470283, rel=1e-5)


def test_zero_tokens_is_zero():
    assert sol.sol_seconds(MODEL, DEVICE, tokens=0, pairs=0,
                           context_reads=0) == 0.0


def test_compute_bound_regime_ignores_memory():
    # Prefill-heavy workloads (both exercises above) are compute-bound
    # by roughly two orders of magnitude - the max() should pick the
    # dense+attention side, not memory, whenever compute dominates
    # this much.
    tokens, pairs = 1_774_233, 444_634_043
    compute = (sol.dense_seconds(MODEL, DEVICE, tokens)
              + sol.attention_seconds(MODEL, DEVICE, pairs))
    memory = sol.memory_seconds(MODEL, DEVICE, tokens, 0)
    assert compute > memory
    assert sol.sol_seconds(MODEL, DEVICE, tokens=tokens, pairs=pairs,
                           context_reads=0) == compute


def test_monotonic_in_tokens_and_pairs():
    base = sol.sol_seconds(MODEL, DEVICE, tokens=1000, pairs=5000,
                           context_reads=0)
    more_tokens = sol.sol_seconds(MODEL, DEVICE, tokens=2000, pairs=5000,
                                  context_reads=0)
    more_pairs = sol.sol_seconds(MODEL, DEVICE, tokens=1000, pairs=10000,
                                 context_reads=0)
    assert more_tokens > base
    assert more_pairs > base