"""Mathematical property tests for the SOL roofline formulas
(quail/quail/planner/budgets.py). The adversarial reviews (reports/
2026-08-23-sol-throughput-cost.md sections 12 and 16) already hunted
for logic bugs in how the formulas get WIRED UP to a real query.
These tests check something different and complementary: does each
formula actually BEHAVE the way the roofline model it's implementing
says it should - monotonic where it must be, quadratic where the
causal-attention derivation says it must be, exactly linear where
there's genuinely no max() involved, and subadditive in exactly the
way the "aggregate before max()" design note (budgets.py, the module
comment above the attention functions) claims. No GPU needed - this
is pure arithmetic, checked against itself and against real hardware
constants."""

import pytest

from quail.specs import H100_SXM, QWEN3_4B_FP8
from quail.planner import budgets as b

M, D = QWEN3_4B_FP8, H100_SXM


# ---- monotonicity: more work must never take less time -----------------

def test_causal_attention_monotonic_in_document_length():
    lengths = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    times = [b._causal_prefill_attention_time(M, D, [L]) for L in lengths]
    assert times == sorted(times)
    assert times[0] < times[-1]   # strictly, not just non-decreasing


def test_streaming_attention_monotonic_in_context():
    contexts = [10, 100, 1_000, 10_000, 100_000]
    times = [b._shared_context_attention_time(M, D, [(40, c)])
            for c in contexts]
    assert times == sorted(times)
    assert times[0] < times[-1]


def test_streaming_attention_monotonic_in_chunk():
    chunks = [1, 10, 100, 1_000]
    times = [b._shared_context_attention_time(M, D, [(c, 1_200)])
            for c in chunks]
    assert times == sorted(times)
    assert times[0] < times[-1]


def test_projection_time_monotonic_in_chunk():
    chunks = [1, 100, 1_000, 100_000, 10_000_000]
    times = [b._projection_time(M, D, c) for c in chunks]
    assert times == sorted(times)
    assert times[0] < times[-1]


def test_elementwise_time_monotonic_in_chunk():
    chunks = [1, 100, 1_000, 100_000]
    times = [b.elementwise_time(M, D, c) for c in chunks]
    assert times == sorted(times)
    assert times[0] < times[-1]


def test_more_layers_never_cheaper():
    """layers is a pure multiplier outside every max() in these
    formulas - a model with more layers must never get a smaller
    floor for the same token counts."""
    from dataclasses import replace
    deep = replace(M, layers=M.layers * 2)
    assert b._projection_time(deep, D, 2000) > b._projection_time(M, D, 2000)
    assert (b._causal_prefill_attention_time(deep, D, [1200])
           > b._causal_prefill_attention_time(M, D, [1200]))
    assert (b._shared_context_attention_time(deep, D, [(40, 1200)])
           > b._shared_context_attention_time(M, D, [(40, 1200)]))
    assert b.elementwise_time(deep, D, 2000) > b.elementwise_time(M, D, 2000)


# ---- exact scaling laws, not just "goes up" -----------------------------

def test_elementwise_time_is_exactly_linear_in_chunk():
    """No max() anywhere in this formula - doubling chunk must
    exactly double the time, to floating-point precision."""
    t1 = b.elementwise_time(M, D, 5_000)
    t2 = b.elementwise_time(M, D, 10_000)
    assert t2 == pytest.approx(2 * t1, rel=1e-12)


def test_causal_attention_is_quadratic_in_document_length_when_compute_bound():
    """Deep in the compute-bound regime (far above the causal
    crossover, ~24,639 tokens at this model's shape), the memory term
    is negligible and flops ~ L*(L+1) ~ L^2 for large L. Doubling L
    should almost exactly quadruple the time - not double (that would
    mean the code is treating it as linear/shared-context shaped by
    mistake, exactly the bug class the causal-vs-streaming split in
    section 04 of the write-up exists to avoid)."""
    L = 50_000_000   # far above the crossover, and L>>1 so L*(L+1)~=L^2
    t1 = b._causal_prefill_attention_time(M, D, [L])
    t2 = b._causal_prefill_attention_time(M, D, [2 * L])
    ratio = t2 / t1
    assert ratio == pytest.approx(4.0, rel=1e-3)


def test_causal_attention_is_not_quadratic_below_the_crossover():
    """Below the crossover the memory term dominates, and moved ~
    linear in chunk (= L for one document) - doubling L there should
    roughly double the time, not quadruple it. Confirms the formula
    genuinely has two regimes, not a constant scaling law throughout."""
    L = 500   # well under the ~24,639-token crossover
    t1 = b._causal_prefill_attention_time(M, D, [L])
    t2 = b._causal_prefill_attention_time(M, D, [2 * L])
    ratio = t2 / t1
    assert ratio == pytest.approx(2.0, rel=0.05)
    assert ratio < 3.0   # nowhere near the compute-bound regime's 4x


def test_streaming_attention_scales_with_chunk_times_context_product_when_compute_bound():
    """flops = 4*chunk*context*n_q*d_head - in the compute-bound
    regime, doubling BOTH chunk and context should roughly quadruple
    the time (product of two doublings), same reasoning as the causal
    quadratic test but for the shared-context shape."""
    chunk, context = 2_000_000, 2_000_000   # deep compute-bound
    t1 = b._shared_context_attention_time(M, D, [(chunk, context)])
    t2 = b._shared_context_attention_time(M, D, [(2 * chunk, 2 * context)])
    assert t2 / t1 == pytest.approx(4.0, rel=1e-6)


def test_projection_time_scales_linearly_with_chunk_when_compute_bound():
    """Deep compute-bound (chunk >> kernel_index_cap-scale), flops ~
    chunk exactly, and the fixed weight-read memory term is
    irrelevant to which side of max() wins - doubling chunk should
    almost exactly double the time."""
    chunk = 50_000_000
    t1 = b._projection_time(M, D, chunk)
    t2 = b._projection_time(M, D, 2 * chunk)
    assert t2 / t1 == pytest.approx(2.0, rel=1e-6)


# ---- the ridge/crossover points are self-consistent ---------------------

def test_compute_knee_is_where_projection_flops_and_memory_are_equal():
    """compute_knee() claims to be the chunk size where the dense
    projections' compute and memory sides cross. Verify directly: at
    that chunk, the flops-bound and memory-bound estimates (computed
    the same way _projection_time does internally) should be equal,
    not just "close to whatever _projection_time returns" (which
    would be true near either side of a real crossover too)."""
    knee = b.compute_knee(M, D)
    tot_p = tot_io = 0.0
    for din, dout in b._projection_shapes(M):
        tot_p += din * dout
        tot_io += din + dout
    flops_time = 2.0 * tot_p * knee / D.peak_flops
    moved_time = (tot_p * M.w_bytes + knee * tot_io * b.ACT_BYTES) / D.hbm_bw
    assert flops_time == pytest.approx(moved_time, rel=1e-6)


def test_attention_crossover_is_where_causal_attention_equals_projection():
    """attention_crossover() (used for the shared-context shape) is
    defined as where attention time equals dense-projection time at
    the chunk budget. Verify the binary search actually landed there,
    not just near it."""
    chunk = b.chunk_budget(M, D)
    crossover = b.attention_crossover(M, D, chunk)
    t_dense = b._projection_time(M, D, chunk)
    t_attn = b._attention_time(M, D, chunk, int(crossover))
    assert t_attn == pytest.approx(t_dense, rel=1e-3)


# ---- subadditivity: the whole reason batching is aggregated first -------

def test_streaming_attention_is_subadditive_across_items():
    """The module comment above _shared_context_attention_time claims
    sum(max(a_i,b_i)) >= max(sum(a_i), sum(b_i)) - i.e. computing many
    small streaming items' true combined cost (aggregate first, one
    max()) must come out LESS THAN OR EQUAL to pricing each item
    separately and summing the results. Use a deliberately lopsided
    mix (one compute-heavy item, one memory-heavy item) so the two
    approaches can actually differ, not coincide."""
    items = [(5_000_000, 5_000_000),   # huge chunk*context -> compute-bound
             (10, 50)]                  # tiny -> memory-bound
    combined = b._shared_context_attention_time(M, D, items)
    summed_separately = sum(
        b._shared_context_attention_time(M, D, [item]) for item in items)
    assert combined <= summed_separately + 1e-15
    assert combined < summed_separately   # strictly, for this lopsided mix


def test_sol_seconds_is_subadditive_across_independent_workloads():
    """Same property, one level up: combining two independent
    workloads into one sol_seconds() call must not produce MORE time
    than computing each separately and adding the results - matches
    how a real fused batch kernel behaves (shared per-launch fixed
    costs get amortized once, not paid twice)."""
    workload_a = dict(causal_doc_lengths=[50_000_000],
                      streaming_chunks_contexts=[])
    workload_b = dict(causal_doc_lengths=[],
                      streaming_chunks_contexts=[(10, 50)] * 1000)
    combined = b.sol_seconds(
        M, D,
        causal_doc_lengths=(workload_a["causal_doc_lengths"]
                            + workload_b["causal_doc_lengths"]),
        streaming_chunks_contexts=(workload_a["streaming_chunks_contexts"]
                                   + workload_b["streaming_chunks_contexts"]))
    separate = b.sol_seconds(M, D, **workload_a) + b.sol_seconds(M, D, **workload_b)
    assert combined <= separate + 1e-12


# ---- breakdown matches the total, and matches real measured data --------

def test_sol_seconds_breakdown_sums_to_sol_seconds():
    """sol_seconds() is defined as sum(sol_seconds_breakdown().values())
    - lock that relationship down directly so the two can't drift
    apart if either is edited independently in the future."""
    causal = [1200, 900, 1500]
    streaming = [(40, 1200)] * 200
    total = b.sol_seconds(M, D, causal_doc_lengths=causal,
                          streaming_chunks_contexts=streaming)
    parts = b.sol_seconds_breakdown(M, D, causal_doc_lengths=causal,
                                    streaming_chunks_contexts=streaming)
    assert set(parts) == {"projection", "elementwise", "causal_attention",
                          "streaming_attention"}
    assert sum(parts.values()) == pytest.approx(total, rel=1e-12)
    assert all(v >= 0 for v in parts.values())


def test_spec_ceiling_exceeds_every_real_measured_tokens_per_s():
    """spec_ceiling_tokens_per_s is supposed to be the absolute
    theoretical maximum. Ground it against reality, not just internal
    consistency: every tokens_per_s actually measured across the
    validated GPU runs (reports/2026-08-23-sol-throughput-cost.md
    section 15's table) must be strictly below it - the same
    "measured can't beat the floor" invariant the whole feature
    enforces per-query, checked here against the headline number
    directly."""
    measured_tokens_per_s = [
        117811, 115160, 115599, 80877, 112434,   # cold pass
        107464, 108559, 106122, 79427, 96143,    # warm pass
    ]
    ceiling = b.spec_ceiling_tokens_per_s(M, D)
    assert all(m < ceiling for m in measured_tokens_per_s)
    # and it should be in the right ballpark - not just "some big
    # number", the real ~275k/s figure quoted throughout the write-up
    assert ceiling == pytest.approx(274_861, rel=0.01)


# ---- edge cases the roofline arithmetic must not choke on ---------------

@pytest.mark.parametrize("L", [1, 2])
def test_causal_attention_tiny_documents_no_crash(L):
    t = b._causal_prefill_attention_time(M, D, [L])
    assert t > 0


def test_causal_attention_empty_batch_is_zero():
    assert b._causal_prefill_attention_time(M, D, []) == 0.0


def test_streaming_attention_empty_batch_is_zero():
    assert b._shared_context_attention_time(M, D, []) == 0.0


def test_sol_seconds_zero_workload_is_zero():
    assert b.sol_seconds(M, D, causal_doc_lengths=[],
                         streaming_chunks_contexts=[]) == 0.0
