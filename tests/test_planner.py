"""The planner must reproduce every measured winner."""

import numpy as np

from quail.configs import DEVICES, MODELS
from quail.plan import Plan, Refusal, StoreSpec, plan_query

H100 = DEVICES["H100-SXM-80GB"]
L40S = DEVICES["L40S-48GB"]
M4B = MODELS["Qwen3-4B-FP8"]

rng = np.random.default_rng(7)
DOCS_10K = list(rng.integers(250, 400, size=10_000))   # ~3.2M tokens
DOCS_1K = DOCS_10K[:1000]


def test_4b_10k_matches_the_measured_run():
    p = plan_query(4, DOCS_10K, M4B, H100, selectivity=0.8)
    assert p.mode == "chain" and p.workers == 1 and p.access == "read"
    # measured: 49.9 seconds, in a container whose speed was not
    # controlled (walls spread up to 45 percent across containers).
    # The band floor moved when c0 was re-anchored host-controlled
    # (3.2 s -> 0.026 s, results/engine/c0_anchor.json).
    assert 38 <= p.predicted_makespan_s <= 58


def test_sequence_cap_robust_to_size_skew():
    """One tiny outlier document must not inflate the cap. The old
    bound divided the budget by the single smallest document (a
    10-token outlier would have claimed sixty thousand sequences);
    the exact bound is the longest ascending prefix that fits. The
    engine bound (ENGINE_SEQS_MAX) clamps what per-sequence boot
    overheads can survive."""
    uniform = plan_query(4, [300] * 500, M4B, H100)
    skewed = plan_query(4, [10] + [300] * 499, M4B, H100)
    assert skewed.engine_max_seqs == uniform.engine_max_seqs
    assert skewed.engine_max_seqs == 500 + 16
    # ten thousand tiny documents all fit the budget at once, so the
    # engine bound is what stops the cap
    big = plan_query(4, [10] * 10_000, M4B, H100)
    assert big.engine_max_seqs == 4096


def test_step_budget_scales_with_pool_slack():
    """The step budget is derived, not a constant: the largest value
    whose activation reservation stays under STEP_POOL_FRACTION of
    the KV pool. A fat pool packs large steps; a thinner pool takes a
    smaller budget rather than trade KV for under a percent of wall;
    the budget never sits below the sequence cap (the engine requires
    the step budget to cover it)."""
    fat = plan_query(4, DOCS_10K, M4B, H100, selectivity=0.8)
    assert 16_384 <= fat.engine_step_tokens <= 32_768
    assert fat.engine_step_tokens >= fat.engine_max_seqs
    # the L40S holds 537,923 pool tokens against the H100's 937,228
    lean = plan_query(4, DOCS_10K, M4B, L40S, selectivity=0.8)
    assert 2_048 <= lean.engine_step_tokens < fat.engine_step_tokens
    assert lean.engine_step_tokens >= lean.engine_max_seqs


def test_spill_plans_with_a_store_instead_of_refusing():
    """A pool too small for the working set is a refusal without a
    store, and a spill plan with one: the tiering store absorbs the
    overflow, priced pessimistically at store bandwidth both ways."""
    long_docs = [30_000] * 40 + [100_000] * 10
    r = plan_query(2, long_docs, M4B, L40S, gpus=2,
                   saturation_width_docs=6)
    assert isinstance(r, Refusal)
    assert r.constraint == "pool_under_working_set"
    disk = StoreSpec(read_bw=2.7e9)
    p = plan_query(2, long_docs, M4B, L40S, gpus=2,
                   saturation_width_docs=6, store=disk)
    assert p.access == "spill"
    assert any("spills the overflow" in x for x in p.remarks)
    # the spill traffic must be priced in, not free
    no_spill_shape = plan_query(2, long_docs, M4B, L40S, gpus=2,
                                saturation_width_docs=1, store=disk)
    assert p.predicted_makespan_s >= no_spill_shape.predicted_makespan_s


def test_store_beats_recompute_only_above_the_breakeven():
    """Restoring KV from a store wins only when the store's bandwidth
    beats the rate at which prefill would recreate the same KV. At 4B
    on the H100 that break-even sits near 7.2 GB/s: a 5.2 GB/s disk
    loses to recompute, a 20 GB/s store wins."""
    slow = plan_query(4, DOCS_10K, M4B, H100, selectivity=0.8,
                      store=StoreSpec(read_bw=5.2e9, warm=True))
    fast = plan_query(4, DOCS_10K, M4B, H100, selectivity=0.8,
                      store=StoreSpec(read_bw=20e9, warm=True))
    assert slow.access == "read"
    assert fast.access == "restore"
    assert any("store beats recompute" in x for x in fast.remarks)
    assert fast.predicted_makespan_s < slow.predicted_makespan_s


def test_sharding_divides_the_floors():
    p1 = plan_query(4, DOCS_10K, M4B, H100, gpus=1, selectivity=0.8)
    p4 = plan_query(4, DOCS_10K, M4B, H100, gpus=4, selectivity=0.8)
    assert p4.workers == 4 and len(p4.shards) == 4
    sizes = [sum(DOCS_10K[i] for i in s) for s in p4.shards]
    assert max(sizes) - min(sizes) <= 400          # balanced by tokens
    assert sorted(i for s in p4.shards for i in s) == list(range(10_000))
    gain = p1.predicted_makespan_s / p4.predicted_makespan_s
    assert 2.5 < gain < 4.1                        # near-linear scaling


def test_model_bigger_than_a_card_splits_it():
    import dataclasses
    huge = dataclasses.replace(M4B, W_mem=200e9)
    p = plan_query(4, DOCS_1K, huge, H100, gpus=8, selectivity=0.8)
    assert p.tensor_parallel == 4 and p.workers == 2


def test_single_filter_uses_requests_without_pins():
    """One filter has no future consumer, so there is nothing a pin
    could buy: read, answer, free."""
    p = plan_query(1, DOCS_10K, M4B, H100)
    assert p.mode == "requests"


def test_budget_stays_under_the_pool():
    p = plan_query(4, DOCS_10K, M4B, H100)
    free = H100.M * 0.92 - M4B.W_mem
    assert p.budget_tokens <= 0.8 * free / M4B.kappa + 1


# ------------------------------------------- the refusal path (Algorithm 2)

def test_refuses_when_weights_need_more_cards_than_exist():
    import dataclasses
    huge = dataclasses.replace(M4B, W_mem=200e9)       # needs a 4-card group
    r = plan_query(4, DOCS_1K, huge, H100, gpus=2, selectivity=0.8)
    assert isinstance(r, Refusal)
    assert r.constraint == "weights_need_more_cards"
    assert r.reasons == ("weights need 4 cards, 2 available",)
    assert r.needed == 4 and r.available == 2 and r.unit == "cards"


def test_refuses_when_weights_leave_no_pool():
    import dataclasses
    # fits the 0.92 provisioning exactly, so zero bytes remain for KV
    full = dataclasses.replace(M4B, W_mem=H100.M * 0.92)
    r = plan_query(4, DOCS_10K, full, H100, selectivity=0.8)
    assert isinstance(r, Refusal)
    assert r.constraint == "pool_exhausted"
    assert "no KV pool left after weights" in r.reasons[0]


def test_refuses_when_the_pool_is_under_the_saturation_working_set():
    # the 4B-on-L40S pool holds 537,923 tokens; 1,500 resident documents
    # of the working set (largest doc 399 + question 46) need 667,500
    r = plan_query(4, DOCS_1K, M4B, L40S, selectivity=0.8,
                   saturation_width_docs=1500)
    assert isinstance(r, Refusal)
    assert r.constraint == "pool_under_working_set"
    assert r.unit == "tokens" and r.available < r.needed


def test_normal_configs_still_return_plans():
    assert isinstance(plan_query(4, DOCS_10K, M4B, H100, selectivity=0.8),
                      Plan)
