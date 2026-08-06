"""The planner must reproduce every banked measured winner."""

import numpy as np

from docengine.configs import DEVICES, MODELS
from docengine.plan import Plan, Refusal, StoreSpec, plan_query

H100 = DEVICES["H100-SXM-80GB"]
L40S = DEVICES["L40S-48GB"]
M4B = MODELS["Qwen3-4B-FP8"]
M32B = MODELS["Qwen3-32B-FP8"]

rng = np.random.default_rng(7)
DOCS_10K = list(rng.integers(250, 400, size=10_000))   # ~3.2M tokens
DOCS_1K = DOCS_10K[:1000]


def test_4b_10k_matches_the_banked_run():
    p = plan_query(4, DOCS_10K, M4B, H100, selectivity=0.8)
    assert p.mode == "chain" and p.workers == 1 and p.access == "read"
    # banked: 49.9 seconds measured
    assert 42 <= p.predicted_makespan_s <= 58


def test_map_operators_by_corpus_size():
    """A classifier (gated=False) on a large corpus runs
    pipelined_map: rounds stay full, so forking buys nothing and
    costs per-sibling CPU (the underfilled-round cell). On a small
    corpus the same query runs hybrid_map, forking after the first
    prompt. The sequence cap multiplies by the filter count only
    when forks can exist."""
    big = plan_query(4, DOCS_10K, M4B, H100, selectivity=0.8,
                     gated=False)
    assert big.mode == "spec" and big.operator == "pipelined_map"
    assert big.spec_after_stage == 0
    small = plan_query(4, DOCS_10K[:60], M4B, H100, selectivity=0.8,
                       gated=False)
    assert small.operator == "hybrid_map"
    assert small.spec_after_stage == 1
    # 60 documents all admitted, forked: 60 x 4 live sequences + 16
    assert small.engine_max_seqs == 60 * 4 + 16


def test_policy_override_forces_either_operator():
    """policy forces the operator family for testing: "hybrid" turns
    the fork on even where the rule says never; "pipelined" turns it
    off even where the rule says switch. Semantics (filter or map)
    stay with `gated` - pipelined_map still answers everything, one
    prompt per round."""
    p = plan_query(4, DOCS_10K, M4B, H100, selectivity=0.8,
                   policy="hybrid")
    assert p.operator == "hybrid_filter" and p.spec_after_stage >= 1
    g = plan_query(6, DOCS_1K, M4B, H100, selectivity=0.5,
                   policy="pipelined")
    assert g.operator == "pipelined_filter" and g.spec_after_stage == 0
    m = plan_query(4, DOCS_10K, M4B, H100, gated=False,
                   policy="pipelined")
    assert m.operator == "pipelined_map"


def test_sequence_cap_robust_to_size_skew():
    """One tiny outlier document must not inflate the cap. The old
    bound divided the budget by the single smallest document (a
    10-token outlier would have claimed sixty thousand sequences);
    the exact bound is the longest ascending prefix that fits."""
    uniform = plan_query(4, [300] * 2000, M4B, H100, gated=False)
    skewed = plan_query(4, [10] + [300] * 1999, M4B, H100, gated=False)
    assert skewed.engine_max_seqs == uniform.engine_max_seqs
    assert skewed.engine_max_seqs == 2000 + 16


def test_switch_uses_per_filter_selectivities():
    """A survivor cliff sits where the selective filter sits; the
    per-filter list finds it, the mean smears it away entirely."""
    sels = [0.9, 0.9, 0.05, 0.9, 0.9, 0.9]
    p = plan_query(6, DOCS_1K, M4B, H100, selectivity=sels)
    # survivors 900, 810, then 40: forty documents' question work
    # underfills a round, so the switch lands right after the cliff
    assert p.spec_after_stage == 3
    m = plan_query(6, DOCS_1K, M4B, H100,
                   selectivity=sum(sels) / len(sels))
    assert m.spec_after_stage != 3


def test_step_budget_scales_with_pool_slack():
    """The step budget is derived, not a constant: the largest value
    whose activation reservation stays under STEP_POOL_FRACTION of
    the KV pool. A fat pool packs large steps; a thin pool keeps the
    floor rather than trade KV for under a percent of wall; the
    budget never sits below the sequence cap (the engine requires
    the step budget to cover it)."""
    fat = plan_query(4, DOCS_10K, M4B, H100, selectivity=0.8)
    assert 16_384 <= fat.engine_step_tokens <= 32_768
    mid = plan_query(4, DOCS_10K, M32B, H100, selectivity=0.8)
    assert 4_096 <= mid.engine_step_tokens < fat.engine_step_tokens
    thin = plan_query(4, [300] * 100, M32B, L40S, selectivity=0.8)
    assert thin.engine_step_tokens == 2_048
    # forks multiply live sequences; the step budget must cover them
    forked = plan_query(4, DOCS_10K[:60], M4B, H100, selectivity=0.8,
                        gated=False)
    assert forked.engine_step_tokens >= forked.engine_max_seqs


def test_generation_is_priced_and_scales_with_context():
    """A generative map's prediction carries decode time from the
    calibrated decode model, and the per-token price grows with
    document length (attention reads the whole context per generated
    token)."""
    base = plan_query(2, DOCS_1K, M4B, H100, gated=False)
    gen = plan_query(2, DOCS_1K, M4B, H100, gated=False,
                     gen_tokens=256)
    short_delta = gen.predicted_makespan_s - base.predicted_makespan_s
    assert short_delta > 5
    assert any("decode" in r for r in gen.remarks)
    long_docs = [30_000] * 20
    lbase = plan_query(2, long_docs, M4B, H100, gated=False)
    lgen = plan_query(2, long_docs, M4B, H100, gated=False,
                      gen_tokens=256)
    long_delta = lgen.predicted_makespan_s - lbase.predicted_makespan_s
    # per generated token, 30k-token contexts must price well above
    # ~350-token contexts (fewer pairs here, so compare per token)
    per_tok_short = short_delta / (1000 * 2 * 256)
    per_tok_long = long_delta / (20 * 2 * 256)
    assert per_tok_long > 5 * per_tok_short


def test_spill_plans_with_a_store_instead_of_refusing():
    """A pool too small for the working set is a refusal without a
    store, and a spill plan with one: the tiering store absorbs the
    overflow, priced pessimistically at store bandwidth both ways."""
    long_docs = [30_000] * 40 + [100_000] * 10
    r = plan_query(2, long_docs, M32B, L40S, gpus=2,
                   saturation_width_docs=4)
    assert isinstance(r, Refusal)
    assert r.constraint == "pool_under_working_set"
    disk = StoreSpec(read_bw=2.7e9)
    p = plan_query(2, long_docs, M32B, L40S, gpus=2,
                   saturation_width_docs=4, store=disk)
    assert p.access == "spill"
    assert any("spills the overflow" in x for x in p.remarks)
    # the spill traffic must be priced in, not free
    no_spill_shape = plan_query(2, long_docs, M32B, L40S, gpus=2,
                                saturation_width_docs=1, store=disk)
    assert p.predicted_makespan_s >= no_spill_shape.predicted_makespan_s


def test_hybrid_switch_stage_planned():
    """A selective gated chain on a mid-size corpus switches to
    speculation where survivors stop filling rounds; a huge corpus
    with permissive filters never switches."""
    p = plan_query(6, DOCS_1K, M4B, H100, selectivity=0.5)
    assert p.mode == "chain" and p.spec_after_stage == 3
    q = plan_query(4, DOCS_10K, M4B, H100, selectivity=0.95)
    assert q.spec_after_stage == 0


def test_spec_constrains_decisive_token_models():
    """A chattering model still plans a spec chain: sampling is
    constrained to the yes/no ids, so the stage window is one token
    and the remark says the contract changed."""
    p = plan_query(4, DOCS_1K, M32B, H100, selectivity=0.8,
                   one_token_answers=False, gated=False)
    assert p.mode == "spec"
    assert p.stage_token_window == 1
    assert any("constrains sampling" in r for r in p.remarks)


def test_32b_overflow_turns_pins_off():
    p = plan_query(4, DOCS_1K, M32B, H100, selectivity=0.8)
    assert p.mode == "chain"
    assert not p.pin
    assert any("overflow" in n for n in p.remarks)
    # banked: 32.0 seconds measured; the rate model runs about
    # fifteen percent under on this tier, so the band is loose
    assert 26 <= p.predicted_makespan_s <= 48


def test_store_breakeven_flips_between_tiers():
    disk = StoreSpec(read_bw=5.2e9, warm=True)
    p4 = plan_query(4, DOCS_10K, M4B, H100, store=disk, selectivity=0.8)
    p32 = plan_query(4, DOCS_1K, M32B, H100, store=disk, selectivity=0.8)
    assert p4.access == "read"          # banked: restore lost two to one
    assert p32.access == "restore"      # banked: 8s restore vs 30s read


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
    huge = dataclasses.replace(M32B, W_mem=200e9)
    p = plan_query(4, DOCS_1K, huge, H100, gpus=8, selectivity=0.8)
    assert p.tensor_parallel == 4 and p.workers == 2


def test_single_filter_uses_requests_without_pins():
    """One filter has no future consumer, so there is nothing a pin
    could buy: read, answer, free."""
    p = plan_query(1, DOCS_10K, M4B, H100)
    assert p.mode == "requests" and not p.pin


def test_budget_stays_under_the_pool():
    for model, docs in ((M4B, DOCS_10K), (M32B, DOCS_1K)):
        p = plan_query(4, docs, model, H100)
        free = H100.M * 0.92 - model.W_mem
        assert p.budget_tokens <= 0.8 * free / model.kappa + 1


# ------------------------------------------- the refusal path (Algorithm 2)

def test_refuses_when_weights_need_more_cards_than_exist():
    import dataclasses
    huge = dataclasses.replace(M32B, W_mem=200e9)      # needs a 4-card group
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
    # the 32B-on-L40S pool holds 81,329 tokens; 300 resident documents
    # of the working set (largest doc 399 + question 46) need 133,500
    r = plan_query(4, DOCS_1K, M32B, L40S, selectivity=0.8,
                   saturation_width_docs=300)
    assert isinstance(r, Refusal)
    assert r.constraint == "pool_under_working_set"
    assert r.unit == "tokens" and r.available < r.needed


def test_normal_configs_still_return_plans():
    for model, docs in ((M4B, DOCS_10K), (M32B, DOCS_1K)):
        assert isinstance(plan_query(4, docs, model, H100, selectivity=0.8),
                          Plan)


def test_e12_32b_on_one_l40s_is_planned_not_refused():
    # PAPER.md E12 expects Algorithm 2 line 8 to refuse this pair. The
    # honest arithmetic disagrees: 0.92 * 48 GB leaves 10.66 GB of pool
    # after the 33.5 GB of weights - 81,329 tokens of KV - and the
    # minimum working set at W* = 1 (largest doc 399 + question 46) is
    # 445 tokens. The gate reports what is true; E12's expected outcome
    # is what needs correcting, not the gate.
    p = plan_query(4, DOCS_1K, M32B, L40S, selectivity=0.8)
    assert isinstance(p, Plan)
    assert p.workers == 1 and p.tensor_parallel == 1
    assert p.budget_tokens == 65_063       # 80 percent of the 81,329 tokens
    assert any("overflow" in n for n in p.remarks)
