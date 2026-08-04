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
