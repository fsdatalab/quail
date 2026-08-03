"""The planner must reproduce every banked measured winner."""

import numpy as np

from docengine.configs import DEVICES, MODELS
from docengine.planner import StoreSpec, plan_query

H100 = DEVICES["H100-SXM-80GB"]
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
    assert any("overflow" in n for n in p.notes)
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
