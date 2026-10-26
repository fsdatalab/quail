"""The map draft policy and the n-gram cache it drafts from."""

import numpy as np

from quail.cost.draft_policy import MapDraftPolicy
from quail.cost.ngram_cache import MAX_CONTINUATIONS, NgramCache


def test_the_cache_prefers_the_longest_match():
    cache = NgramCache()
    cache.add([1, 2, 3, 4, 5], document=0)
    short = cache.lookup([9, 9, 3, 4], document=0)
    assert short.token == 5 and short.order == 2
    cache.add([7, 2, 3, 4, 6], document=1)
    longer = cache.lookup([7, 2, 3, 4], document=1)
    assert longer.token == 6 and longer.order == 4


def test_the_cache_counts_give_the_share_and_fan_out():
    cache = NgramCache()
    for _ in range(3):
        cache.add([1, 2, 5], document=0)
    cache.add([1, 2, 6], document=1)
    match = cache.lookup([1, 2], document=1)
    assert match.token == 5
    assert match.fan_out == 2
    assert match.share == 0.75
    # document 0 last inserted the winning continuation
    assert not match.own
    assert cache.lookup([1, 2], document=0).own


def test_the_cache_caps_continuations_per_key():
    cache = NgramCache()
    for token in range(10):
        cache.add([1, 2, token + 100], document=0)
    assert len(cache.continuations((1, 2))) == MAX_CONTINUATIONS


def _length(policy, rho, written, rate, slack):
    """One round's draft length when every hop has the same odds."""
    room = min(policy.cap, policy.max_answer_tokens - written - 1)
    return int(policy.chain_lengths(
        np.array([rho]), np.array([written]), np.full((1, policy.cap), rate),
        slack, np.array([room]))[0])


def test_hit_rates_blend_the_count_share_with_observed_accepts():
    policy = MapDraftPolicy(cap=15, max_answer_tokens=64,
                            expected_answer_tokens=16)
    match = dict(order=np.array([3]), fan_out=np.array([2]),
                 own=np.array([True]), share=np.array([0.8]))
    cells, seed = policy.odds_many(**match)
    assert seed[0] == 0.8
    policy.trials[cells[0]] += 100
    assert policy.odds_many(**match)[1][0] < 0.1


def test_chained_drafts_shrink_with_the_hit_rate():
    policy = MapDraftPolicy(cap=15, max_answer_tokens=64,
                            expected_answer_tokens=16)
    sure = _length(policy, 5.0, 1, 0.95, slack=3.0)
    doubtful = _length(policy, 5.0, 1, 0.3, slack=3.0)
    assert doubtful < sure


def test_slack_lengthens_the_draft():
    # below the ridge a marginal draft token is free, so even weak
    # guesses are worth feeding
    policy = MapDraftPolicy(cap=15, max_answer_tokens=64,
                            expected_answer_tokens=16)
    busy = _length(policy, 5.0, 1, 0.5, slack=0.0)
    idle = _length(policy, 5.0, 1, 0.5, slack=15.0)
    assert idle > busy


def test_copy_misses_shrink_the_copied_draft():
    policy = MapDraftPolicy(cap=15, max_answer_tokens=256,
                            expected_answer_tokens=64)
    seeded = _length(policy, 17.0, 10, policy.copy_rate(), slack=0.0)
    policy.copy_trials += 100
    assert policy.copy_rate() < 0.1
    assert _length(policy, 17.0, 10, policy.copy_rate(), slack=0.0) <= 1
    assert seeded > 1


def test_the_kv_read_is_what_a_draft_amortizes():
    # a fed token costs a token and writes at most one, so a draft pays
    # only against the round's price, the KV read every round pays:
    # with no read to save and no slack, no odds justify a draft
    policy = MapDraftPolicy(cap=15, max_answer_tokens=256,
                            expected_answer_tokens=64)
    assert _length(policy, 0.0, 10, 0.95, slack=0.0) == 0
    assert _length(policy, 3.6, 10, 0.5, slack=0.0) >= 1
    assert _length(policy, 3.6, 10, 0.1, slack=0.0) == 0
