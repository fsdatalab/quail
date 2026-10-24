"""The map draft policy and the n-gram cache it drafts from."""

from quail.cost.draft_policy import MapDraftPolicy
from quail.cost.ngram_cache import MAX_CONTINUATIONS, NgramCache, NgramMatch


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
    (key,) = [k for k in cache.entries if k == (1, 2)]
    assert len(cache.entries[key]) == MAX_CONTINUATIONS


def test_hit_rates_blend_the_count_share_with_observed_accepts():
    policy = MapDraftPolicy(cap=15, max_answer_tokens=64,
                            expected_answer_tokens=16)
    match = NgramMatch(token=5, order=3, fan_out=2, share=0.8, own=True)
    seed = policy.hit_rate(match)
    assert seed == 0.8
    cell = policy.cell(match)
    for _ in range(100):
        policy.observe(cell, False)
    assert policy.hit_rate(match) < 0.1


def test_chained_drafts_shrink_with_the_hit_rate():
    policy = MapDraftPolicy(cap=15, max_answer_tokens=64,
                            expected_answer_tokens=16)
    sure = policy.chain_length(5.0, 1, [0.95] * 15, slack=0.0)
    doubtful = policy.chain_length(5.0, 1, [0.3] * 15, slack=0.0)
    assert doubtful < sure


def test_slack_lengthens_the_draft():
    # below the ridge a marginal draft token is free, so even weak
    # guesses are worth feeding
    policy = MapDraftPolicy(cap=15, max_answer_tokens=64,
                            expected_answer_tokens=16)
    busy = policy.chain_length(5.0, 1, [0.5] * 15, slack=0.0)
    idle = policy.chain_length(5.0, 1, [0.5] * 15, slack=15.0)
    assert idle > busy
    assert policy.copy_length(5.0, 1, slack=15.0) >= policy.copy_length(
        5.0, 1, slack=0.0)
