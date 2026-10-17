"""Tests for the adaptive draft-length policy."""

from quail.cost.draft_policy import DraftPolicy


def test_draft_length_grows_with_the_kv_read_and_shrinks_with_live_positions():
    policy = DraftPolicy(cap=63, max_answer_tokens=64, expected_answer_tokens=16)
    one_match = [
        policy.draft_length(rho, 1, num_streams=1) for rho in (0, 5, 50, 500)
    ]
    assert one_match == sorted(one_match) and one_match[0] < one_match[-1]
    by_streams = [
        policy.draft_length(100, 1, num_streams) for num_streams in (1, 2, 10, 100)
    ]
    assert by_streams == sorted(by_streams, reverse=True)
    assert by_streams[-1] < by_streams[0]
    # a cheap round does not pay for a bet on one of many positions
    assert policy.draft_length(0, 1, num_streams=100) == 0


def test_draft_length_respects_the_cap_and_the_answer_limit():
    policy = DraftPolicy(cap=16, max_answer_tokens=64, expected_answer_tokens=1000)
    assert policy.draft_length(1000, 1, num_streams=1) == 16
    assert policy.draft_length(1000, 60, num_streams=1) == 3
    assert policy.draft_length(1000, 63, num_streams=1) == 0


def test_settled_answers_replace_the_planner_guess():
    policy = DraftPolicy(cap=63, max_answer_tokens=64, expected_answer_tokens=16)
    before = policy.draft_length(5, 1, num_streams=1)
    for _ in range(64):
        policy.settle(3)        # every answer is three tokens
    short = policy.draft_length(5, 1, num_streams=1)
    assert short < before
    policy = DraftPolicy(cap=63, max_answer_tokens=64, expected_answer_tokens=16)
    for _ in range(64):
        policy.settle(64)       # every answer runs to the cap
    assert policy.draft_length(5, 1, num_streams=1) > before
    assert policy.survival(0)[0] == 1.0


def test_drafted_rounds_replace_the_even_odds_seed():
    policy = DraftPolicy(cap=63, max_answer_tokens=64, expected_answer_tokens=16)
    assert policy.source_odds(1) == 1.0
    assert policy.source_odds(2) == 0.5
    seeded = policy.draft_length(50, 1, num_streams=2)
    for _ in range(64):
        policy.observe(2, True)     # the first position is always the source
    assert policy.source_odds(2) > 0.85
    assert policy.draft_length(50, 1, num_streams=2) > seeded
    policy = DraftPolicy(cap=63, max_answer_tokens=64, expected_answer_tokens=16)
    for _ in range(64):
        policy.observe(2, False)
    assert policy.source_odds(2) < 0.1
    assert policy.draft_length(50, 1, num_streams=2) < seeded
