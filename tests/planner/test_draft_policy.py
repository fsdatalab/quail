"""Tests for the adaptive draft-length policy."""

from quail.cost.draft_policy import DraftPolicy


def test_draft_length_grows_with_the_kv_read():
    policy = DraftPolicy(cap=63, max_answer_tokens=64, expected_answer_tokens=16)
    one_match = [policy.draft_length(rho, 1) for rho in (0, 5, 50, 500)]
    assert one_match == sorted(one_match) and one_match[0] < one_match[-1]


def test_a_draft_stops_where_the_positions_split_unless_the_kv_read_is_large():
    policy = DraftPolicy(cap=63, max_answer_tokens=64, expected_answer_tokens=16)
    # the positions agree on two tokens, then one of three continues
    split = [1.0, 1.0] + [1 / 3] * 40
    assert policy.draft_length(5, 1, split) == 2
    assert policy.draft_length(5, 1) > 2
    # a long document's round pays for the bet on one of three
    assert policy.draft_length(500, 1, split) > 2
    # a cheap round does not pay for a bet on one of many positions
    assert policy.draft_length(0, 1, [0.01] * 40) == 0
    # the draft is no longer than the positions' text
    assert policy.draft_length(500, 1, [1.0, 1.0, 1.0]) == 3


def test_draft_length_respects_the_cap_and_the_answer_limit():
    policy = DraftPolicy(cap=16, max_answer_tokens=64, expected_answer_tokens=1000)
    assert policy.draft_length(1000, 1) == 16
    assert policy.draft_length(1000, 60) == 3
    assert policy.draft_length(1000, 63) == 0


def test_settled_answers_replace_the_planner_guess():
    policy = DraftPolicy(cap=63, max_answer_tokens=64, expected_answer_tokens=16)
    before = policy.draft_length(5, 1)
    for _ in range(64):
        policy.settle(3)        # every answer is three tokens
    short = policy.draft_length(5, 1)
    assert short < before
    policy = DraftPolicy(cap=63, max_answer_tokens=64, expected_answer_tokens=16)
    for _ in range(64):
        policy.settle(64)       # every answer runs to the cap
    assert policy.draft_length(5, 1) > before
    assert policy.survival(0)[0] == 1.0
