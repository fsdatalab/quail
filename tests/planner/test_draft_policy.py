"""Tests for the adaptive draft-length policy."""

from quail.cost.draft_policy import DraftPolicy


def test_draft_length_grows_with_the_kv_read_and_is_zero_when_ambiguous():
    policy = DraftPolicy(cap=63, max_answer_tokens=64, expected_answer_tokens=16)
    one_match = [policy.draft_length(rho, 1, ambiguous=False)
                 for rho in (0, 5, 50, 500)]
    assert one_match == sorted(one_match) and one_match[0] < one_match[-1]
    assert all(policy.draft_length(rho, 1, ambiguous=True) == 0
               for rho in (0, 5, 50, 500))


def test_draft_length_respects_the_cap_and_the_answer_limit():
    policy = DraftPolicy(cap=16, max_answer_tokens=64, expected_answer_tokens=1000)
    assert policy.draft_length(1000, 1, ambiguous=False) == 16
    assert policy.draft_length(1000, 60, ambiguous=False) == 3
    assert policy.draft_length(1000, 63, ambiguous=False) == 0


def test_settled_answers_replace_the_planner_guess():
    policy = DraftPolicy(cap=63, max_answer_tokens=64, expected_answer_tokens=16)
    before = policy.draft_length(5, 1, ambiguous=False)
    for _ in range(64):
        policy.settle(3)        # every answer is three tokens
    short = policy.draft_length(5, 1, ambiguous=False)
    assert short < before
    policy = DraftPolicy(cap=63, max_answer_tokens=64, expected_answer_tokens=16)
    for _ in range(64):
        policy.settle(64)       # every answer runs to the cap
    assert policy.draft_length(5, 1, ambiguous=False) > before
    assert policy.survival(0)[0] == 1.0
