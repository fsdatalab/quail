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






