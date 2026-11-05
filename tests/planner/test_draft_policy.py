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


def test_fit_drafts_keeps_the_largest_gains_in_the_room():
    import numpy as np

    from quail.cost.draft_policy import fit_drafts

    gains = np.array([[0.9, 0.8, 0.1, 0.1],
                      [0.5, 0.5, 0.5, 0.0],
                      [0.95, 0.2, 0.2, 0.2]])
    drafts = np.array([4, 3, 4])
    assert fit_drafts(gains, drafts, 11).tolist() == drafts.tolist()
    assert fit_drafts(gains, drafts, 5).tolist() == [2, 2, 1]
    # ties go to the earlier answers, and the room is filled exactly
    assert fit_drafts(gains, drafts, 4).tolist() == [2, 1, 1]
    assert fit_drafts(gains, drafts, 0).tolist() == [0, 0, 0]
    # gains of 1 / j shrink drafts evenly, as fixed drafts are
    even = np.tile(1.0 / np.arange(1, 9), (3, 1))
    assert fit_drafts(even, np.array([8, 8, 2]), 10).tolist() == [4, 4, 2]


def test_draft_gains_match_the_policy_costs():
    import numpy as np

    from quail.cost.draft_policy import draft_gains

    policy = DraftPolicy(cap=8, max_answer_tokens=10, expected_answer_tokens=4)
    gains = draft_gains(8, 10, policy.curve(), [0, 5], [1, 3],
                        policy.trials, policy.accepts)
    assert gains.shape == (2, 8)
    assert np.all(np.diff(gains, axis=1) <= 0)
    # one live position: the source is certain, so the gain is S(j)
    assert np.allclose(gains[0], policy.survival(0)[1:9])
    # draft tokens at or past the limit write nothing
    assert np.all(gains[1, 4:] == 0)

