"""The draft length of each extraction round, chosen from its cost and odds.

A round feeds the chosen token and a draft of the document's next
tokens along one of the positions the answer so far fits, and pays one
read of the document's KV however long the draft is. A draft token is
written when the answer still has a token to go and the position
followed is the answer's source. The first is the field's answer
survival curve (quail.common.answers); the second is certain with one
live position and is learned per count of live positions from the
rounds that drafted. The policy picks the draft length that costs the
least per written token.
"""

import _quail_native
import numpy as np

from quail.common.answers import _array, _float64, _int64

# ranges of live positions the source odds are learned per
LIVE_BUCKETS = tuple(_quail_native.LIVE_BUCKETS)


def live_buckets(num_streams) -> np.ndarray:
    """Return each count of live positions' range, nearest on a log scale."""
    counts = _int64(num_streams)
    found = _array(_quail_native.live_buckets(counts.ravel()), np.int64)
    return found.reshape(counts.shape)


def source_odds(num_streams, trials, accepts) -> np.ndarray:
    """Return P(the position followed is the answer's source) for each round.

    Args:
        num_streams: How many document positions each answer so far fits.
        trials: Drafted rounds per range of live positions.
        accepts: Of those, the rounds whose first draft token was written.
    """
    return _array(_quail_native.source_odds(
        _int64(num_streams), _int64(trials), _int64(accepts)), np.float64)


def draft_lengths(cap, limit, survival, written, num_streams, rho, trials,
                  accepts) -> np.ndarray:
    """Return each answer's draft length: the least cost per written token.

    Args:
        cap: The most draft tokens a round may feed.
        limit: The answer token limit; drafts stop before it.
        survival: The field's S(t), t = 0..limit.
        written: Each answer's tokens so far.
        num_streams: How many document positions each answer so far fits.
        rho: Each round's KV read, in fed tokens.
        trials: Drafted rounds per range of live positions.
        accepts: Of those, the rounds whose first draft token was written.
    """
    return _array(_quail_native.draft_lengths(
        int(cap), int(limit), _float64(survival), _int64(written),
        _int64(num_streams), _float64(rho), _int64(trials), _int64(accepts)),
        np.int64)


def draft_gains(cap, limit, survival, written, num_streams, trials,
                accepts) -> np.ndarray:
    """Return each answer's expected written tokens from each draft token.

    Column j - 1 is draft token j's: the chance the answer goes on past
    j more tokens times the chance the position followed is its source.
    Draft tokens at or past the limit write none.
    """
    written = _int64(written)
    found = _quail_native.draft_gains(
        int(cap), int(limit), _float64(survival), written, _int64(num_streams),
        _int64(trials), _int64(accepts))
    return _array(found, np.float64).reshape(len(written), int(cap))


def fit_drafts(gains, drafts, room) -> np.ndarray:
    """Return the draft lengths that keep the most expected writes in room rows.

    Args:
        gains: Each answer's expected writes per draft token (draft_gains).
        drafts: Each answer's draft length before the fit.
        room: The draft rows the pass can hold.
    """
    gains = _float64(gains)
    return _array(_quail_native.fit_drafts(
        gains.ravel(), gains.shape[1], _int64(drafts), int(room)), np.int64)
