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

import numpy as np

from quail.common.answers import prior_survival, survival_rows

# drafted rounds the seed of the source odds counts as
SOURCE_PRIOR_WEIGHT = 8
# the source odds are learned per range of live positions, since a
# count can be anything from 2 to the document's length: a round with
# m live positions counts toward the range whose edge is nearest on a
# log scale
LIVE_BUCKETS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000)


def live_buckets(num_streams) -> np.ndarray:
    """Return each count of live positions' range, nearest on a log scale."""
    num_streams = np.asarray(num_streams)
    return np.abs(np.log1p(num_streams[..., None])
                  - np.log1p(LIVE_BUCKETS)).argmin(-1)


def source_odds(num_streams, trials, accepts) -> np.ndarray:
    """Return P(the position followed is the answer's source) for each round.

    The seed is 1 / live positions, no position preferred, blended by
    count with the drafted rounds seen in that range of live positions.

    Args:
        num_streams: How many document positions each answer so far fits.
        trials: Drafted rounds per range of live positions.
        accepts: Of those, the rounds whose first draft token was written.
    """
    num_streams = np.asarray(num_streams)
    buckets = live_buckets(num_streams)
    seeded = ((SOURCE_PRIOR_WEIGHT / np.array(LIVE_BUCKETS)[buckets]
               + np.asarray(accepts)[buckets])
              / (SOURCE_PRIOR_WEIGHT + np.asarray(trials)[buckets]))
    return np.where(num_streams <= 1, 1.0, seeded)


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
    written = np.asarray(written)
    columns = written[:, None] + np.arange(cap + 1)
    tail = np.asarray(survival)[np.minimum(columns, limit)]
    base = tail[:, :1]
    tail = np.divide(tail, base, out=np.ones_like(tail), where=base > 0)
    # the chosen token, then each draft token while the answer goes on
    # and the position followed is its source
    writes = 1 + source_odds(num_streams, trials, accepts)[:, None] * np.cumsum(
        np.c_[np.zeros(len(written)), tail[:, 1:]], axis=1)
    costs = (1 + np.arange(cap + 1) + np.asarray(rho, dtype=float)[:, None]) / writes
    costs[columns >= limit] = np.inf
    return costs.argmin(axis=1)


class DraftPolicy:
    """Chooses each round's draft length for one question.

    Args:
        cap: The most draft tokens a round may feed.
        max_answer_tokens: The answer limit; drafts stop before it.
        expected_answer_tokens: The planner's guess at the mean answer
            length, the prior of the answer survival curve.
    """

    def __init__(self, cap: int, max_answer_tokens: int, expected_answer_tokens: int):
        self.cap = cap
        self.max_answer_tokens = max_answer_tokens
        self.prior = prior_survival(expected_answer_tokens, max_answer_tokens)
        # settled answers by length
        self.lengths = np.zeros(max_answer_tokens + 1, dtype=np.int64)
        # drafted rounds and the ones whose first draft token was written,
        # by range of live positions
        self.trials = np.zeros(len(LIVE_BUCKETS), dtype=np.int64)
        self.accepts = np.zeros(len(LIVE_BUCKETS), dtype=np.int64)
        self._curve = None

    def curve(self) -> np.ndarray:
        """Return the answer survival curve of the settled answers."""
        if self._curve is None:
            self._curve = survival_rows(self.lengths[None], self.lengths[None],
                                        self.prior[None], [self.max_answer_tokens])[0]
        return self._curve

    def survival(self, written: int) -> np.ndarray:
        """Return P(the answer has more than j tokens left), j = 0.., given written."""
        tail = self.curve()[written:]
        return tail / tail[0] if tail[0] > 0 else np.ones_like(tail)

    def source_odds(self, num_streams: int) -> float:
        """Return P(the position followed is the answer's source) for one round."""
        return float(source_odds([num_streams], self.trials, self.accepts)[0])

    def draft_length(self, rho: float, written: int, num_streams: int) -> int:
        """Return the draft length for a round.

        Args:
            rho: The round's KV read, in fed tokens.
            written: The answer's tokens so far.
            num_streams: How many document positions the answer so far fits.
        """
        return int(draft_lengths(self.cap, self.max_answer_tokens, self.curve(),
                                 [written], [num_streams], [rho], self.trials,
                                 self.accepts)[0])

    def settle(self, length: int) -> None:
        """Record a settled answer's length."""
        self.lengths[min(length, self.max_answer_tokens)] += 1
        self._curve = None

    def observe(self, num_streams: int, accepted: bool) -> None:
        """Record whether a drafted round's first draft token was written.

        Args:
            num_streams: How many document positions the answer fit when drafting.
            accepted: Whether the model chose the first draft token.
        """
        b = int(live_buckets(num_streams))
        self.trials[b] += 1
        self.accepts[b] += accepted
