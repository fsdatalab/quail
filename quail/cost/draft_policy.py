"""The draft length of each extraction round, chosen from its cost and odds.

A round feeds the chosen token and a draft of the document's next
tokens along one of the positions the answer so far fits, and pays one
read of the document's KV however long the draft is. A draft token is
written when the answer still has a token to go and the position
followed is the answer's source. The first is the answer-length curve,
learned from the answers that settle; the second is certain with one
live position and is learned per count of live positions from the
rounds that drafted. The policy picks the draft length that costs the
least per written token.
"""

import bisect
import math

import numpy as np

# how many observations the seed of each curve weighs
PRIOR_WEIGHT = 8
# the source odds are learned per range of live positions, since a
# count can be anything from 2 to the document's length: a round with
# m live positions counts toward the range whose edge is nearest on a
# log scale
LIVE_BUCKETS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000)
_LIVE_LOGS = [math.log1p(b) for b in LIVE_BUCKETS]


def _bucket(num_streams: int) -> int:
    """Return the index of the live-position range nearest live on a log scale."""
    x = math.log1p(num_streams)
    i = bisect.bisect_left(_LIVE_LOGS, x)
    if i == 0 or i == len(_LIVE_LOGS):
        return min(i, len(_LIVE_LOGS) - 1)
    return i if _LIVE_LOGS[i] - x < x - _LIVE_LOGS[i - 1] else i - 1


class DraftPolicy:
    """Chooses each round's draft length for one question.

    Args:
        cap: The most draft tokens a round may feed.
        max_answer_tokens: The answer limit; drafts stop before it.
        expected_answer_tokens: The planner's guess at the mean answer
            length, the seed of the answer-length curve until answers
            settle.
    """

    def __init__(self, cap: int, max_answer_tokens: int, expected_answer_tokens: int):
        self.cap = cap
        self.max_answer_tokens = max_answer_tokens
        self.q = 1.0 - 1.0 / max(1, expected_answer_tokens)
        self.lengths = np.zeros(max_answer_tokens + 1, dtype=np.int64)
        self.settled = 0
        # drafted rounds and the ones whose first draft token was written,
        # by range of live positions
        self.trials = np.zeros(len(LIVE_BUCKETS), dtype=np.int64)
        self.accepts = np.zeros(len(LIVE_BUCKETS), dtype=np.int64)
        self._curve = None

    def survival(self, written: int) -> np.ndarray:
        """Return P(the answer has more than j tokens left), j = 0.., given written.

        The planner's geometric guess and the settled answers' lengths
        are blended by count, so the guess fades as answers settle.
        """
        if self._curve is None:
            j = np.arange(self.max_answer_tokens + 1)
            prior = PRIOR_WEIGHT * self.q ** j
            # answers longer than j, from the histogram of settled lengths
            longer = self.settled - np.cumsum(self.lengths)
            self._curve = (prior + longer) / (PRIOR_WEIGHT + self.settled)
        tail = self._curve[written:]
        return tail / tail[0] if tail[0] > 0 else np.ones_like(tail)

    def source_odds(self, num_streams: int) -> float:
        """Return P(the position followed is the answer's source) given live positions.

        The seed is 1 / num_streams, no position preferred, blended by count
        with the drafted rounds seen at that many positions.
        """
        if num_streams <= 1:
            return 1.0
        b = _bucket(num_streams)
        seed = PRIOR_WEIGHT / LIVE_BUCKETS[b]
        return (seed + self.accepts[b]) / (PRIOR_WEIGHT + self.trials[b])

    def draft_length(self, rho: float, written: int, num_streams: int) -> int:
        """Return the draft length for a round.

        Args:
            rho: The round's KV read, in fed tokens.
            written: The answer's tokens so far.
            num_streams: How many document positions the answer so far fits.
        """
        room = min(self.cap, self.max_answer_tokens - written - 1)
        if room <= 0:
            return 0
        gammas = np.arange(room + 1)
        survival = self.survival(written)[:room + 1]
        # the chosen token, then each draft token while the answer goes
        # on and the position followed is its source
        writes = survival[0] + self.source_odds(num_streams) * np.cumsum(
            np.concatenate(([0.0], survival[1:])))
        return int(gammas[int(np.argmin((1 + gammas + rho) / writes))])

    def settle(self, length: int) -> None:
        """Record a settled answer's length."""
        self.lengths[min(length, self.max_answer_tokens)] += 1
        self.settled += 1
        self._curve = None

    def observe(self, num_streams: int, accepted: bool) -> None:
        """Record whether a drafted round's first draft token was written.

        Args:
            num_streams: How many document positions the answer fit when drafting.
            accepted: Whether the model chose the first draft token.
        """
        b = _bucket(num_streams)
        self.trials[b] += 1
        self.accepts[b] += accepted
