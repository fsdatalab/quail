"""The draft length of each extraction round, chosen from its cost and odds.

A round feeds the chosen token and a draft of the document's next
tokens, and pays one read of the document's KV however long the draft
is. With one live stream a draft token is written when the answer
still has a token to go, so the expected tokens a round writes follow
the answer-length curve, and the policy picks the draft length that
costs the least per written token. With several live streams the
draft may follow the wrong one, so the round feeds the chosen token
alone and the model's next token narrows the streams.
"""

import numpy as np

# how many settled answers the planner's answer-length guess weighs
PRIOR_WEIGHT = 8
# the round's KV read in tokens, rounded to one of these for the tables
RHO_BUCKETS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


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
        self._tables = {}

    def survival(self, written: int) -> np.ndarray:
        """Return P(the answer has more than j tokens left), j = 0.., given written.

        The planner's geometric guess and the settled answers' lengths
        are blended by count, so the guess fades as answers settle.
        """
        n = self.max_answer_tokens
        j = np.arange(n + 1)
        prior = PRIOR_WEIGHT * self.q ** j
        # answers longer than j, from the histogram of settled lengths
        longer = self.settled - np.cumsum(self.lengths)
        curve = (prior + longer) / (PRIOR_WEIGHT + self.settled)
        tail = curve[written:]
        return tail / tail[0] if tail[0] > 0 else np.ones_like(tail)

    def draft_length(self, rho: float, written: int, ambiguous: bool) -> int:
        """Return the draft length for a round.

        Args:
            rho: The round's KV read, in fed tokens.
            written: The answer's tokens so far.
            ambiguous: Whether more than one live stream fits the answer;
                such a round feeds no draft.
        """
        room = min(self.cap, self.max_answer_tokens - written - 1)
        if ambiguous or room <= 0:
            return 0
        bucket = int(np.argmin([abs(np.log1p(rho) - np.log1p(b)) for b in RHO_BUCKETS]))
        key = (bucket, written, room)
        if key not in self._tables:
            self._tables[key] = self._best(RHO_BUCKETS[bucket], written, room)
        return self._tables[key]

    def _best(self, rho: float, written: int, room: int) -> int:
        gammas = np.arange(room + 1)
        writes = np.cumsum(self.survival(written)[:room + 1])
        cost = (1 + gammas + rho) / writes
        return int(gammas[int(np.argmin(cost))])

    def settle(self, length: int) -> None:
        """Record a settled answer's length; the curve picks it up at powers of two."""
        self.lengths[min(length, self.max_answer_tokens)] += 1
        self.settled += 1
        if self.settled & (self.settled - 1) == 0:
            self._tables.clear()
