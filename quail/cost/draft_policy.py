"""The draft length of each extraction round, chosen from its cost and odds.

A round feeds the chosen token and a draft of the document's next
tokens, and pays one read of the document's KV however long the draft
is. A draft token is written when the answer still has a token to go
and the position the draft follows is the answer's source. The first
is the answer-length curve, learned from the answers that settle; the
second the decoder reads off the live positions: certain while they
agree on the next token, and the share that continue the drafted way
past each point where they split. The policy picks the draft length
that costs the least per written token.
"""

import numpy as np

# how many observations the seed of the answer-length curve weighs
PRIOR_WEIGHT = 8


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

    def draft_length(self, rho: float, written: int, agreement=None) -> int:
        """Return the draft length for a round.

        Args:
            rho: The round's KV read, in fed tokens.
            written: The answer's tokens so far.
            agreement: For each draft token in order, the odds that the
                position the draft follows is the answer's source: 1
                while every live position agrees, and the share of
                them that continue the drafted way past each split.
                None means one live position. The draft is no longer
                than this sequence.
        """
        room = min(self.cap, self.max_answer_tokens - written - 1)
        if agreement is not None:
            room = min(room, len(agreement))
        if room <= 0:
            return 0
        gammas = np.arange(room + 1)
        survival = self.survival(written)[:room + 1]
        odds = (np.ones(room) if agreement is None
                else np.asarray(agreement[:room], dtype=np.float64))
        # the chosen token, then each draft token while the answer goes
        # on and the position followed is its source
        writes = survival[0] + np.cumsum(
            np.concatenate(([0.0], survival[1:] * odds)))
        return int(gammas[int(np.argmin((1 + gammas + rho) / writes))])

    def settle(self, length: int) -> None:
        """Record a settled answer's length."""
        self.lengths[min(length, self.max_answer_tokens)] += 1
        self.settled += 1
        self._curve = None
