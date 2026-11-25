"""One question's draft policy, settled answer by answer, as a test oracle."""

import numpy as np

from quail.common.answers import prior_survival, survival_rows
from quail.common.draft_policy import (
    LIVE_BUCKETS,
    draft_lengths,
    live_buckets,
    source_odds,
)


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
