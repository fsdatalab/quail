"""The draft length of each map round, chosen from its cost and odds.

A round feeds the chosen token and a draft, and pays one read of the
document's KV however long the draft is. A draft copied from the
document is all-or-nothing: once the model is copying, the whole run
tends to write. A draft chained from the n-gram cache fails per hop,
so its expected writes shrink with each hop's acceptance odds, learned
per match length, fan-out range, and source. A fed token's price
follows the forward pass: while the pass is below the compute-memory
ridge, a marginal draft token is close to free; above it, it costs one
token of compute. The policy picks the draft length that costs the
least per written token.
"""

import bisect
import math

import numpy as np

from quail.cost.draft_policy import PRIOR_WEIGHT, AnswerLengthCurve
from quail.cost.ngram_cache import MATCH_ORDERS, NgramMatch

# the acceptance odds are learned per range of distinct continuations
FAN_BUCKETS = (1, 2, 4, 8)
_FAN_LOGS = [math.log1p(b) for b in FAN_BUCKETS]
_SOURCES = 2    # the drafting document's own output, or a sibling's


def _fan_bucket(fan_out: int) -> int:
    """Return the index of the fan-out range nearest on a log scale."""
    x = math.log1p(fan_out)
    i = bisect.bisect_left(_FAN_LOGS, x)
    if i == 0 or i == len(_FAN_LOGS):
        return min(i, len(_FAN_LOGS) - 1)
    return i if _FAN_LOGS[i] - x < x - _FAN_LOGS[i - 1] else i - 1


class MapDraftPolicy:
    """Chooses each round's draft length for one instruction.

    Args:
        cap: The most draft tokens a round may feed.
        max_answer_tokens: The answer limit; drafts stop before it.
        expected_answer_tokens: The planner's guess at the mean answer
            length, the seed of the answer-length curve until answers
            settle.
    """

    def __init__(self, cap: int, max_answer_tokens: int,
                 expected_answer_tokens: int):
        self.cap = cap
        self.max_answer_tokens = max_answer_tokens
        self.curve = AnswerLengthCurve(max_answer_tokens, expected_answer_tokens)
        cells = len(MATCH_ORDERS) * len(FAN_BUCKETS) * _SOURCES
        self.trials = np.zeros(cells, dtype=np.int64)
        self.accepts = np.zeros(cells, dtype=np.int64)

    def cell(self, match: NgramMatch) -> int:
        """Return the acceptance table's cell for one cache match."""
        order = MATCH_ORDERS.index(match.order)
        return ((order * len(FAN_BUCKETS) + _fan_bucket(match.fan_out)) * _SOURCES
                + int(match.own))

    def hit_rate(self, match: NgramMatch) -> float:
        """Return P(the model writes the match's token).

        The seed is the continuation's count share in the cache,
        blended by count with the drafts verified at the same cell.
        """
        c = self.cell(match)
        seed = PRIOR_WEIGHT * match.share
        return (seed + self.accepts[c]) / (PRIOR_WEIGHT + self.trials[c])

    def _room(self, written: int) -> int:
        return min(self.cap, self.max_answer_tokens - written - 1)

    def _costs(self, length: int, rho: float, slack: float) -> np.ndarray:
        """Return each draft length's round cost, in fed tokens of compute.

        The first slack draft tokens ride a pass below the ridge, so
        they cost nothing; each one past the slack costs one token.
        """
        gammas = np.arange(length + 1)
        return rho + 1 + np.maximum(0.0, gammas - slack)

    def copy_length(self, rho: float, written: int, slack: float) -> int:
        """Return the draft length for a round copying from the document.

        A copied draft writes while the answer goes on: the model is
        either copying this run or it is not, and the odds it is are
        carried by the answer-length curve alone.

        Args:
            rho: The round's KV read, in fed tokens.
            written: The answer's tokens so far.
            slack: Fed tokens the current passes carry for free.
        """
        room = self._room(written)
        if room <= 0:
            return 0
        survival = self.curve.survival(written)[:room + 1]
        writes = survival[0] + np.cumsum(np.concatenate(([0.0], survival[1:])))
        return int(np.argmin(self._costs(room, rho, slack) / writes))

    def chain_length(self, rho: float, written: int, rates, slack: float) -> int:
        """Return how many of a chained draft's tokens to feed.

        Args:
            rho: The round's KV read, in fed tokens.
            written: The answer's tokens so far.
            rates: Each hop's acceptance odds, in hop order; its length
                bounds the draft.
            slack: Fed tokens the current passes carry for free.
        """
        room = min(self._room(written), len(rates))
        if room <= 0:
            return 0
        survival = self.curve.survival(written)[:room + 1]
        # hop j writes when every hop before it wrote and the answer
        # goes on
        chained = np.cumprod(np.asarray(rates[:room], dtype=np.float64))
        writes = survival[0] + np.cumsum(
            np.concatenate(([0.0], chained * survival[1:])))
        return int(np.argmin(self._costs(room, rho, slack) / writes))

    def observe(self, match_cell: int, accepted: bool) -> None:
        """Record whether a chained draft token was written.

        Args:
            match_cell: The cell of the match that proposed the token.
            accepted: Whether the model chose the token.
        """
        self.trials[match_cell] += 1
        self.accepts[match_cell] += accepted
