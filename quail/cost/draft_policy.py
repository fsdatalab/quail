"""The draft length of each decode round, chosen from its cost and odds.

A round feeds the chosen token and a draft, and pays one read of the
document's KV however long the draft is; each policy picks the draft
length that costs the least per written token, under the answer-length
curve learned from the answers that settle. An extraction's draft
follows one of the document positions the answer so far fits, and its
odds of writing are learned per count of live positions. A map's draft
is copied from the document or chained from the n-gram cache, with the
chain's per-hop odds learned per match length, fan-out range, and
source, and a fed token priced by the forward pass: close to free
below the compute-memory ridge, one token of compute above it.
"""

import bisect
import math

import numpy as np

from quail.cost.ngram_cache import MATCH_ORDERS, NgramMatch

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


class AnswerLengthCurve:
    """P(the answer has more than j tokens left), learned from settled answers.

    The planner's geometric guess and the settled answers' lengths are
    blended by count, so the guess fades as answers settle.

    Args:
        max_answer_tokens: The answer limit.
        expected_answer_tokens: The planner's guess at the mean answer
            length, the seed of the curve until answers settle.
    """

    def __init__(self, max_answer_tokens: int, expected_answer_tokens: int):
        self.max_answer_tokens = max_answer_tokens
        self.q = 1.0 - 1.0 / max(1, expected_answer_tokens)
        self.lengths = np.zeros(max_answer_tokens + 1, dtype=np.int64)
        self.settled = 0
        self._curve = None
        self._levels = None

    def survival(self, written: int) -> np.ndarray:
        """Return P(the answer has more than j tokens left), j = 0.., given written."""
        if self._curve is None:
            j = np.arange(self.max_answer_tokens + 1)
            prior = PRIOR_WEIGHT * self.q ** j
            # answers longer than j, from the histogram of settled lengths
            longer = self.settled - np.cumsum(self.lengths)
            self._curve = (prior + longer) / (PRIOR_WEIGHT + self.settled)
        tail = self._curve[written:]
        return tail / tail[0] if tail[0] > 0 else np.ones_like(tail)

    def levels(self) -> list:
        """Return the whole unnormalized curve as a list, for scalar reads.

        levels()[written + j] / levels()[written] is survival(written)[j];
        a draft-length loop reads a few scalars per round, so a list
        beats slicing and renormalizing an array each time.
        """
        if self._levels is None:
            self.survival(0)
            self._levels = self._curve.tolist()
        return self._levels

    def settle(self, length: int) -> None:
        """Record a settled answer's length."""
        self.lengths[min(length, self.max_answer_tokens)] += 1
        self.settled += 1
        self._curve = None
        self._levels = None


class ExtractDraftPolicy:
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
        self.curve = AnswerLengthCurve(max_answer_tokens, expected_answer_tokens)
        # drafted rounds and the ones whose first draft token was written,
        # by range of live positions
        self.trials = np.zeros(len(LIVE_BUCKETS), dtype=np.int64)
        self.accepts = np.zeros(len(LIVE_BUCKETS), dtype=np.int64)


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
        survival = self.curve.survival(written)[:room + 1]
        # the chosen token, then each draft token while the answer goes
        # on and the position followed is its source
        writes = survival[0] + self.source_odds(num_streams) * np.cumsum(
            np.concatenate(([0.0], survival[1:])))
        return int(gammas[int(np.argmin((1 + gammas + rho) / writes))])

    def settle(self, length: int) -> None:
        """Record a settled answer's length."""
        self.curve.settle(length)

    def observe(self, num_streams: int, accepted: bool) -> None:
        """Record whether a drafted round's first draft token was written.

        Args:
            num_streams: How many document positions the answer fit when drafting.
            accepted: Whether the model chose the first draft token.
        """
        b = _bucket(num_streams)
        self.trials[b] += 1
        self.accepts[b] += accepted


# a round's own cost when the pass is compute bound, in fed tokens:
# the KV read rides other rows' compute there, so saving a round saves
# only its dispatch and readout, not the read rho prices
BUSY_ROUND_TOKENS = 4.0
# the seed of the copy acceptance rate: a model that is quoting tends
# to keep quoting, and the rate is corrected from verified rounds
COPY_RATE_SEED = 0.8
# the acceptance odds are learned per range of distinct continuations
FAN_BUCKETS = (1, 2, 4, 8)
_FAN_LOGS = [math.log1p(b) for b in FAN_BUCKETS]
_SOURCES = 2    # the drafting document's own output, or a sibling's


def _nearest_bucket(fan_out: int) -> int:
    """Return the index of the fan-out range nearest on a log scale."""
    x = math.log1p(fan_out)
    i = bisect.bisect_left(_FAN_LOGS, x)
    if i == 0 or i == len(_FAN_LOGS):
        return min(i, len(_FAN_LOGS) - 1)
    return i if _FAN_LOGS[i] - x < x - _FAN_LOGS[i - 1] else i - 1


# the cache keeps at most a handful of continuations per key, so the
# buckets of small fan-outs cover every lookup
_SMALL_FAN_BUCKETS = tuple(_nearest_bucket(f) for f in range(64))


def _fan_bucket(fan_out: int) -> int:
    """Return the index of the fan-out range nearest on a log scale."""
    if fan_out < len(_SMALL_FAN_BUCKETS):
        return _SMALL_FAN_BUCKETS[fan_out]
    return _nearest_bucket(fan_out)


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
        # copied drafts' verified tokens: unlike an extraction, a map's
        # answer may leave the document at any token, so a copy is a
        # per-token bet like a chain hop
        self.copy_trials = 0
        self.copy_accepts = 0

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
        return self.odds(match)[1]

    def odds(self, match: NgramMatch) -> tuple:
        """Return one cache match's cell and its hit rate, computed once."""
        c = self.cell(match)
        seed = PRIOR_WEIGHT * match.share
        return c, (seed + self.accepts[c]) / (PRIOR_WEIGHT + self.trials[c])

    def _room(self, written: int) -> int:
        return min(self.cap, self.max_answer_tokens - written - 1)

    def copy_rate(self) -> float:
        """Return P(the model writes a copied draft's next token)."""
        return ((PRIOR_WEIGHT * COPY_RATE_SEED + self.copy_accepts)
                / (PRIOR_WEIGHT + self.copy_trials))

    def observe_copy(self, accepted: bool) -> None:
        """Record whether a copied draft token was written."""
        self.copy_trials += 1
        self.copy_accepts += accepted

    def copy_length(self, rho: float, written: int, slack: float) -> int:
        """Return the draft length for a round copying from the document.

        A copied draft writes while the model keeps quoting the
        document and the answer goes on; the quoting odds are learned
        per token from the verified rounds.

        Args:
            rho: The round's KV read, in fed tokens.
            written: The answer's tokens so far.
            slack: Fed tokens the current passes carry for free.
        """
        return self._best_length(rho, written, self.copy_rate(), slack)

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
        return self._best_length(rho, written, rates, slack, room=room)

    def _best_length(self, rho: float, written: int, rates, slack: float,
                     room: int | None = None) -> int:
        """Return the draft length with the least cost per written token.

        The first slack draft tokens ride a pass below the ridge, so
        they cost nothing; each one past the slack costs one token.
        Draft token j writes when the answer goes on and every hop
        through j wrote. The arrays are at most cap plus one long, so
        plain loops beat array allocation here.

        Args:
            rho: The round's KV read, in fed tokens.
            written: The answer's tokens so far.
            rates: Each hop's acceptance odds, in hop order; a float is
                one rate for every hop, as a copied draft has.
            slack: Fed tokens the current passes carry for free.
            room: The most tokens the draft may have; None derives it
                from the answer limit.
        """
        if room is None:
            room = self._room(written)
        if room <= 0:
            return 0
        uniform = rates if isinstance(rates, float) else None
        levels = self.curve.levels()
        base = levels[written]
        writes = 1.0
        chained = 1.0
        best, best_cost = 0, rho + 1.0
        for gamma in range(1, room + 1):
            chained *= uniform if uniform is not None else rates[gamma - 1]
            # survival at gamma tokens past written, 1 past a dead base
            writes += chained * (levels[written + gamma] / base if base > 0
                                 else 1.0)
            extra = gamma - slack
            cost = (rho + 1.0 + (extra if extra > 0 else 0.0)) / writes
            if cost < best_cost:
                best, best_cost = gamma, cost
        return best

    def observe(self, match_cell: int, accepted: bool) -> None:
        """Record whether a chained draft token was written.

        Args:
            match_cell: The cell of the match that proposed the token.
            accepted: Whether the model chose the token.
        """
        self.trials[match_cell] += 1
        self.accepts[match_cell] += accepted
