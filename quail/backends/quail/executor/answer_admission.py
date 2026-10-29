"""Pool answer-page uncertainty and price admission in token-computations."""

from math import erfc, sqrt

import numpy as np

from quail.backends.quail.executor.answer_survival import answer_survival


def page_moments(survival, written, available, page_tokens):
    """Return mean and variance of additional pages, conditional on survival.

    available counts answer KV tokens already covered by allocated pages.
    The terminal answer token is not fed back into the model. For extra pages Y,
    P(Y > j) = S(available + 1 + j * page_tokens) / S(written). Summing these
    tails gives E[Y]; weighting them by 2*j + 1 gives E[Y squared].
    """
    limit = len(survival) - 1
    padded = np.pad(survival, (0, (-len(survival)) % page_tokens))
    rows = padded.reshape(-1, page_tokens)
    tail = np.cumsum(rows[::-1], axis=0)[::-1]
    second = 2 * np.cumsum(tail[::-1], axis=0)[::-1] - tail
    index = np.minimum(limit, np.maximum(written, available + 1))
    probability = survival[written]
    mean = np.divide(tail.ravel()[index], probability,
                     out=np.zeros_like(probability), where=probability > 0)
    square = np.divide(second.ravel()[index], probability,
                       out=np.zeros_like(probability), where=probability > 0)
    return mean, np.maximum(0, square - mean * mean)


def overflow_probability(free_pages, mean, variance, maximum):
    """Return the pooled normal tail, respecting the bounded answer support."""
    if free_pages >= maximum:
        return 0.0
    if free_pages < 0:
        return 1.0
    if variance <= 0:
        return float(free_pages < mean)
    return erfc((free_pages - mean) / sqrt(2 * variance)) / 2


def admission_worthwhile(free_pages, prompt_pages, mean, variance, maximum,
                        added_mean, added_variance, added_maximum,
                        gain, recompute):
    """Admit when free-pass work pays for the added expected recomputation."""
    before = overflow_probability(free_pages, mean, variance, maximum)
    after = overflow_probability(free_pages - prompt_pages, mean + added_mean,
                                 variance + added_variance,
                                 maximum + added_maximum)
    risk = max(0.0, after - before)
    return risk == 0 or gain > risk * recompute


class AnswerAdmission:
    """Per-pass pooled page demand for admitted documents and their branches."""

    def __init__(self, decoder, prefix_lengths, capacity_extra, dependencies,
                 shared, page_cost, recompute_cost=None):
        self.decoder = decoder
        self.prefix = np.asarray(prefix_lengths, dtype=np.int64)
        self.capacity_extra = capacity_extra
        self.parent = np.arange(len(self.prefix))
        for anchor, (parent, _) in dependencies.items():
            self.parent[anchor] = parent
        self.prompt_pages = np.array([
            page_cost(int(prefix) + capacity_extra - shared[anchor])
            if anchor in dependencies else 0
            for anchor, prefix in enumerate(self.prefix)])
        self.recompute_cost = recompute_cost
        self.mean = self.variance = self.maximum = 0.0

    def refresh(self, stages):
        """Rebuild moments with NumPy arrays, once before an admission batch."""
        admitted = stages >= 0
        decoder = self.decoder
        page_tokens = decoder.page_tokens
        num_anchors = len(self.prefix)
        self.added_mean = np.zeros(num_anchors)
        self.added_variance = np.zeros(num_anchors)
        self.added_maximum = np.zeros(num_anchors)
        self.gain = np.zeros(num_anchors)
        self.committed = admitted.copy()
        self.pending_pages = np.where(stages == -1, self.prompt_pages, 0)
        start_tokens = min(page_tokens, int(decoder.limits.max()) - 1)
        prompt = self.prefix + self.capacity_extra - start_tokens
        capacity = -(-(self.prefix + self.capacity_extra) // page_tokens) * page_tokens
        for q, slots in decoder.answer_slots.items():
            slots = np.asarray(slots, dtype=np.int64)
            slots = slots[(decoder.lengths[q][slots] < 0) & ~decoder.paused[slots]
                          & (stages[slots] != -2)]
            roots = self.parent[slots]
            curve = answer_survival(decoder.lengths[q], decoder.written[q],
                                    decoder.expected_tokens[q], int(decoder.limits[q]))
            current = decoder.map[slots] == q
            available = np.where(current,
                                 decoder.num_pages[slots] * page_tokens
                                 - decoder.origin[slots],
                                 capacity[slots] - prompt[slots])
            mean, variance = page_moments(
                curve, decoder.written[q][slots], available, page_tokens)
            maximum = np.maximum(0, -(-(decoder.limits[q] - 1 - available)
                                       // page_tokens))
            np.add.at(self.added_mean, roots, mean)
            np.add.at(self.added_variance, roots, variance)
            np.add.at(self.added_maximum, roots, maximum)
            np.add.at(self.gain, roots, curve.sum())
        # A pending branch also needs its deterministic prompt pages.
        np.add.at(self.added_mean, self.parent, self.pending_pages)
        np.add.at(self.added_maximum, self.parent, self.pending_pages)
        roots = admitted & (self.parent == np.arange(num_anchors))
        self.mean = float(self.added_mean[roots].sum())
        self.variance = float(self.added_variance[roots].sum())
        self.maximum = float(self.added_maximum[roots].sum())
        order = decoder.eviction_order()
        self.recompute = None
        if len(order):
            victim = order[0]
            tokens = (decoder.origin[victim]
                      + decoder.written[decoder.map[victim]][victim])
            self.recompute = self._recompute_work(tokens)

    def _recompute_work(self, tokens):
        policy = self.recompute_cost
        if policy is None or not policy.linear_seconds:
            return float(tokens)
        return policy.prefill_seconds(tokens) / policy.linear_seconds

    def allows(self, anchor, free_pages, prompt_pages, rows, knee_rows):
        """Price a new document; its already-priced branches can start freely."""
        if self.committed[self.parent[anchor]]:
            return True
        recompute = self.recompute
        if recompute is None:
            recompute = self._recompute_work(self.prefix[anchor] + self.capacity_extra)
        return admission_worthwhile(
            free_pages, prompt_pages, self.mean, self.variance, self.maximum,
            self.added_mean[anchor], self.added_variance[anchor],
            self.added_maximum[anchor],
            self.gain[anchor] if rows < knee_rows else 0.0, recompute)

    def admit(self, anchor):
        """Commit a document's demand or remove a branch's allocated prompt pages."""
        if self.committed[self.parent[anchor]]:
            self.mean -= self.pending_pages[anchor]
            self.maximum -= self.pending_pages[anchor]
            self.pending_pages[anchor] = 0
        else:
            self.mean += self.added_mean[anchor]
            self.variance += self.added_variance[anchor]
            self.maximum += self.added_maximum[anchor]
            self.committed[anchor] = True
