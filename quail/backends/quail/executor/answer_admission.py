"""Pool answer-page uncertainty and price admission in seconds."""

from math import erfc, sqrt

import numpy as np

from quail.backends.quail.executor.answer_survival import answer_survival
from quail.backends.quail.executor.chunk import decode_groups
from quail.cost.work import Work


def page_moments(survival, written, available, page_tokens):
    """Return mean and variance of additional pages, conditional on survival.

    available counts answer KV tokens already covered by allocated pages.
    The terminal answer token is not fed back into the model. For extra pages Y,
    P(Y > j) = S(available + 1 + j * page_tokens) / S(written). Summing these
    tails gives E[Y]; weighting them by 2*j + 1 gives E[Y squared].
    """
    limit = len(survival) - 1
    padded = np.zeros(len(survival) + (-len(survival)) % page_tokens)
    padded[:len(survival)] = survival
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
                 shared, page_cost, decode_cost=None, page_holds=None):
        self.decoder = decoder
        self.prefix = np.asarray(prefix_lengths, dtype=np.int64)
        self.parent = np.arange(len(self.prefix))
        for anchor, (parent, _) in dependencies.items():
            self.parent[anchor] = parent
        self.prompt_pages = np.array([
            page_cost(int(prefix) + capacity_extra - shared[anchor])
            if anchor in dependencies else 0
            for anchor, prefix in enumerate(self.prefix)])
        page_tokens = decoder.page_tokens
        start_tokens = min(page_tokens, int(decoder.limits.max()) - 1)
        self.prompt = self.prefix + capacity_extra - start_tokens
        self.capacity = -(-(self.prefix + capacity_extra) // page_tokens) * page_tokens
        self.roots = self.parent == np.arange(len(self.prefix))
        self.decode_cost = decode_cost
        self.page_holds = page_holds
        self.borrowed = np.where(self.roots, 0, shared)
        self.lead = np.zeros(len(self.prefix), dtype=np.int64)
        np.maximum.at(self.lead, self.parent, self.borrowed)
        self.mean = self.variance = self.maximum = 0.0

    def refresh(self, stages, replays=()):
        """Rebuild moments for running answers and this pass's reserved replays."""
        admitted = stages >= 0
        decoder = self.decoder
        ready = ~decoder.paused
        if replays:
            ready[np.asarray(replays)[:, 0]] = True
        page_tokens = decoder.page_tokens
        num_anchors = len(self.prefix)
        self.added_mean = np.zeros(num_anchors)
        self.added_variance = np.zeros(num_anchors)
        self.added_maximum = np.zeros(num_anchors)
        self.expected_tokens = np.zeros(num_anchors)
        answer_slots, contexts = [], []
        self.committed = admitted.copy()
        self.pending_pages = np.where(stages == -1, self.prompt_pages, 0)
        curves = {}
        for q, slots in decoder.answer_slots.items():
            slots = np.asarray(slots, dtype=np.int64)
            slots = slots[(decoder.lengths[q][slots] < 0) & ready[slots]
                          & (stages[slots] != -2)]
            roots = self.parent[slots]
            curves[q] = curve = answer_survival(
                decoder.lengths[q], decoder.written[q],
                decoder.expected_tokens[q], int(decoder.limits[q]))
            current = decoder.map[slots] == q
            available = np.where(current,
                                 decoder.num_pages[slots] * page_tokens
                                 - decoder.origin[slots],
                                 self.capacity[slots] - self.prompt[slots])
            mean, variance = page_moments(
                curve, decoder.written[q][slots], available, page_tokens)
            maximum = np.maximum(0, -(-(decoder.limits[q] - 1 - available)
                                       // page_tokens))
            # Each map has one answer per document.
            self.added_mean[roots] += mean
            self.added_variance[roots] += variance
            self.added_maximum[roots] += maximum
            expected_tokens = curve.sum()
            self.expected_tokens[roots] += expected_tokens
            answer_slots.append(slots)
            contexts.append(self.prompt[slots] + expected_tokens / 2)
        self._refresh_work(answer_slots, contexts, replays)
        # A pending branch also needs its deterministic prompt pages.
        pending = np.bincount(self.parent, weights=self.pending_pages,
                              minlength=num_anchors)
        self.added_mean += pending
        self.added_maximum += pending
        roots = admitted & self.roots
        self.mean = float(self.added_mean[roots].sum())
        self.variance = float(self.added_variance[roots].sum())
        self.maximum = float(self.added_maximum[roots].sum())
        order = self.eviction_order(curves)
        self.recompute = None
        if len(order):
            victim = order[0]
            tokens = (decoder.origin[victim]
                      + decoder.written[decoder.map[victim]][victim])
            self.recompute = self._recompute_seconds(tokens)

    def eviction_order(self, curves=None):
        """Rank the same victims for admission pricing and actual eviction."""
        policy = None if self.decode_cost is None else self.decode_cost.recompute_cost
        holds = None if self.page_holds is None else self.page_holds()
        return self.decoder.eviction_order(curves, policy, holds)

    def _refresh_work(self, answer_slots, contexts, replays):
        cost = self.decode_cost
        self.work = Work()
        self.added_work = {}
        if cost is None:
            return
        decoder = self.decoder
        slots = np.concatenate(answer_slots)
        contexts = np.concatenate(contexts)
        groups, shared = (decode_groups(slots, self.parent, self.borrowed, self.lead)
                          if decoder.share else (None, None))
        work = cost.row_work(contexts, groups, shared)
        self.added_work = {
            name: np.bincount(self.parent[slots], weights=values,
                              minlength=len(self.prefix))
            for name, values in vars(work).items() if np.ndim(values)}
        running = decoder.running()
        counts = decoder.row_counts(running)
        offsets = (np.arange(int(counts.sum()))
                   - np.repeat(np.cumsum(counts) - counts, counts))
        slots = np.repeat(running, counts)
        contexts = decoder.origin[slots] + decoder.fed[slots] + offsets + 1
        if replays:
            replay = np.asarray(replays)
            slots = np.r_[slots, replay[:, 0]]
            contexts = np.r_[contexts, replay[:, 2] + 1]
        groups, shared = (decode_groups(slots, decoder.parent, decoder.borrowed,
                                        decoder.lead)
                          if decoder.share else (None, None))
        self.work = cost.row_work(contexts, groups, shared).total()

    def _recompute_seconds(self, tokens):
        if self.decode_cost is None:
            return float("inf")
        return self.decode_cost.recompute_cost.prefill_seconds(tokens)

    def _added_work(self, anchor):
        return Work(**{name: values[anchor]
                       for name, values in self.added_work.items()})

    def allows(self, anchor, free_pages, prompt_pages):
        """Price a new document; its already-priced branches can start freely."""
        if self.committed[self.parent[anchor]]:
            return True
        recompute = self.recompute
        if recompute is None:
            recompute = self._recompute_seconds(self.prompt[anchor])
        gain = (0.0 if self.decode_cost is None else self.decode_cost.gain(
            self.work, self._added_work(anchor), self.expected_tokens[anchor]))
        return admission_worthwhile(
            free_pages, prompt_pages, self.mean, self.variance, self.maximum,
            self.added_mean[anchor], self.added_variance[anchor],
            self.added_maximum[anchor], gain, recompute)

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
            self.work = self.work + self._added_work(anchor)
