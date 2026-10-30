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
                                 decoder.kv.num_pages[slots] * page_tokens
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
        groups, shared = (decode_groups(slots, decoder.kv.parent, decoder.kv.borrowed,
                                        decoder.kv.lead)
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

    def waits(self):
        """Return whether a refused document can wait for pages to free."""
        return bool(self.maximum or len(self.decoder.running()))


def input_bucket(tokens):
    """Return the input-length bucket: one per doubling of the prefix length."""
    return np.floor(np.log2(np.maximum(np.asarray(tokens, dtype=float), 1.0))
                    ).astype(np.int64)


class HeadroomAdmission(AnswerAdmission):
    """Admit a document when its prompt and its answers' headroom fit.

    Each answer has its own headroom: pages its answer may grow into
    beyond the pages it holds. Free pages minus the headroom of running
    answers, minus the pages evicted answers need to come back, must
    cover a new document's prompt and its answers' headroom. The pages
    are counted, not allocated: answers still grow one page at a time.

    With choose_headroom False every headroom is zero, so a document is
    admitted when its prompt fits. Otherwise each answer's headroom is
    the page count j minimizing D(j) + P(Y > j) * R(j):

    - P(Y > j): the chance the answer needs more than j more pages,
      from the Kaplan-Meier curve of its map and input-length bucket,
      conditional on the tokens it has written. A bucket uses the curve
      of all its map's answers until bucket_answers of its own ended.
    - D(j): j times the value of one page to waiting documents, the
      SoL seconds one waiting document saves by joining passes that
      run anyway, divided by the pages its prompt takes.
    - R(j): the SoL seconds rebuilding the answer's prefix adds when
      it outgrows j pages and is evicted.

    Headrooms are chosen again each time the curves are rebuilt.
    """

    def __init__(self, decoder, prefix_lengths, capacity_extra, dependencies,
                 shared, page_cost, decode_cost=None, page_holds=None, *,
                 choose_headroom=True, chunk_tokens=0, curve_refresh=32,
                 bucket_answers=30):
        super().__init__(decoder, prefix_lengths, capacity_extra, dependencies,
                         shared, page_cost, decode_cost, page_holds)
        self.page_cost = page_cost
        self.capacity_extra = capacity_extra
        self.choose_headroom = choose_headroom and decode_cost is not None
        self.chunk_tokens = chunk_tokens
        self.curve_refresh = curve_refresh
        self.bucket_answers = bucket_answers
        self.bucket = input_bucket(self.prefix)
        self.slot_map = np.full(len(self.prefix), -1, dtype=np.int64)
        # per slot, the KV position its pages and headroom reach
        self.reach = np.zeros(len(self.prefix), dtype=np.int64)
        self.headroom_pages = 0
        self.replay_pages = 0
        self.page_value = 0.0
        self.bucket_curves = {}
        self._ended = -1
        self._rebuilt = 0
        self._refreshes = 0
        # (refresh, map, bucket, answers, mean headroom pages) per rebuild
        self.trace = []
        # headroom pages each admitted answer was given, and
        # (refresh, bucket, pages) for each
        self.granted = []
        self.granted_at = []

    def refresh(self, stages, replays=(), replay_pages=0):
        """Rebuild the work, the curves when answers ended, and each headroom."""
        super().refresh(stages, replays)
        decoder = self.decoder
        for q, slots in decoder.answer_slots.items():
            self.slot_map[np.asarray(slots, dtype=np.int64)] = q
        self.replay_pages = int(replay_pages)
        self._refreshes += 1
        ended = sum(int((lengths >= 0).sum()) for lengths in decoder.lengths)
        # rebuild after a tenth more answers ended, or every curve_refresh
        if self.choose_headroom and (
                self._ended < 0 or ended > 1.1 * self._ended + 1
                or self._refreshes - self._rebuilt >= self.curve_refresh):
            self._ended, self._rebuilt = ended, self._refreshes
            self._rebuild_curves()
            self.page_value = self._page_value(stages)
            running = self._running()
            self.reach[running] = (decoder.kv.num_pages[running] * decoder.page_tokens
                                   + self._choose(running) * decoder.page_tokens)
            self._record(running)
        running = self._running()
        self.headroom_pages = int(self._remaining(running).sum())
        # pending branches of admitted documents need their prompt pages
        roots = self.committed[self.parent]
        self.headroom_pages += int(self.pending_pages[roots].sum())

    def _remaining(self, slots):
        """Headroom pages the slots' answers have not grown into yet."""
        page_tokens = self.decoder.page_tokens
        held = self.decoder.kv.num_pages[slots] * page_tokens
        return np.maximum(0, -(-(self.reach[slots] - held) // page_tokens))

    def _running(self):
        decoder = self.decoder
        return np.flatnonzero((decoder.map >= 0) & ~decoder.paused)

    def _available(self, slots):
        """Answer tokens the slots' allocated pages cover."""
        decoder = self.decoder
        return (decoder.kv.num_pages[slots] * decoder.page_tokens
                - decoder.origin[slots])

    def _rebuild_curves(self):
        decoder = self.decoder
        self.bucket_curves = {}
        for q, slots in decoder.answer_slots.items():
            slots = np.asarray(slots, dtype=np.int64)
            limit = int(decoder.limits[q])
            lengths, written = decoder.lengths[q][slots], decoder.written[q][slots]
            whole = answer_survival(lengths, written, decoder.expected_tokens[q],
                                    limit)
            buckets = self.bucket[slots]
            for b in np.unique(buckets):
                mine = buckets == b
                # a bucket with few ended answers keeps the map's curve
                if int((lengths[mine] >= 0).sum()) < self.bucket_answers:
                    self.bucket_curves[(q, int(b))] = whole
                    continue
                self.bucket_curves[(q, int(b))] = answer_survival(
                    lengths[mine], written[mine], decoder.expected_tokens[q],
                    limit, prior=whole)

    def _page_value(self, stages):
        """SoL seconds one page saves a waiting document, 0 when none waits."""
        waiting = np.flatnonzero((stages == -1) & self.roots & ~self.committed)
        if not len(waiting):
            return 0.0
        anchor = int(waiting[0])
        added = self._added_work(anchor)
        # with nothing running, price joining a pass that holds one such answer
        base = self.work if self.work.tokens else added
        gain = self.decode_cost.gain(base, added, self.expected_tokens[anchor])
        pages = self.page_cost(int(self.prefix[anchor]) + self.capacity_extra)
        return gain / max(1, pages)

    def _choose(self, slots, written=None, available=None, origin=None):
        """Return each slot's headroom pages, minimizing D(j) + P(Y > j) R(j)."""
        decoder = self.decoder
        page_tokens = decoder.page_tokens
        chosen = np.zeros(len(slots), dtype=np.int64)
        if not len(slots):
            return chosen
        current = decoder.map[slots]
        maps = np.where(current >= 0, current, self.slot_map[slots])
        if written is None:
            written = np.zeros(len(slots), dtype=np.int64)
            for q in np.unique(maps):
                mine = maps == q
                written[mine] = decoder.written[q][slots[mine]]
            available = self._available(slots)
            origin = decoder.origin[slots]
        keys = maps * (1 << 32) + self.bucket[slots]
        for key in np.unique(keys):
            mine = np.flatnonzero(keys == key)
            curve = self.bucket_curves.get((int(key >> 32), int(key & 0xFFFFFFFF)))
            if curve is None:
                continue
            limit = len(curve) - 1
            pages = np.arange(-(-limit // page_tokens) + 1)
            held = np.minimum(limit, available[mine])[:, None]
            reach = np.minimum(limit, held + 1 + pages * page_tokens)
            survived = curve[np.minimum(limit, written[mine])][:, None]
            overrun = np.divide(curve[reach], survived, out=np.zeros(reach.shape),
                                where=survived > 0)
            # an overrun is evicted with the tokens its pages held
            tokens = origin[mine][:, None] + np.minimum(
                limit, held + pages * page_tokens)
            recover = self.decode_cost.replay_seconds(self.work, tokens,
                                                      self.chunk_tokens)
            chosen[mine] = np.argmin(pages * self.page_value + overrun * recover,
                                     axis=1)
        return chosen

    def _record(self, running):
        for (q, b) in self.bucket_curves:
            mine = running[(self.slot_map[running] == q)
                           & (self.bucket[running] == b)]
            if len(mine):
                self.trace.append((self._refreshes, q, b, len(mine),
                                   float(self._remaining(mine).mean())))

    def _candidate_headroom(self, anchor):
        """Headroom pages a document's answers would hold, chosen now."""
        if not self.choose_headroom:
            return 0
        slots = np.flatnonzero((self.parent == anchor) & (self.slot_map >= 0))
        if not len(slots):
            return 0
        zeros = np.zeros(len(slots), dtype=np.int64)
        chosen = self._choose(slots, zeros, self.capacity[slots] - self.prompt[slots],
                              self.prompt[slots])
        self.reach[slots] = self.capacity[slots] + chosen * self.decoder.page_tokens
        return int(chosen.sum())

    def allows(self, anchor, free_pages, prompt_pages):
        """Admit when the prompt and its answers' headroom fit beside what is held."""
        if self.committed[self.parent[anchor]]:
            return True
        need = prompt_pages + self._candidate_headroom(anchor)
        return free_pages - self.headroom_pages - self.replay_pages >= need

    def admit(self, anchor):
        """Count an admitted document's headroom, or a branch's allocated prompt."""
        if self.committed[self.parent[anchor]]:
            self.headroom_pages -= int(self.pending_pages[anchor])
            self.pending_pages[anchor] = 0
            return
        self.committed[anchor] = True
        self.work = self.work + self._added_work(anchor)
        slots = np.flatnonzero((self.parent == anchor) & (self.slot_map >= 0))
        granted = ((self.reach[slots] - self.capacity[slots])
                   // self.decoder.page_tokens).clip(0)
        self.granted.extend(granted.tolist())
        self.granted_at.extend((self._refreshes, int(self.bucket[slot]), int(pages))
                               for slot, pages in zip(slots, granted))
        self.headroom_pages += int(granted.sum())
        branches = np.flatnonzero((self.parent == anchor)
                                  & (np.arange(len(self.prefix)) != anchor))
        self.headroom_pages += int(self.pending_pages[branches].sum())

    def waits(self):
        """Return whether a refused document can wait for pages to free."""
        decoder = self.decoder
        evicted = decoder.paused & (decoder.map >= 0)
        return bool(len(decoder.running()) or evicted.any())
