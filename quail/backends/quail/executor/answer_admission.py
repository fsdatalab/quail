"""Admit map documents by the headroom their answers' survival curves call for."""

import numpy as np

from quail.backends.quail.executor.chunk import decode_groups
from quail.cost.answers import prior_survival
from quail.cost.work import Work

PRIOR_WEIGHT = 20.0


def input_bucket(tokens):
    """Return the input-length bucket: one per doubling of the prefix length."""
    return np.floor(np.log2(np.maximum(np.asarray(tokens, dtype=float), 1.0))
                    ).astype(np.int64)


def survival_rows(counts, events, prior, limits):
    """Return Kaplan-Meier curves, one per row, as answer_survival builds one.

    Args:
        counts: Answers observed last at each token, ended or censored,
            one row per curve.
        events: Answers that ended at each token, one row per curve.
        prior: The prior curve of each row, 0 past the row's limit.
        limits: Each row's answer token limit.

    Returns:
        S(t) per row, 0 from each row's limit on.
    """
    risk = (np.cumsum(counts[:, ::-1], axis=1)[:, ::-1][:, 1:]
            + PRIOR_WEIGHT * prior[:, :-1])
    events = events[:, 1:] + PRIOR_WEIGHT * (prior[:, :-1] - prior[:, 1:])
    hazard = np.divide(events, risk, out=np.zeros(risk.shape), where=risk > 0)
    survival = np.empty(counts.shape)
    survival[:, 0] = 1.0
    np.cumprod(1.0 - hazard, axis=1, out=survival[:, 1:])
    survival[np.arange(counts.shape[1]) >= limits[:, None]] = 0.0
    return survival


class HeadroomAdmission:
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
      run anyway, divided by the pages it takes: its prompt and the
      headroom its answers get at that value.
    - R(j): the SoL seconds rebuilding the answer's whole prefix adds
      when it outgrows j pages and is evicted.

    Every refresh rebuilds the curves of maps whose answers moved, the
    value of a page, and the headroom of running answers, pricing
    every R(j) in one array. With trace, each refresh also records the
    running answers' headroom by map and bucket.
    """

    def __init__(self, decoder, prefix_lengths, capacity_extra, dependencies,
                 shared, page_cost, decode_cost=None, page_holds=None, *,
                 choose_headroom=True, chunk_tokens=0, bucket_answers=30,
                 trace=False):
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
        self.page_cost = page_cost
        self.capacity_extra = capacity_extra
        self.choose_headroom = choose_headroom and decode_cost is not None
        self.chunk_tokens = chunk_tokens
        self.bucket_answers = bucket_answers
        self.bucket = input_bucket(self.prefix)
        self.num_buckets = int(self.bucket.max()) + 1 if len(self.prefix) else 1
        limits = decoder.limits
        self.width = int(limits.max()) + 1
        # per map, the planner's geometric prior, 0 past the map's limit
        self.prior = np.zeros((len(limits), self.width))
        for q, limit in enumerate(limits):
            self.prior[q, :limit + 1] = prior_survival(
                decoder.expected_tokens[q], int(limit))
        self.slot_map = np.full(len(self.prefix), -1, dtype=np.int64)
        # per slot, its curve's row: map * num_buckets + bucket
        self.slot_row = np.full(len(self.prefix), -1, dtype=np.int64)
        self.slots = None
        self.curves = None
        self.whole = None
        self._observed = None
        # per slot, the KV position its pages and headroom reach
        self.reach = np.zeros(len(self.prefix), dtype=np.int64)
        self.headroom_pages = 0
        self.replay_pages = 0
        self.page_value = 0.0
        self.work = Work()
        self.added_work = {}
        # (anchor, its answer slots, their R(j) grid) for the first
        # waiting document, valid until the work changes
        self._candidate = None
        self._refreshes = 0
        # (refresh, map, bucket, answers, mean headroom pages) per rebuild,
        # kept only when traced
        self.traced = trace
        self.trace = []
        # headroom pages each admitted answer was given, and
        # (refresh, bucket, pages) for each
        self.granted = []
        self.granted_at = []

    def _bind(self):
        decoder = self.decoder
        self.slots = np.concatenate(
            [np.asarray(slots, dtype=np.int64)
             for slots in decoder.answer_slots.values()])
        self.slot_map[self.slots] = np.concatenate(
            [np.full(len(slots), q) for q, slots in decoder.answer_slots.items()])
        maps = self.slot_map[self.slots]
        self.slot_row[self.slots] = maps * self.num_buckets + self.bucket[self.slots]
        self.rows = np.unique(self.slot_row[self.slots])
        self.row_limit = np.repeat(decoder.limits, self.num_buckets)
        self.curves = np.zeros((len(self.row_limit), self.width))
        self.whole = np.zeros((len(decoder.limits), self.width))

    @property
    def bucket_curves(self):
        """The curve of each (map, bucket) with answers, cut at its map's limit."""
        if self.curves is None:
            return {}
        return {(int(row // self.num_buckets), int(row % self.num_buckets)):
                self.curves[row, :self.row_limit[row] + 1] for row in self.rows}

    def refresh(self, stages, replays=(), replay_pages=0):
        """Count held pages; with headroom, rebuild the curves and choose it."""
        self.committed = stages >= 0
        self.pending_pages = np.where(stages == -1, self.prompt_pages, 0)
        self.replay_pages = int(replay_pages)
        self._refreshes += 1
        running = self._running()
        if self.choose_headroom:
            self._refresh_curves(stages, replays)
            self._choose_running(stages, running)
            if self.traced:
                self._record(running)
        self.headroom_pages = int(self._remaining(running).sum())
        # pending branches of admitted documents need their prompt pages
        roots = self.committed[self.parent]
        self.headroom_pages += int(self.pending_pages[roots].sum())

    def _refresh_curves(self, stages, replays):
        """Rebuild the curves of maps whose answers moved, and the work."""
        if self.slots is None:
            self._bind()
        decoder = self.decoder
        slots, maps = self.slots, self.slot_map[self.slots]
        lengths = decoder.lengths[maps, slots]
        observed = np.where(lengths >= 0, lengths, decoder.written[maps, slots])
        changed = (np.ones(len(decoder.limits), dtype=bool)
                   if self._observed is None
                   else np.bincount(maps[observed != self._observed],
                                    minlength=len(decoder.limits)) > 0)
        self._observed = observed
        if changed.any():
            self._rebuild_curves(maps, lengths, observed, changed)
        expected = self.whole.sum(axis=1)
        ready = ~decoder.paused
        if replays:
            ready[np.asarray(replays)[:, 0]] = True
        unfinished = (lengths < 0) & ready[slots] & (stages[slots] != -2)
        slots, expected = slots[unfinished], expected[maps[unfinished]]
        self.expected_tokens = np.bincount(self.parent[slots], weights=expected,
                                           minlength=len(self.prefix))
        self._refresh_work(slots, self.prompt[slots] + expected / 2, replays)
        self._candidate = None

    def _rebuild_curves(self, maps, lengths, observed, changed):
        """Rebuild the whole-map and bucket curves of the changed maps."""
        num_maps, width = len(self.decoder.limits), self.width
        ended = lengths >= 0
        counts = np.bincount(maps * width + observed,
                             minlength=num_maps * width).reshape(num_maps, width)
        events = np.bincount(maps[ended] * width + lengths[ended],
                             minlength=num_maps * width).reshape(num_maps, width)
        mine = np.flatnonzero(changed)
        self.whole[mine] = survival_rows(counts[mine], events[mine],
                                         self.prior[mine],
                                         self.decoder.limits[mine])
        rows = self.rows[changed[self.rows // self.num_buckets]]
        self.curves[rows] = self.whole[rows // self.num_buckets]
        # a bucket with few ended answers keeps its map's curve
        row_of = self.slot_row[self.slots]
        num_ended = np.bincount(row_of[ended], minlength=len(self.curves))
        rows = rows[num_ended[rows] >= self.bucket_answers]
        if not len(rows):
            return
        index = np.full(len(self.curves), -1, dtype=np.int64)
        index[rows] = np.arange(len(rows))
        position = index[row_of]
        take = position >= 0
        counts = np.bincount(position[take] * width + observed[take],
                             minlength=len(rows) * width).reshape(len(rows), width)
        take &= ended
        events = np.bincount(position[take] * width + lengths[take],
                             minlength=len(rows) * width).reshape(len(rows), width)
        self.curves[rows] = survival_rows(counts, events,
                                          self.whole[rows // self.num_buckets],
                                          self.row_limit[rows])

    def eviction_order(self, curves=None):
        """Rank running answers for eviction, the first evicted first."""
        policy = None if self.decode_cost is None else self.decode_cost.recompute_cost
        holds = None if self.page_holds is None else self.page_holds()
        return self.decoder.eviction_order(curves, policy, holds)

    def _refresh_work(self, slots, contexts, replays):
        cost = self.decode_cost
        self.work = Work()
        self.added_work = {}
        if cost is None:
            return
        decoder = self.decoder
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

    def _added_work(self, anchor):
        return Work(**{name: values[anchor]
                       for name, values in self.added_work.items()})

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

    def _waiting(self, stages):
        """The first document waiting for admission, or None."""
        waiting = np.flatnonzero((stages == -1) & self.roots & ~self.committed)
        return int(waiting[0]) if len(waiting) else None

    def _answer_slots(self, anchor):
        return np.flatnonzero((self.parent == anchor) & (self.slot_map >= 0))

    def _choose_running(self, stages, running):
        """Price a page and choose running answers' headroom from one R(j) grid."""
        decoder = self.decoder
        anchor = self._waiting(stages)
        pending = (self._answer_slots(anchor) if anchor is not None
                   else np.zeros(0, dtype=np.int64))
        maps = decoder.map[running]
        risk = self._risk(
            np.r_[running, pending], np.r_[maps, self.slot_map[pending]],
            np.r_[decoder.written[maps, running], np.zeros(len(pending), np.int64)],
            np.r_[self._available(running),
                  self.capacity[pending] - self.prompt[pending]],
            np.r_[decoder.origin[running], self.prompt[pending]])
        if anchor is not None:
            self._candidate = (anchor, pending, risk[len(running):])
            self.page_value = self._page_value(anchor, risk[len(running):])
        else:
            self.page_value = 0.0
        chosen = self._argmin(risk[:len(running)], self.page_value)
        self.reach[running] = (decoder.kv.num_pages[running] + chosen
                               ) * decoder.page_tokens

    def _page_value(self, anchor, risk):
        """SoL seconds one page saves a waiting document, 0 when none waits.

        The document takes its prompt's pages and its answers' headroom,
        and that headroom depends on the value of a page. The value is
        the fixed point v = gain / (prompt pages + headroom at v), found
        from the prompt-only value: each step lowers v and raises the
        headroom, a whole number of pages, so it ends where the headroom
        repeats.
        """
        added = self._added_work(anchor)
        # with nothing running, price joining a pass that holds one such answer
        base = self.work if self.work.tokens else added
        gain = self.decode_cost.gain(base, added, self.expected_tokens[anchor])
        prompt_pages = self.page_cost(int(self.prefix[anchor]) + self.capacity_extra)
        value, headroom = gain / max(1, prompt_pages), None
        while True:
            pages = int(self._argmin(risk, value).sum())
            if pages == headroom:
                return value
            headroom = pages
            value = gain / max(1, prompt_pages + headroom)

    @staticmethod
    def _argmin(risk, page_value):
        """Each row's page count minimizing j * page_value + R(j) grid."""
        if not len(risk):
            return np.zeros(0, dtype=np.int64)
        return np.argmin(np.arange(risk.shape[1]) * page_value + risk, axis=1)

    def _risk(self, slots, maps, written, available, origin):
        """Return P(Y > j) * R(j) per slot and headroom j, inf past its limit.

        Args:
            slots: Answer slots.
            maps: Each slot's map.
            written: Tokens each answer has written.
            available: Answer tokens each slot's pages cover.
            origin: Each answer's first decode position.
        """
        page_tokens = self.decoder.page_tokens
        limits = self.decoder.limits[maps][:, None]
        pages = np.arange(-(-(self.width - 1) // page_tokens) + 1)
        held = np.minimum(limits, available[:, None])
        flat = self.curves.reshape(-1)
        base = (self.slot_row[slots] * self.width)[:, None]
        reach = np.minimum(limits, held + 1 + pages * page_tokens)
        survived = flat[base + np.minimum(limits, written[:, None])]
        overrun = np.divide(flat[base + reach], survived, out=np.zeros(reach.shape),
                            where=survived > 0)
        # an overrun is evicted with the tokens its pages held
        tokens = origin[:, None] + np.minimum(limits, held + pages * page_tokens)
        recover = self.decode_cost.replay_seconds(self.work, tokens,
                                                  self.chunk_tokens)
        risk = overrun * recover
        risk[pages > -(-limits // page_tokens)] = np.inf
        return risk

    def _record(self, running):
        rows = self.slot_row[running]
        remaining = self._remaining(running)
        num_answers = np.bincount(rows, minlength=len(self.curves))
        pages = np.bincount(rows, weights=remaining, minlength=len(self.curves))
        for row in np.flatnonzero(num_answers).tolist():
            self.trace.append((self._refreshes, row // self.num_buckets,
                               row % self.num_buckets, int(num_answers[row]),
                               float(pages[row] / num_answers[row])))

    def _admission_headroom(self, anchor, page_value=None):
        """Headroom pages each of a document's answers would get if admitted.

        Returns:
            The answer slots and their headroom pages, before any answer
            token is written.
        """
        if self._candidate is not None and self._candidate[0] == anchor:
            _, slots, risk = self._candidate
        else:
            slots = self._answer_slots(anchor)
            risk = self._risk(
                slots, self.slot_map[slots], np.zeros(len(slots), dtype=np.int64),
                self.capacity[slots] - self.prompt[slots], self.prompt[slots])
            self._candidate = (anchor, slots, risk)
        return slots, self._argmin(
            risk, self.page_value if page_value is None else page_value)

    def _candidate_headroom(self, anchor):
        """Headroom pages a document's answers would hold, chosen now."""
        if not self.choose_headroom:
            return 0
        slots, chosen = self._admission_headroom(anchor)
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
        self._candidate = None
        slots = self._answer_slots(anchor)
        granted = ((self.reach[slots] - self.capacity[slots])
                   // self.decoder.page_tokens).clip(0)
        self.granted.extend(granted.tolist())
        self.granted_at.extend(zip([self._refreshes] * len(slots),
                                   self.bucket[slots].tolist(), granted.tolist()))
        self.headroom_pages += int(granted.sum())
        branches = np.flatnonzero((self.parent == anchor)
                                  & (np.arange(len(self.prefix)) != anchor))
        self.headroom_pages += int(self.pending_pages[branches].sum())

    def waits(self):
        """Return whether a refused document can wait for pages to free."""
        decoder = self.decoder
        evicted = decoder.paused & (decoder.map >= 0)
        return bool(len(decoder.running()) or evicted.any())
