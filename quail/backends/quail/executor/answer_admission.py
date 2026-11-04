"""Admit map documents by the headroom their answers' survival curves call for."""

import numpy as np

from quail.backends.quail.executor.chunk import decode_groups
from quail.cost.answers import prior_survival, survival_rows
from quail.cost.work import Work


def input_bucket(tokens):
    """Return the input-length bucket: one per doubling of the prefix length."""
    return np.floor(np.log2(np.maximum(np.asarray(tokens, dtype=float), 1.0))
                    ).astype(np.int64)


PRICE_STEPS = 24


def allocate_headroom(risk, budget):
    """Choose each row's headroom pages to keep the summed risk low within a budget.

    Rows take the choices that minimize risk[r, j] + price * j at the
    lowest page price whose pages fit the budget, so a row whose risk
    falls only after several pages takes them all or none. Pages left
    under the budget then go, a whole step at a time, to the row whose
    step saves the most.

    Args:
        risk: Expected rebuild seconds per row and headroom page count,
            inf past a row's limit; column 0 is finite.
        budget: Pages all rows may reserve together.

    Returns:
        Headroom pages per row.
    """
    risk = np.asarray(risk, dtype=float)
    budget = max(0, int(budget))
    chosen = np.argmin(risk, axis=1)
    if int(chosen.sum()) <= budget:
        return chosen
    pages = np.arange(risk.shape[1])
    rows = np.arange(len(risk))
    # at this price no page saves more than it costs, so every row takes 0
    high = float(np.max(risk[:, 0] - risk[rows, chosen]))
    low = high * 1e-12

    def choose(price):
        return np.argmin(risk + price * pages, axis=1)

    chosen = choose(low)
    if int(chosen.sum()) > budget:
        for _ in range(PRICE_STEPS):
            middle = float(np.sqrt(low * high))
            trial = choose(middle)
            used = int(trial.sum())
            if used > budget:
                low = middle
            else:
                high = middle
                if used == budget:
                    break
        chosen = choose(high)
    left = budget - int(chosen.sum())
    while left > 0:
        steps = (pages > chosen[:, None]) & (pages <= chosen[:, None] + left)
        reachable = np.where(steps, risk, np.inf)
        target = np.argmin(reachable, axis=1)
        saved = risk[rows, chosen] - reachable[rows, target]
        row = int(np.argmax(saved))
        if not saved[row] > 0:
            break
        left -= int(target[row] - chosen[row])
        chosen[row] = target[row]
    return chosen


def admission_cost(held_risk, held_pages, candidate_risk, budget):
    """Expected rebuild seconds that admitting a document adds to the held answers.

    The candidate's own least risk is left out: its answers carry it
    whenever they start.

    Args:
        held_risk: Risk grid of the answers already holding pages.
        held_pages: Their headroom without the candidate, within the
            whole budget.
        candidate_risk: Risk grid of the candidate's answers.
        budget: Pages left for headroom once the candidate's prompts
            are placed.

    Returns:
        The added seconds and the headroom of the held answers followed
        by the candidate's.
    """
    risk = np.concatenate((held_risk, candidate_risk))
    chosen = allocate_headroom(risk, budget)
    before = held_risk[np.arange(len(held_risk)), held_pages].sum()
    after = risk[np.arange(len(risk)), chosen].sum()
    least = np.min(candidate_risk, axis=1).sum()
    return max(0.0, float(after - before - least)), chosen


class HeadroomAdmission:
    """Admit a document when the time it saves covers the rebuilds it risks.

    Each answer has its own headroom: pages its answer may grow into
    beyond the pages it holds. The headroom of all answers holding
    pages, the prompts of admitted branches, and the pages evicted
    answers need to come back must fit in the free pages. The pages
    are counted, not allocated: answers still grow one page at a time.

    With choose_headroom False every headroom is zero, so a document is
    admitted when its prompt fits. Otherwise headroom keeps the summed
    P(Y > j) * R(j) of the answers holding pages low within the free
    pages (allocate_headroom):

    - P(Y > j): the chance the answer needs more than j more pages,
      from the Kaplan-Meier curve of its map and input-length bucket,
      conditional on the tokens it has written. A bucket uses the curve
      of all its map's answers until bucket_answers of its own ended.
    - R(j): the SoL seconds rebuilding the answer's whole prefix adds
      when it outgrows j pages and is evicted.

    A waiting document is admitted when its prompts fit and the SoL
    seconds its answers save by joining passes that run anyway are at
    least the rebuild seconds that fitting it adds (admission_cost).
    Its answers' headroom is then re-chosen with everyone else's at the
    next admission. Every refresh rebuilds the curves of maps whose
    answers moved and the headroom of running answers. With trace,
    each refresh also records the running answers' headroom by map and
    bucket.
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
        # prompt pages of admitted documents' branches not yet placed
        self.fixed_pages = 0
        # answers holding pages: slots, risk grids, the pages each holds
        # at headroom 0, the headroom each reserves, and the budget it fits
        grid = -(-(self.width - 1) // page_tokens) + 1
        self.held_slots = np.zeros(0, dtype=np.int64)
        self.held_risk = np.zeros((0, grid))
        self.held_base = np.zeros(0, dtype=np.int64)
        self.held_pages = np.zeros(0, dtype=np.int64)
        self.held_budget = None
        # (anchor, budget, headroom of held answers then its own) of the
        # last document allows() accepted
        self._decision = None
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

    def refresh(self, stages, replays=(), replay_pages=0, free_pages=None):
        """Count held pages; with headroom, rebuild the curves and choose it.

        Args:
            stages: Each anchor's stage: -1 waiting, -2 done.
            replays: (slot, start, end) of evicted answers rebuilt this round.
            replay_pages: Pages evicted answers not yet rebuilding need.
            free_pages: Free KV pages, or None for no limit.
        """
        self.committed = stages >= 0
        self.pending_pages = np.where(stages == -1, self.prompt_pages, 0)
        self.replay_pages = int(replay_pages)
        self._refreshes += 1
        running = self._running()
        self._update_curves()
        roots = self.committed[self.parent]
        self.fixed_pages = int(self.pending_pages[roots].sum())
        self._decision = None
        if self.choose_headroom:
            self._refresh_curves(stages, replays)
            self._hold_running(running, free_pages)
            if self.traced:
                self._record(running)
        self.headroom_pages = self.fixed_pages + int(self._remaining(running).sum())

    def _update_curves(self):
        """Rebuild the curves of maps whose answers moved since the last update."""
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

    def _refresh_curves(self, stages, replays):
        """Recompute the expected answer tokens and the work from the curves."""
        decoder = self.decoder
        slots, maps = self.slots, self.slot_map[self.slots]
        lengths = decoder.lengths[maps, slots]
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

    def eviction_order(self):
        """Rank running answers for eviction, the first evicted first.

        Each answer's remaining tokens come from the curve of its map and
        input-length bucket, the curve its headroom is chosen from.
        """
        policy = None if self.decode_cost is None else self.decode_cost.recompute_cost
        holds = None if self.page_holds is None else self.page_holds()
        self._update_curves()
        return self.decoder.eviction_order(None, policy, holds,
                                           remaining=self._remaining_tokens())

    def _remaining_tokens(self):
        """Per slot, E[L - t | L > t] from its bucket curve, t its written tokens."""
        decoder = self.decoder
        remaining = np.zeros(len(self.prefix))
        running = self._running()
        if not len(running):
            return remaining
        curves = self.curves[self.slot_row[running]]
        tails = np.cumsum(curves[:, ::-1], axis=1)[:, ::-1]
        written = decoder.written[decoder.map[running], running]
        index = np.minimum(written, self.width - 1)
        rows = np.arange(len(running))
        survived = curves[rows, index]
        remaining[running] = np.divide(tails[rows, index], survived,
                                       out=np.zeros(len(running)), where=survived > 0)
        return remaining

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

    def _hold_running(self, running, free_pages):
        """Price running answers' risk; choose their headroom within the free pages."""
        decoder = self.decoder
        maps = decoder.map[running]
        self.held_slots = running
        self.held_risk = self._risk(running, maps, decoder.written[maps, running],
                                    self._available(running), decoder.origin[running])
        self.held_base = decoder.kv.num_pages[running].astype(np.int64)
        if free_pages is None:
            self._hold(np.argmin(self.held_risk, axis=1), None)
        else:
            budget = free_pages - self.fixed_pages - self.replay_pages
            self._hold(allocate_headroom(self.held_risk, budget), budget)

    def _hold(self, chosen, budget):
        """Reserve the chosen headroom pages for the held answers."""
        self.held_pages = np.asarray(chosen, dtype=np.int64)
        self.held_budget = budget
        self.reach[self.held_slots] = ((self.held_base + self.held_pages)
                                       * self.decoder.page_tokens)

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

    def _admission_risk(self, anchor):
        """The answer slots of a waiting document and their R(j) grid."""
        if self._candidate is not None and self._candidate[0] == anchor:
            _, slots, risk = self._candidate
        else:
            slots = self._answer_slots(anchor)
            risk = self._risk(
                slots, self.slot_map[slots], np.zeros(len(slots), dtype=np.int64),
                self.capacity[slots] - self.prompt[slots], self.prompt[slots])
            self._candidate = (anchor, slots, risk)
        return slots, risk

    def _branch_pages(self, anchor):
        """Prompt pages a document's branches will place after it is admitted."""
        branches = (self.parent == anchor) & (np.arange(len(self.prefix)) != anchor)
        return int(self.pending_pages[branches].sum())

    def allows(self, anchor, free_pages, prompt_pages):
        """Admit when the prompts fit and the time saved covers the rebuilds added.

        Args:
            anchor: The waiting document, or a branch of an admitted one.
            free_pages: Free KV pages.
            prompt_pages: Pages the document's prompt takes now.
        """
        if self.committed[self.parent[anchor]]:
            return True
        self._decision = None
        if not self.choose_headroom:
            return free_pages - self.headroom_pages - self.replay_pages >= prompt_pages
        budget = free_pages - self.fixed_pages - self.replay_pages
        need = prompt_pages + self._branch_pages(anchor)
        if need > budget:
            return False
        if self.held_budget != budget:
            self._hold(allocate_headroom(self.held_risk, budget), budget)
            self.headroom_pages = self.fixed_pages + int(self.held_pages.sum())
        _, risk = self._admission_risk(anchor)
        extra, chosen = admission_cost(self.held_risk, self.held_pages, risk,
                                       budget - need)
        added = self._added_work(anchor)
        # with nothing running, price joining a pass that holds one such answer
        base = self.work if self.work.tokens else added
        gain = self.decode_cost.gain(base, added, self.expected_tokens[anchor])
        if extra > gain * (1 + 1e-9):
            return False
        self._decision = (anchor, budget - need, chosen)
        return True

    def admit(self, anchor):
        """Count an admitted document's headroom, or a branch's allocated prompt."""
        if self.committed[self.parent[anchor]]:
            pages = int(self.pending_pages[anchor])
            self.headroom_pages -= pages
            self.fixed_pages -= pages
            self.pending_pages[anchor] = 0
            return
        self.committed[anchor] = True
        slots = self._answer_slots(anchor)
        if self.choose_headroom:
            self._hold_admitted(anchor, slots)
        granted = ((self.reach[slots] - self.capacity[slots])
                   // self.decoder.page_tokens).clip(0)
        self.work = self.work + self._added_work(anchor)
        self._candidate = None
        self.granted.extend(granted.tolist())
        self.granted_at.extend(zip([self._refreshes] * len(slots),
                                   self.bucket[slots].tolist(), granted.tolist()))
        self.fixed_pages += self._branch_pages(anchor)
        if self.choose_headroom:
            self.headroom_pages = self.fixed_pages + int(self.held_pages.sum())
        else:
            self.headroom_pages += int(granted.sum()) + self._branch_pages(anchor)

    def _hold_admitted(self, anchor, slots):
        """Add an admitted document's answers to the held ones with allows()'s headroom.

        A document admitted without allows() accepting it, because no
        answer could free pages, starts with no headroom.
        """
        _, risk = self._admission_risk(anchor)
        decision = self._decision
        self._decision = None
        self.held_slots = np.r_[self.held_slots, slots]
        self.held_risk = np.concatenate((self.held_risk, risk))
        self.held_base = np.r_[self.held_base,
                               self.capacity[slots] // self.decoder.page_tokens]
        if decision is not None and decision[0] == anchor:
            self._hold(decision[2], decision[1])
        else:
            self._hold(np.r_[self.held_pages, np.zeros(len(slots), np.int64)], None)

    def waits(self):
        """Return whether a refused document can wait for pages to free."""
        decoder = self.decoder
        evicted = decoder.paused & (decoder.map >= 0)
        return bool(len(decoder.running()) or evicted.any())
