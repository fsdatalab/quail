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
    return _allocate(risk, budget)[0]


def _hull_pages(values, first):
    """Page counts on the lower convex hull of the points (first + i, values[i])."""
    hull = []
    for i, value in enumerate(values.tolist()):
        while len(hull) >= 2:
            (a, risk_a), (b, risk_b) = hull[-2], hull[-1]
            if (risk_b - risk_a) * (i - a) < (value - risk_a) * (b - a):
                break
            hull.pop()
        hull.append((i, value))
    return [first + i for i, _ in hull]


def _allocate(risk, budget, price=None):
    """allocate_headroom, searching for the page price from a nearby one.

    The search brackets the price by doubling or halving from price,
    then halves the bracket. A row's choice only falls as the price
    rises, so a row whose choices at both ends agree is settled, and
    once one row is left its choice is read off its lower convex hull.

    Args:
        risk: As allocate_headroom.
        budget: As allocate_headroom.
        price: A page price near the one sought, or None.

    Returns:
        Headroom pages per row, and a page price at which they fit: the
        given price when every row's least risk fits.
    """
    risk = np.asarray(risk, dtype=float)
    budget = max(0, int(budget))
    chosen = np.argmin(risk, axis=1)
    if int(chosen.sum()) <= budget:
        return chosen, price
    pages = np.arange(risk.shape[1])
    rows = np.arange(len(risk))
    # at this price no page saves more than it costs, so every row takes 0
    high = float(np.max(risk[:, 0] - risk[rows, chosen]))
    floor = high * 1e-12

    def choose(at):
        return np.argmin(risk + at * pages, axis=1)

    # each row's choice at the bracket's low price (over budget) and high price
    upper = lower = None
    if price is not None and floor < price < high:
        at = price
        trial = choose(at)
        if int(trial.sum()) > budget:
            low, upper = at, trial
            while at < high:
                at = min(high, 2 * at)
                trial = choose(at)
                if int(trial.sum()) <= budget:
                    high, lower = at, trial
                    break
                low, upper = at, trial
        else:
            high, lower = at, trial
            while at > floor:
                at = max(floor, at / 2)
                trial = choose(at)
                if int(trial.sum()) > budget:
                    low, upper = at, trial
                    break
                high, lower = at, trial
    if upper is None:
        low = floor
        upper = choose(low)
        if int(upper.sum()) <= budget:
            chosen, upper, high = upper, None, low
    if upper is not None:
        if lower is None:
            lower = np.zeros_like(upper)
        for _ in range(PRICE_STEPS):
            unsettled = np.flatnonzero(upper > lower)
            if len(unsettled) == 1:
                (row,) = unsettled
                room = budget - int(lower.sum()) + int(lower[row])
                first, last = int(lower[row]), int(upper[row])
                hull = _hull_pages(risk[row, first:last + 1], first)
                lower[row] = max(j for j in hull if j <= room)
                break
            if not len(unsettled):
                break
            width = int(upper[unsettled].max()) + 1
            middle = float(np.sqrt(low * high))
            trial = np.argmin(risk[unsettled, :width] + middle * pages[:width], axis=1)
            used = int(lower.sum()) - int(lower[unsettled].sum()) + int(trial.sum())
            if used > budget:
                low = middle
                upper[unsettled] = trial
            else:
                high = middle
                lower[unsettled] = trial
                if used == budget:
                    break
        chosen = lower
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
    return chosen, high


def admission_cost(held_risk, held_pages, candidate_risk, budget, price=None):
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
        price: A page price near the one the budget sets, or None.

    Returns:
        The added seconds and the headroom of the held answers followed
        by the candidate's.
    """
    risk = np.concatenate((held_risk, candidate_risk))
    chosen, _ = _allocate(risk, budget, price)
    before = held_risk[np.arange(len(held_risk)), held_pages].sum()
    after = risk[np.arange(len(risk)), chosen].sum()
    least = np.min(candidate_risk, axis=1).sum()
    return max(0.0, float(after - before - least)), chosen


class HeadroomAdmission:
    """Admit a document when the time it saves covers the rebuilds it risks.

    A document's answers are admitted together when they fit, and one at
    a time when they do not. allows() first prices every answer of the
    anchor's document not yet admitted; if that is refused, it prices
    the anchor's answer alone. Each branch, an answer that borrows the
    document's KV, then waits in the queue with its prompt unplaced
    while the document's KV stays held for it.

    Each answer has its own headroom: pages its answer may grow into
    beyond the pages it holds. The headroom of all answers holding
    pages, the prompts of admitted branches, and the pages evicted
    answers need to come back must fit in the free pages. The pages
    are counted, not allocated: answers still grow one page at a time.

    With choose_headroom False every headroom is zero, so answers are
    admitted when their prompts fit. Otherwise headroom keeps the summed
    P(Y > j) * R(j) of the answers holding pages low within the free
    pages (allocate_headroom):

    - P(Y > j): the chance the answer needs more than j more pages,
      from the Kaplan-Meier curve of its map and input-length bucket,
      conditional on the tokens it has written. A bucket uses the curve
      of all its map's answers until bucket_answers of its own ended.
    - R(j): the SoL seconds rebuilding the answer's whole prefix adds
      when it outgrows j pages and is evicted.

    Waiting answers are admitted when their prompts fit and the SoL
    seconds they save by joining passes that run anyway are at least the
    rebuild seconds that fitting them adds (admission_cost). Their decode
    work counts the document's shared KV once, and not at all when its
    running answers already read it. Their headroom is then re-chosen
    with everyone else's at the next admission. Every refresh rebuilds
    the curves of maps whose answers moved and the headroom of running
    answers.

    With trace, each refresh also records the running answers' headroom
    by map and bucket.
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
        self.decode_cost = decode_cost
        self.page_holds = page_holds
        # per document, the most tokens any of its branches borrows from it
        borrowed = np.where(self.parent == np.arange(len(self.prefix)), 0, shared)
        self.lead = np.zeros(len(self.prefix), dtype=np.int64)
        np.maximum.at(self.lead, self.parent, borrowed)
        self.choose_headroom = choose_headroom and decode_cost is not None
        # each anchor's dependents, the anchor first: order[start[a]:start[a + 1]]
        order = np.argsort(self.parent, kind="stable")
        self.child_order = order
        self.child_start = np.searchsorted(self.parent[order],
                                           np.arange(len(self.prefix) + 1))
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
        # per slot, the tokens it was last counted at and whether it had
        # ended; per map and per curve row, those counts by token
        self._observed = np.zeros(len(self.prefix), dtype=np.int64)
        self._ended = np.zeros(len(self.prefix), dtype=bool)
        self._watched = np.zeros(0, dtype=np.int64)
        # anchors whose answers allows() priced and admit() counted
        self.committed = np.zeros(len(self.prefix), dtype=bool)
        self.stages = np.full(len(self.prefix), -1)
        # committed branches whose prompts are not placed yet
        self.unplaced = set()
        # their prompt pages
        self.fixed_pages = 0
        # per slot, the KV position its pages and headroom reach
        self.reach = np.zeros(len(self.prefix), dtype=np.int64)
        self.headroom_pages = 0
        self.replay_pages = 0
        # answers holding pages: slots, risk grids, the pages each holds
        # at headroom 0, the headroom each reserves, and the budget it fits
        grid = -(-(self.width - 1) // page_tokens) + 1
        self.held_slots = np.zeros(0, dtype=np.int64)
        self.held_risk = np.zeros((0, grid))
        self.held_base = np.zeros(0, dtype=np.int64)
        self.held_pages = np.zeros(0, dtype=np.int64)
        self.held_budget = None
        # the page price of the last headroom chosen, where the next search starts
        self.price = None
        # (anchor, the anchors it commits, budget, headroom of held answers
        # then theirs) of the last anchor allows() accepted
        self._decision = None
        self.work = Work()
        # per map, its mean answer tokens; per considered set of answers,
        # their expected tokens and decode work, until the next refresh
        self._expected = np.zeros(len(decoder.limits))
        self._added = {}
        # (anchors, their answer slots, their R(j) grid) for the last
        # priced answers, valid until the work changes
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
        num_maps, num_rows = len(decoder.limits), len(self.row_limit)
        self.curves = np.zeros((num_rows, self.width))
        self.whole = np.zeros((num_maps, self.width))
        self.map_counts = np.zeros((num_maps, self.width), dtype=np.int64)
        self.map_events = np.zeros((num_maps, self.width), dtype=np.int64)
        self.row_counts = np.zeros((num_rows, self.width), dtype=np.int64)
        self.row_events = np.zeros((num_rows, self.width), dtype=np.int64)
        self.num_ended = np.zeros(num_rows, dtype=np.int64)
        decoder.ended_log = []
        slots = self.slots
        self._count(slots, *self._observation(slots), sign=1)

    @property
    def bucket_curves(self):
        """The curve of each (map, bucket) with answers, cut at its map's limit."""
        if self.curves is None:
            return {}
        return {(int(row // self.num_buckets), int(row % self.num_buckets)):
                self.curves[row, :self.row_limit[row] + 1] for row in self.rows}

    def refresh(self, stages, replays=(), replay_pages=0, free_pages=None):
        """Count held pages; with headroom, rebuild the curves and choose it.

        Reads the answers being written, found in one pass over the
        decoder's slots, and the stages of branches waiting to place
        their prompts.

        Args:
            stages: Each anchor's stage, indexable by anchor: -1 waiting,
                -2 done. Kept until the next refresh.
            replays: (slot, start, end) of evicted answers rebuilt this round.
            replay_pages: Pages evicted answers not yet rebuilding need.
            free_pages: Free KV pages, or None for no limit.
        """
        self.stages = stages
        self.replay_pages = int(replay_pages)
        self._refreshes += 1
        writing = np.flatnonzero(self.decoder.map >= 0)
        running = writing[~self.decoder.paused[writing]]
        self._update_curves(writing)
        self.unplaced = {b for b in self.unplaced if stages[b] == -1}
        self.fixed_pages = int(sum(int(self.prompt_pages[b]) for b in self.unplaced))
        self._decision = None
        self._candidate = None
        self._added = {}
        if self.choose_headroom:
            self._expected = self.whole.sum(axis=1)
            self._refresh_work(writing, replays)
            self._hold_running(running, free_pages)
            if self.traced:
                self._record(running)
        self.headroom_pages = self.fixed_pages + int(self._remaining(running).sum())

    def _observation(self, slots):
        """Each slot's last token count and whether its answer ended."""
        decoder = self.decoder
        maps = self.slot_map[slots]
        lengths = decoder.lengths[maps, slots]
        ended = lengths >= 0
        return np.where(ended, lengths, decoder.written[maps, slots]), ended

    def _count(self, slots, observed, ended, sign):
        """Add (sign 1) or remove (sign -1) slots' observations from the counts."""
        maps, rows = self.slot_map[slots], self.slot_row[slots]
        np.add.at(self.map_counts, (maps, observed), sign)
        np.add.at(self.map_events, (maps[ended], observed[ended]), sign)
        np.add.at(self.row_counts, (rows, observed), sign)
        np.add.at(self.row_events, (rows[ended], observed[ended]), sign)
        np.add.at(self.num_ended, rows[ended], sign)
        if sign > 0:
            self._observed[slots] = observed
            self._ended[slots] = ended

    def _update_curves(self, writing=None):
        """Rebuild the curves of maps whose answers moved since the last update.

        Only answers written now, at the last update, or ended since
        can have moved.

        Args:
            writing: Slots whose answers are being written, or None to find them.
        """
        decoder = self.decoder
        first = self.slots is None
        if first:
            self._bind()
        if writing is None:
            writing = np.flatnonzero(decoder.map >= 0)
        watch = np.unique(np.concatenate([writing, self._watched, *decoder.ended_log]))
        decoder.ended_log.clear()
        self._watched = writing
        watch = watch[self.slot_map[watch] >= 0]
        observed, ended = self._observation(watch)
        moved = (observed != self._observed[watch]) | (ended != self._ended[watch])
        changed = np.zeros(len(decoder.limits), dtype=bool)
        if moved.any():
            slots = watch[moved]
            self._count(slots, self._observed[slots], self._ended[slots], sign=-1)
            self._count(slots, observed[moved], ended[moved], sign=1)
            changed[self.slot_map[slots]] = True
        if first:
            changed[:] = True
        if changed.any():
            self._rebuild_curves(changed)

    def _rebuild_curves(self, changed):
        """Rebuild the whole-map and bucket curves of the changed maps."""
        mine = np.flatnonzero(changed)
        self.whole[mine] = survival_rows(self.map_counts[mine], self.map_events[mine],
                                         self.prior[mine],
                                         self.decoder.limits[mine])
        rows = self.rows[changed[self.rows // self.num_buckets]]
        self.curves[rows] = self.whole[rows // self.num_buckets]
        # a bucket with few ended answers keeps its map's curve
        rows = rows[self.num_ended[rows] >= self.bucket_answers]
        if len(rows):
            self.curves[rows] = survival_rows(self.row_counts[rows],
                                              self.row_events[rows],
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

    def _refresh_work(self, writing, replays):
        """Count the decode work of the answers running this round."""
        cost = self.decode_cost
        self.work = Work()
        if cost is None:
            return
        decoder = self.decoder
        running = decoder.running(writing)
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

    def _added_work(self, members):
        """The answers' expected tokens and the decode work they add.

        The answers of one document read its shared KV once, and not at
        all when its running answers already read it.

        Args:
            members: Waiting anchors of one document.
        """
        if members not in self._added:
            decoder = self.decoder
            slots = self._answer_slots(members)
            maps = self.slot_map[slots]
            slots = slots[(decoder.lengths[maps, slots] < 0) & ~decoder.paused[slots]]
            expected = self._expected[self.slot_map[slots]]
            work = Work()
            if len(slots) and self.decode_cost is not None:
                covered = np.zeros(len(slots), dtype=np.int64)
                if decoder.share:
                    document = self.parent[members[0]]
                    children = self._children(document)
                    running = (decoder.map[children] >= 0) & ~decoder.paused[children]
                    covered[0 if running.any() else 1:] = self.lead[document]
                work = self.decode_cost.row_work(
                    self.prompt[slots] + expected / 2, covered=covered).total()
            self._added[members] = (float(expected.sum()), work)
        return self._added[members]

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

    def _children(self, anchor):
        """The anchor, if a root, and its branches."""
        return self.child_order[self.child_start[anchor]:self.child_start[anchor + 1]]

    def _answer_slots(self, members):
        """The answer slots among the anchors."""
        slots = np.asarray(members, dtype=np.int64)
        return slots[self.slot_map[slots] >= 0]

    def _waiting_members(self, anchor):
        """The anchor, then its document's other anchors not committed or placed."""
        return (anchor, *(b for b in self._children(self.parent[anchor]).tolist()
                          if b != anchor and not self.committed[b]
                          and self.stages[b] == -1))

    def order_waiting(self, waiting):
        """Put each document's waiting branches expected-longest first, in place.

        Documents keep the order of their first waiting branch. A
        branch's expected tokens come from its map and bucket curve.

        Args:
            waiting: The scheduler's deque of released branches.
        """
        if len(waiting) < 2 or self.curves is None:
            return
        anchors = np.fromiter(waiting, dtype=np.int64, count=len(waiting))
        _, inverse = np.unique(self.parent[anchors], return_inverse=True)
        first = np.full(inverse.max() + 1, len(anchors))
        np.minimum.at(first, inverse, np.arange(len(anchors)))
        rows = self.slot_row[anchors]
        expected = np.where(rows >= 0, self.curves[rows].sum(axis=1), 0.0)
        order = np.lexsort((-expected, first[inverse]))
        waiting.clear()
        waiting.extend(anchors[order].tolist())

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
            self._hold(self._allocate_held(budget), budget)

    def _allocate_held(self, budget):
        """Headroom of the held answers within the budget, from the last price."""
        chosen, self.price = _allocate(self.held_risk, budget, self.price)
        return chosen

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

    def _admission_risk(self, members):
        """The answer slots of waiting anchors and their R(j) grid."""
        if self._candidate is not None and self._candidate[0] == members:
            _, slots, risk = self._candidate
        else:
            slots = self._answer_slots(members)
            risk = self._risk(
                slots, self.slot_map[slots], np.zeros(len(slots), dtype=np.int64),
                self.capacity[slots] - self.prompt[slots], self.prompt[slots])
            self._candidate = (members, slots, risk)
        return slots, risk

    def allows(self, anchor, free_pages, prompt_pages):
        """Admit when the prompts fit and the time saved covers the rebuilds added.

        Prices the anchor with its document's other waiting anchors
        first, then the anchor alone.

        Args:
            anchor: A waiting document, or a waiting branch of one.
            free_pages: Free KV pages.
            prompt_pages: Pages the anchor's prompt takes now.
        """
        if self.committed[anchor]:
            return True
        self._decision = None
        group = self._waiting_members(anchor)
        choices = (group, (anchor,)) if len(group) > 1 else (group,)
        if not self.choose_headroom:
            room = free_pages - self.headroom_pages - self.replay_pages
            for members in choices:
                if room >= prompt_pages + self._branch_pages(members):
                    self._decision = (anchor, members)
                    return True
            return False
        budget = free_pages - self.fixed_pages - self.replay_pages
        held = None
        for members in choices:
            need = prompt_pages + self._branch_pages(members)
            if need > budget:
                continue
            if self.held_budget != budget:
                self._hold(self._allocate_held(budget), budget)
                self.headroom_pages = self.fixed_pages + int(self.held_pages.sum())
            _, risk = self._admission_risk(members)
            expected, added = self._added_work(members)
            # with nothing running, price joining a pass that holds such answers
            base = self.work if self.work.tokens else added
            gain = self.decode_cost.gain(base, added, expected)
            if held is None:
                held = self._held_bound()
            if self._least_extra(held, risk, budget - need) > gain * (1 + 1e-9):
                continue
            extra, chosen = admission_cost(self.held_risk, self.held_pages, risk,
                                           budget - need, self.price)
            if extra <= gain * (1 + 1e-9):
                self._decision = (anchor, members, budget - need, chosen)
                return True
        return False

    def _held_bound(self):
        """The held answers' summed least risk plus page cost at the last price.

        Returns:
            That sum, and the held answers' risk at their chosen headroom.
        """
        rows = np.arange(len(self.held_risk))
        before = float(self.held_risk[rows, self.held_pages].sum())
        if not self.price:
            return 0.0, before
        pages = np.arange(self.held_risk.shape[1])
        return float(np.min(self.held_risk + self.price * pages, axis=1).sum()), before

    def _least_extra(self, held, risk, budget):
        """A lower bound on the seconds admission_cost would add.

        At any page price p, headroom within a budget risks at least the
        sum over rows of min_j (risk[r, j] + p * j), less p * budget.

        Args:
            held: _held_bound() for the held answers.
            risk: Risk grid of the candidate answers.
            budget: As admission_cost.
        """
        if not self.price:
            return 0.0
        bound, before = held
        pages = np.arange(risk.shape[1])
        candidate = float(np.min(risk + self.price * pages, axis=1).sum())
        least = float(np.min(risk, axis=1).sum())
        return bound + candidate - self.price * budget - before - least

    def _branch_pages(self, members):
        """Prompt pages the anchors after the first will place later."""
        return int(sum(int(self.prompt_pages[b]) for b in members[1:]))

    def admit(self, anchor):
        """Count an admitted anchor's answers and headroom, or a placed prompt."""
        if anchor in self.unplaced:
            self.unplaced.discard(anchor)
            pages = int(self.prompt_pages[anchor])
            self.headroom_pages -= pages
            self.fixed_pages -= pages
            return
        decision = self._decision
        members = (decision[1] if decision is not None and decision[0] == anchor
                   else (anchor,))
        self.committed[list(members)] = True
        slots = self._answer_slots(members)
        if self.choose_headroom:
            self._hold_admitted(members, slots)
        self._decision = None
        granted = ((self.reach[slots] - self.capacity[slots])
                   // self.decoder.page_tokens).clip(0)
        self.work = self.work + self._added_work(members)[1]
        self._candidate = None
        self.granted.extend(granted.tolist())
        self.granted_at.extend(zip([self._refreshes] * len(slots),
                                   self.bucket[slots].tolist(), granted.tolist()))
        branch_pages = self._branch_pages(members)
        self.unplaced.update(members[1:])
        self.fixed_pages += branch_pages
        if self.choose_headroom:
            self.headroom_pages = self.fixed_pages + int(self.held_pages.sum())
        else:
            self.headroom_pages += int(granted.sum()) + branch_pages

    def _hold_admitted(self, members, slots):
        """Add admitted answers to the held ones with allows()'s headroom.

        Answers admitted without allows() accepting them, because no
        answer could free pages, start with no headroom.
        """
        _, risk = self._admission_risk(members)
        decision = self._decision
        self.held_slots = np.r_[self.held_slots, slots]
        self.held_risk = np.concatenate((self.held_risk, risk))
        self.held_base = np.r_[self.held_base,
                               self.capacity[slots] // self.decoder.page_tokens]
        if decision is not None and decision[1] == members:
            self._hold(decision[3], decision[2])
        else:
            self._hold(np.r_[self.held_pages, np.zeros(len(slots), np.int64)], None)

    def waits(self):
        """Return whether a refused document can wait for pages to free."""
        decoder = self.decoder
        evicted = decoder.paused & (decoder.map >= 0)
        return bool(len(decoder.running()) or evicted.any())
