"""Chunk packing and admission - no GPU, no torch, unit-tested.

- JoinAdmission: continuous anchor admission for joins. Continuing
  partner streams pack first, then anchors starting their next stage,
  then fresh anchors whose pages fit the free list.
- FilterAdmission: continuous admission for filter chains. Survivors
  pack before fresh admissions; fresh documents admit when their
  page-rounded tokens fit the free list.

Length units are tokens. Suffixes are atomic and never split across
chunks.
"""

from array import array
from bisect import bisect_right
from collections import deque


def orient(mean_left_tokens, mean_right_tokens):
    """Which side anchors: the longer one.

    Anchor tokens are paid once per document; partner tokens are paid
    once per pair.
    """
    return "left" if mean_left_tokens >= mean_right_tokens else "right"


def gate(answer_rows):
    """Anchors that survive a conjunctive stage: any TRUE in the row.

    answer_rows: dict anchor_index -> iterable of 0/1 answers.
    Returns the sorted surviving anchor indices.
    """
    return sorted(a for a, row in answer_rows.items() if any(row))


def matches(answer_rows):
    """anchor_index -> sorted list of partner indices answered TRUE."""
    return {a: sorted(i for i, v in enumerate(row) if v)
            for a, row in answer_rows.items()}


def assemble(ans1_rows, ans2_rows):
    """Output triples from two pairwise join stages sharing one anchor.

    ans1_rows: b -> row of 0/1 over A (stage 1, anchored on b).
    ans2_rows: b -> row of 0/1 over C, present only for gated
    survivors.
    """
    m1, m2 = matches(ans1_rows), matches(ans2_rows)
    out = []
    for b in sorted(ans2_rows):
        for a in m1.get(b, ()):
            for c in m2[b]:
                out.append((a, b, c))
    return sorted(out)


def brute_force_triples(ans1_rows, ans2_rows):
    """Nested-loop reference over recorded answers without gating."""
    out = []
    for b, row1 in sorted(ans1_rows.items()):
        for a, v1 in enumerate(row1):
            if v1:
                for c, v2 in enumerate(ans2_rows[b]):
                    if v2:
                        out.append((a, b, c))
    return sorted(out)


# --------------------------------------------- continuous admission


class _CompactQueue:
    """Keep ordered document positions in ranges or integer arrays."""

    def __init__(self, count: int):
        self._pieces = deque()
        self._length = 0
        if count:
            self._pieces.append([range(count), 0])
            self._length = count

    def __bool__(self):
        return bool(self._length)

    def popleft(self) -> int:
        if not self._pieces:
            raise IndexError("pop from an empty document queue")
        values, position = self._pieces[0]
        value = values[position]
        position += 1
        self._length -= 1
        if position == len(values):
            self._pieces.popleft()
        else:
            self._pieces[0][1] = position
        return value

    def appendleft(self, value: int) -> None:
        self.prepend(array("I", [value]))

    def prepend(self, values) -> None:
        if values:
            self._pieces.appendleft([values, 0])
            self._length += len(values)

def pages_for(tokens: int, page_tokens: int) -> int:
    """Pages needed for `tokens` rows (suffix KV is never paged)."""
    return -(-tokens // page_tokens)


# --------------------------------------------- join admission

_DONE = -2      # the anchor's last stage answered, or it failed a gate
_STRANDED = -3  # advanced into a stage with no partners: stays resident


class JoinAdmission:
    """Join scheduler: continuous anchor admission, stages mixed in one chunk.

    Args:
        prefix_tokens: Per-anchor prefix token counts.
        stage_suffixes: Per stage, the partner suffix token counts.
            Every anchor streams the same partner list at a stage.
        chunk_budget: Tokens per forward pass.
        arena_pages: Pages in the KV arena.
        page_tokens: Tokens per arena page.
        frame_tokens: Per stage, tokens of the framing written into
            the anchor's KV ahead of its first suffix.
        resident: anchor -> pages already held. A resident anchor's
            prefix KV is in the arena, so it packs no prefix tokens.

    Each chunk fills in priority order: partner streams cut by the
    previous chunk, then anchors starting their next stage, then
    fresh anchors whose pages fit the free list. Pages are granted in
    queue order; chunk room may be skipped. An anchor advances to its
    next stage once its whole stream at the current stage is launched
    and one partner has answered TRUE; the remaining answers fill in
    while the next stage runs. It drops out when every partner
    answered FALSE.
    """

    def __init__(self, prefix_tokens, stage_suffixes, chunk_budget,
                 arena_pages, page_tokens, frame_tokens=None,
                 resident=None):
        self.prefix = []
        self.stages = [list(s) for s in stage_suffixes]
        self.frames = (list(frame_tokens) if frame_tokens
                       else [0] * len(self.stages))
        if len(self.frames) != len(self.stages):
            raise ValueError("frame_tokens must match stage_suffixes")
        if not self.stages or not self.stages[0]:
            raise ValueError("the first stage needs partners")
        self.chunk_budget = chunk_budget
        self.arena_pages = arena_pages
        self.page_tokens = page_tokens
        k = len(self.stages)
        self._cum = []
        for j, (lens, frame) in enumerate(zip(self.stages, self.frames)):
            cum = [0]
            for t in lens:
                cum.append(cum[-1] + t)
            self._cum.append(cum)
            if lens and frame + max(lens) > chunk_budget:
                raise ValueError(
                    f"stage {j}: a {frame + max(lens)}-token partner "
                    f"exceeds the {chunk_budget}-token chunk budget "
                    f"(suffixes are atomic)")
        self._extra = max(self.frames)
        self._first = self.frames[0] + self.stages[0][0]
        self._page_cost = []
        self._carried = []
        self._min_fresh = float("inf")
        self._zero_cost = 0
        self.pending = deque()
        self.ready = deque()       # placed anchors with partners left
        self._stage = []           # current stage, or _DONE/_STRANDED
        self._next = []            # next partner index at the stage
        self._true = []
        self.in_flight = 0         # launched groups not yet reported
        self.blocked_pages = 0
        self.answers = [dict() for _ in range(k)]
        resident = dict(resident or {})
        for a, prefix in enumerate(prefix_tokens):
            self._register(prefix, resident.get(a))
        # resident anchors first: they cost no prefix tokens and few
        # or no pages, so they never wait behind a page-blocked anchor
        n = len(self.prefix)
        self.pending.extend(a for a in range(n) if a in resident)
        self.pending.extend(a for a in range(n) if a not in resident)

    def _register(self, prefix, resident_pages):
        """Record one anchor's costs; returns its index."""
        a = len(self.prefix)
        need = pages_for(prefix + self._extra, self.page_tokens)
        if resident_pages is not None:
            need = max(0, need - resident_pages)
        elif need > self.arena_pages:
            raise ValueError(
                f"anchor {a} needs {need} KV pages; the arena "
                f"holds {self.arena_pages}")
        elif prefix + self._first > self.chunk_budget:
            raise ValueError(
                f"anchor {a}: prefix {prefix} tokens leaves no "
                f"room for a partner in a {self.chunk_budget}-token "
                f"chunk")
        carried = 0 if resident_pages is not None else prefix
        self.prefix.append(prefix)
        self._page_cost.append(need)
        self._carried.append(carried)
        self._min_fresh = min(self._min_fresh, carried + self._first)
        if not need:
            self._zero_cost += 1
        self._stage.append(-1)
        self._next.append(0)
        self._true.append([False] * len(self.stages))
        return a

    def admit(self, prefix_tokens, resident_pages=None):
        """Queue one more anchor behind the pending ones; returns its index.

        resident_pages says the anchor's prefix KV is already in the
        arena on that many pages, so it packs no prefix tokens.
        """
        a = self._register(prefix_tokens, resident_pages)
        self.pending.append(a)
        return a

    def buildable_tokens(self):
        """Tokens the next chunk could pack, capped at the chunk budget.

        Counts every placed anchor's remaining stream and every pending
        anchor's first chunk. Chunk room, not pages: a page-blocked
        anchor still counts.
        """
        total = 0
        for a in self.ready:
            j, i = self._stage[a], self._next[a]
            total += (self.frames[j] if i == 0 else 0) \
                + self._cum[j][-1] - self._cum[j][i]
            if total >= self.chunk_budget:
                return self.chunk_budget
        for a in self.pending:
            total += self._carried[a] + self.frames[0] + self._cum[0][-1]
            if total >= self.chunk_budget:
                return self.chunk_budget
        return total

    # ---- chunk building ------------------------------------------------

    def _first_cost(self, j, i):
        return self.stages[j][i] + (self.frames[j] if i == 0 else 0)

    def _take(self, j, i, room):
        """(end, tokens): the longest partner run from i that fits."""
        cum = self._cum[j]
        frame = self.frames[j] if i == 0 else 0
        end = bisect_right(cum, cum[i] + room - frame) - 1
        return end, frame + cum[end] - cum[i]

    def _launch(self, a, j, end):
        """Record a launched group; True when the stream continues."""
        self._next[a] = end
        self.in_flight += 1
        return end < len(self.stages[j])

    def next_chunk(self, free_pages):
        """Groups for the next chunk: [(anchor, stage, start, end, carried)].

        carried means the anchor's prefix tokens are packed and its KV
        written to its pages. free_pages is the arena's free list;
        fresh anchors admit against it. Returns [] when nothing is
        buildable.
        """
        self.blocked_pages = 0
        room = self.chunk_budget
        groups = []
        continued = []
        # 1) placed anchors: cut streams (front) and next-stage starts
        for _ in range(len(self.ready)):
            a = self.ready.popleft()
            j, i = self._stage[a], self._next[a]
            if self._first_cost(j, i) > room:
                self.ready.appendleft(a)    # FIFO; chunk nearly full
                break
            end, tokens = self._take(j, i, room)
            groups.append((a, j, i, end, False))
            room -= tokens
            if self._launch(a, j, end):
                continued.append(a)
        # 2) fresh anchors: pages in queue order, chunk room may skip
        held = []
        blocked = False
        while self.pending and room >= self._min_fresh:
            a = self.pending.popleft()
            need = self._page_cost[a]
            if blocked and need:
                held.append(a)
                if not self._zero_cost:
                    break
                continue
            if need > free_pages:
                blocked = True
                self.blocked_pages = need - free_pages
                held.append(a)
                if not self._zero_cost:
                    break
                continue
            carried = self._carried[a]
            if carried + self._first_cost(0, 0) > room:
                held.append(a)      # chunk room only; retry next chunk
                continue
            end, tokens = self._take(0, 0, room - carried)
            groups.append((a, 0, 0, end, carried > 0))
            room -= carried + tokens
            free_pages -= need
            if not need:
                self._zero_cost -= 1
            self._stage[a] = 0
            if self._launch(a, 0, end):
                continued.append(a)
        self.pending.extendleft(reversed(held))
        self.ready.extendleft(reversed(continued))
        return groups

    # ---- gating --------------------------------------------------------

    def report(self, a, j, start, end, bits):
        """Record one group's answers.

        Returns events: ("dropped", a) when every partner at a stage
        before the last answered FALSE, so the anchor's pages can go;
        ("finished", a) when its last-stage row is complete.
        """
        row = self.answers[j].setdefault(a, [])
        if len(row) != start or len(bits) != end - start:
            raise AssertionError(
                f"anchor {a} stage {j}: answers for partners "
                f"{start}:{end} arrived with {len(row)} recorded")
        row.extend(bits)
        self.in_flight -= 1
        if any(bits):
            self._true[a][j] = True
        k = len(self.stages)
        n_j = len(self.stages[j])
        complete = len(row) == n_j
        events = []
        if j == self._stage[a] and j + 1 < k:
            if self._true[a][j] and self._next[a] == n_j:
                # the whole stream is launched, so every later chunk
                # is behind it on the stream and the next stage's
                # frame write cannot race a read of this stage's
                self._stage[a] = j + 1
                self._next[a] = 0
                if self.stages[j + 1]:
                    self.ready.append(a)
                else:
                    self._stage[a] = _STRANDED
            elif complete and not self._true[a][j]:
                self._stage[a] = _DONE
                events.append(("dropped", a))
        if j == k - 1 and complete:
            self._stage[a] = _DONE
            events.append(("finished", a))
        return events

    # ---- progress ------------------------------------------------------

    def done(self):
        return not self.pending and not self.ready \
            and not self.in_flight


class FilterAdmission:
    """Filter chain scheduler: builds chunk groups, tracks arena residency.

    Args:
        doc_tokens: Per-document token counts.
        stage_tokens: Per-stage question suffix token counts.
        chunk_budget: Tokens per forward pass.
        arena_pages: Page budget for admission. None disables page
            accounting (single-stage).
        page_tokens: Tokens per arena page.
        kept_extra_tokens: Extra tokens per document that must fit in
            pages (shared preamble plus tail room).
        limit: Stop after this many survivors.

    Survivor suffixes pack before fresh admissions. Pages are granted
    in queue order; chunk room may be skipped.
    """

    def __init__(self, doc_tokens, stage_tokens, chunk_budget,
                 arena_pages, page_tokens, kept_extra_tokens=0,
                 limit=None, available_pages=None):
        self.doc_tokens = doc_tokens
        self.stage_tokens = list(stage_tokens)
        self.chunk_budget = chunk_budget
        self.arena_pages = arena_pages
        self.page_tokens = page_tokens
        # None: no page bin - nothing is ever written to the arena,
        # so there is nothing to account
        self.free_pages = (arena_pages if available_pages is None
                           else available_pages)
        if (arena_pages is not None
                and not 0 <= self.free_pages <= arena_pages):
            raise ValueError("available_pages must fit inside arena_pages")
        self.blocked_pages = 0
        self.limit = limit
        self._survivor_count = 0
        # kept_extra_tokens: the shared question preamble that joins
        # the document's kept KV after stage 1, so pages must cover it
        self.kept_extra = kept_extra_tokens
        for d, t in enumerate(self.doc_tokens):
            need = t + max(stage_tokens)
            if need > chunk_budget:
                raise ValueError(f"document {d} + question needs {need} "
                                 f"tokens > chunk budget {chunk_budget}")
            if arena_pages is not None and pages_for(
                    t + kept_extra_tokens, page_tokens) > arena_pages:
                raise ValueError(f"document {d} needs more pages than "
                                 f"the arena holds")
        self.pending = _CompactQueue(len(self.doc_tokens))
        self.ready = deque()       # (doc, stage) gated TRUE, next suffix
        self.in_flight = set()     # docs inside a launched chunk
        self.resident = {}         # doc -> pages held
        self.answers = {}          # doc -> [0/1 per answered stage]

    # ---- chunk building ------------------------------------------------

    def next_chunk(self):
        """Groups for the next chunk: [(doc, stage, fresh)].

        fresh means the document's tokens are packed and its KV is
        written to its pages. Returns [] when nothing is buildable.
        """
        if self._limit_reached():
            return []
        self.blocked_pages = 0
        room = self.chunk_budget
        groups = []
        # 1) survivor suffixes, oldest first; one live stage per doc
        n_ready = len(self.ready)
        for _ in range(n_ready):
            doc, stage = self.ready[0]
            cost = self.stage_tokens[stage]
            if cost > room:
                break    # ready is FIFO; the chunk is nearly full
            self.ready.popleft()
            groups.append((doc, stage, False))
            self.in_flight.add(doc)
            room -= cost
        # 2) fresh admissions: pages in queue order, chunk room may skip
        blocked_pages = False
        skipped = array("I")
        while self.pending and not blocked_pages:
            doc = self.pending.popleft()
            if self.free_pages is not None:
                need_pages = pages_for(
                    self.doc_tokens[doc] + self.kept_extra,
                    self.page_tokens)
                if need_pages > self.free_pages:
                    # pages are granted in order: put it back and stop
                    # claiming pages behind it
                    self.pending.appendleft(doc)
                    self.blocked_pages = need_pages - self.free_pages
                    blocked_pages = True
                    break
            cost = self.stage_tokens[0] + self.doc_tokens[doc]
            if cost > room:
                skipped.append(doc)   # chunk room only; retry next chunk
                continue
            if self.free_pages is not None:
                self.free_pages -= need_pages
                self.resident[doc] = need_pages
            groups.append((doc, 0, True))
            self.in_flight.add(doc)
            room -= cost
        self.pending.prepend(skipped)
        return groups

    # ---- gating --------------------------------------------------------

    def report(self, doc, stage, passed, release=True):
        """Record one answer.

        Frees pages on FALSE or last stage; otherwise queues the
        next-stage suffix.

        release=False assumes the caller rewinds the kept document
        to its own tokens: the tail pages return here, the rest come
        back through add_free_pages if the pool evicts it.

        Returns docs whose pages were freed.
        """
        self.in_flight.discard(doc)
        self.answers.setdefault(doc, []).append(1 if passed else 0)
        last = stage == len(self.stage_tokens) - 1
        if passed and last:
            self._survivor_count += 1
        if passed and not last:
            self.ready.append((doc, stage + 1))
            return ()
        if self.free_pages is None:
            return ()
        held = self.resident.pop(doc)
        if release:
            self.free_pages += held
            return (doc,)
        self.free_pages += held - pages_for(self.doc_tokens[doc],
                                            self.page_tokens)
        return ()

    def add_free_pages(self, pages):
        """Add pages released by retained KV outside this chain."""
        if pages < 0 or self.free_pages is None:
            raise ValueError("invalid external page release")
        self.free_pages += pages

    def sync_free_pages(self, pages):
        """Take the arena's free page count before building a chunk.

        Used when another operator frees and claims pages between this
        chain's chunks, so the chain's own arithmetic cannot track them.
        """
        if self.free_pages is None:
            raise ValueError("this chain has no page bin to sync")
        if not 0 <= pages <= self.arena_pages:
            raise ValueError("free pages must fit inside the arena")
        self.free_pages = pages

    # ---- progress ------------------------------------------------------

    def _limit_reached(self):
        return (self.limit is not None
                and self._survivor_count >= self.limit)

    def done(self):
        if self._limit_reached():
            return not self.in_flight
        return (not self.pending and not self.ready
                and not self.in_flight)

    def drain_ready(self):
        """Free pages of docs still queued when the limit ends the run early.

        Returns drained docs for arena key cleanup.
        """
        out = []
        while self.ready:
            doc, _ = self.ready.popleft()
            if self.free_pages is not None:
                self.free_pages += self.resident.pop(doc)
                out.append(doc)
        return out

    def survivors(self):
        """Documents that answered TRUE at every stage."""
        n = len(self.stage_tokens)
        return sorted(d for d, row in self.answers.items()
                      if len(row) == n and all(row))
