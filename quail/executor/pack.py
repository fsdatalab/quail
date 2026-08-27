"""Chunk packing and admission - no GPU, no torch, unit-tested.

- pack_stream: brim-packing a known pair list into chunks (join path).
- FilterAdmission: continuous admission for filter chains. Survivors
  pack before fresh admissions; fresh documents admit when their
  page-rounded tokens fit the free list.

Length units are tokens. Suffixes are atomic and never split across
chunks.
"""

from collections import deque


def orient(mean_left_tokens, mean_right_tokens):
    """Which side anchors: the longer one. Anchor tokens are paid once
    per document; partner tokens are paid once per pair."""
    return "left" if mean_left_tokens >= mean_right_tokens else "right"


def plan_groups(prefix_tokens, suffix_tokens, budget):
    """Split one anchor's suffix stream into (start, end) chunk groups
    that fit under `budget` together with the prefix."""
    room = budget - prefix_tokens
    if room <= 0:
        raise ValueError(
            f"prefix {prefix_tokens} tokens leaves no room in a "
            f"{budget}-token chunk")
    groups, start, used = [], 0, 0
    for i, s in enumerate(suffix_tokens):
        if s > room:
            raise ValueError(
                f"suffix {i} has {s} tokens; at most {room} fit beside "
                f"a {prefix_tokens}-token prefix (suffixes are atomic)")
        if used + s > room:
            groups.append((start, i))
            start, used = i, 0
        used += s
    if used or not suffix_tokens:
        groups.append((start, len(suffix_tokens)))
    return groups


def pack_stream(anchors, budget, keep=(), already_kept=()):
    """Brim-pack a pair list into chunks, keeping cut prefixes in the
    arena.

    Args:
        anchors: List of (prefix_tokens, [suffix_tokens]) in run order.
        keep: Anchor indices whose prefix KV a later stage needs.
        already_kept: Anchors whose prefix KV is already resident.

    Returns:
        (chunks, kv_to_cache). Each chunk is a list of
        (anchor_index, start, end, carried); carried means the group
        packs the anchor's prefix tokens. kv_to_cache is the set of
        anchors whose prefix KV must be written to the arena.
    """
    keep, already = set(keep), set(already_kept)
    chunks, chunk, used = [], [], 0
    kv_to_cache = set()
    for a, (prefix, suffixes) in enumerate(anchors):
        n = len(suffixes)
        placed = a in already
        if n == 0:
            if placed or a not in keep:
                continue    # nothing streams against it and no later
                #             stage needs it: computing it serves no one
            # cache-only placement: a prefix a later stage needs,
            # with nothing streamed against it in this one
            if prefix > budget:
                raise ValueError(
                    f"anchor {a}: prefix {prefix} tokens exceeds the "
                    f"{budget}-token chunk budget")
            if used + prefix > budget:
                chunks.append(chunk)
                chunk, used = [], 0
            chunk.append((a, 0, 0, True))
            used += prefix
            if a in keep:
                kv_to_cache.add(a)
            continue
        i = 0
        while i < n:
            carried = 0 if placed else prefix
            if suffixes[i] + carried > budget:
                raise ValueError(
                    f"anchor {a} suffix {i} has {suffixes[i]} tokens; "
                    f"at most {budget - carried} fit beside what its "
                    f"group must carry (suffixes are atomic)")
            room = budget - used - carried
            if room < suffixes[i]:
                chunks.append(chunk)
                chunk, used = [], 0
                continue
            j, group_tokens = i, 0
            while j < n and group_tokens + suffixes[j] <= room:
                group_tokens += suffixes[j]
                j += 1
            chunk.append((a, i, j, not placed))
            used += carried + group_tokens
            placed = True
            i = j
            if i < n and a not in already:
                kv_to_cache.add(a)  # a later chunk reads this KV
        if a in keep and a not in already:
            kv_to_cache.add(a)
    if chunk:
        chunks.append(chunk)
    return chunks, kv_to_cache


def gate(answer_rows):
    """Anchors that survive a conjunctive stage: any TRUE in the row.
    answer_rows: dict anchor_index -> iterable of 0/1 answers.
    Returns the sorted surviving anchor indices."""
    return sorted(a for a, row in answer_rows.items() if any(row))


def matches(answer_rows):
    """anchor_index -> sorted list of partner indices answered TRUE."""
    return {a: sorted(i for i, v in enumerate(row) if v)
            for a, row in answer_rows.items()}


def assemble(ans1_rows, ans2_rows):
    """Output triples from two pairwise join stages sharing one anchor.

    ans1_rows: b -> row of 0/1 over A (stage 1, anchored on b).
    ans2_rows: b -> row of 0/1 over C, present only for gated
    survivors."""
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

def pages_for(tokens: int, page_tokens: int) -> int:
    """Pages needed for `tokens` rows (suffix KV is never paged)."""
    return -(-tokens // page_tokens)


class FilterAdmission:
    """Filter chain scheduler: builds chunk groups and tracks arena
    residency.

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
                 limit=None):
        self.doc_tokens = list(doc_tokens)
        self.stage_tokens = list(stage_tokens)
        self.chunk_budget = chunk_budget
        self.page_tokens = page_tokens
        # None: no page bin - nothing is ever written to the arena,
        # so there is nothing to account
        self.free_pages = arena_pages
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
        self.pending = deque(range(len(self.doc_tokens)))
        self.ready = deque()       # (doc, stage) gated TRUE, next suffix
        self.in_flight = set()     # docs inside a launched chunk
        self.resident = {}         # doc -> pages held
        self.answers = {}          # doc -> [0/1 per answered stage]

    # ---- chunk building ------------------------------------------------

    def next_chunk(self):
        """Groups for the next chunk: [(doc, stage, fresh)].

        fresh means the document's tokens are packed and its KV is
        written to its pages. Returns [] when nothing is buildable."""
        if self._limit_reached():
            return []
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
        skipped = deque()
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
        while skipped:
            self.pending.appendleft(skipped.pop())
        return groups

    # ---- gating --------------------------------------------------------

    def report(self, doc, stage, passed):
        """Record one answer. Frees pages on FALSE or last stage;
        otherwise queues the next-stage suffix.

        Returns docs whose pages were freed."""
        self.in_flight.discard(doc)
        self.answers.setdefault(doc, []).append(1 if passed else 0)
        last = stage == len(self.stage_tokens) - 1
        if passed and last:
            self._survivor_count += 1
        if passed and not last:
            self.ready.append((doc, stage + 1))
            return ()
        if self.free_pages is not None:
            self.free_pages += self.resident.pop(doc)
            return (doc,)
        return ()

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
        """Free pages of docs still queued when the limit ends the
        run early. Returns drained docs for arena key cleanup."""
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
