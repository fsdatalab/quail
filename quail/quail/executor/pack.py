"""Chunk packing and admission - no GPU, no torch, unit-tested.

Two schedulers live here:

- pack_stream and friends: brim-packing a known pair list into chunks
  (the join path, ported from the exploration's joinlogic).
- FilterAdmission: continuous admission for filter chains - a pending
  queue and a resident set, nothing in lockstep, no barriers. Each
  chunk fills from two sources in priority order: next-stage suffixes
  of resident survivors (every one completed moves a document toward
  freeing its pages), then fresh documents admitted whenever their
  page-rounded tokens fit the free list.

Length units are tokens everywhere. A "suffix" is one partner document
plus the question tail (join) or one question (filter); suffixes are
atomic - a suffix's tokens attend to each other, so one suffix never
splits across two chunks.
"""

from collections import deque


def orient(mean_left_tokens, mean_right_tokens):
    """Which side anchors: the longer one. Anchor tokens are paid
    once per document plus a cheap shared read; partner tokens are
    paid once per pair, so the short side must stream."""
    return "left" if mean_left_tokens >= mean_right_tokens else "right"


def plan_groups(prefix_tokens, suffix_tokens, budget):
    """Split one anchor's suffix stream into consecutive chunk groups.

    Each group is a (start, end) slice of the suffix list whose
    tokens fit under `budget` together with one copy of the prefix
    (the prefix is recomputed - or read from kept KV - at the head of
    every group)."""
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
    """Brim-pack a whole pair list into chunks, keeping cut prefixes.

    anchors: list of (prefix_tokens, [suffix_tokens]) in run order.
    keep: anchor indices whose prefix KV a later stage needs.
    already_kept: anchor indices whose prefix KV is already resident
    from an earlier stage; their groups never pack prefix tokens.

    Returns (chunks, kv_to_cache). Each chunk is a list of groups
    (anchor_index, start, end, carried); carried means the group
    packs a fresh copy of the anchor's prefix tokens ahead of
    suffixes start..end. An anchor's prefix is packed at most once,
    ever: when the budget cuts its stream, the anchor continues in
    the next chunk with carried False, and its suffixes read the
    prefix KV from the arena instead. kv_to_cache is the set of
    anchors whose prefix KV must be written to the arena - their
    stream is cut mid-chunk, or a later stage needs them (keep).
    Suffix KV is never cached anywhere. Cuts happen only at suffix
    boundaries; when an anchor's stream ends mid-chunk, the next
    anchor starts in the same chunk.
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
    """Output triples of a pairwise-chained 3-way from recorded
    answers. Kept for the milestone-1 replay cell only: the engine
    now runs an n-way join as one cross-product stage under a single
    prompt, so the session never chains stages or calls this.

    ans1_rows: b -> row of 0/1 over A (stage 1, anchored on b).
    ans2_rows: b -> row of 0/1 over C, present only for gated
    survivors. Triples fan out from each surviving b's matched a's
    crossed with its matched c's - no model calls."""
    m1, m2 = matches(ans1_rows), matches(ans2_rows)
    out = []
    for b in sorted(ans2_rows):
        for a in m1.get(b, ()):
            for c in m2[b]:
                out.append((a, b, c))
    return sorted(out)


def brute_force_triples(ans1_rows, ans2_rows):
    """The nested-loop reference over the same recorded answers: no
    gating, no dedup. Short-circuit order means a gated b - whose
    stage-2 row was never recorded - is never looked up, because its
    stage-1 row has no TRUE."""
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
    """Page-rounded residency: only document tokens occupy pages;
    suffix KV never exists."""
    return -(-tokens // page_tokens)


class FilterAdmission:
    """The filter chain's scheduler: which groups go in the next
    chunk, and what the arena holds.

    doc_tokens: per-document token counts.
    stage_tokens: per-stage question suffix token counts.
    chunk_budget: tokens per forward pass (one bin).
    arena_pages / page_tokens: the admission budget (the other bin).

    Rules, from the design:
    - survivor suffixes pack before fresh admissions, so residency
      drains monotonically;
    - a document's pages are claimed at admission and returned the
      instant it fails a stage or answers its last one;
    - pages are granted in queue order (a document that cannot fit
      its pages blocks later page claims, so its wait is bounded -
      pages only ever flow back), but chunk ROOM may be skipped:
      a document too big for what is left of this chunk does not
      stop smaller work from filling it;
    - a document that could not fit even into an empty arena or an
      empty chunk was refused at plan time, so there is no deadlock
      case here.
    """

    def __init__(self, doc_tokens, stage_tokens, chunk_budget,
                 arena_pages, page_tokens, kept_extra_tokens=0,
                 restored=(), limit=None):
        self.doc_tokens = list(doc_tokens)
        self.stage_tokens = list(stage_tokens)
        self.chunk_budget = chunk_budget
        self.page_tokens = page_tokens
        self.free_pages = arena_pages
        self.limit = limit
        self._survivor_count = 0
        # kept_extra_tokens: the shared question preamble that joins
        # the document's kept KV after stage 1 (chain mode kept it
        # resident; so do we), so pages must cover it
        self.kept_extra = kept_extra_tokens
        # restored: documents whose KV loads from the store instead of
        # computing - their admission claims the same pages but their
        # chunk cost is the first question only
        self.restored = set(restored)
        for d, t in enumerate(self.doc_tokens):
            need = t + max(stage_tokens)
            if need > chunk_budget:
                raise ValueError(f"document {d} + question needs {need} "
                                 f"tokens > chunk budget {chunk_budget}")
            if pages_for(t + kept_extra_tokens, page_tokens) > arena_pages:
                raise ValueError(f"document {d} needs more pages than "
                                 f"the arena holds")
        self.pending = deque(range(len(self.doc_tokens)))
        self.ready = deque()       # (doc, stage) gated TRUE, next suffix
        self.in_flight = set()     # docs inside a launched chunk
        self.resident = {}         # doc -> pages held
        self.answers = {}          # doc -> [0/1 per answered stage]

    # ---- chunk building ------------------------------------------------

    def next_chunk(self):
        """Groups for the next chunk: [(doc, stage, fresh)], fresh
        meaning the document's tokens ride along and its KV is written
        to its pages. Empty list means nothing is buildable right now
        (answers are still in flight)."""
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
            need_pages = pages_for(self.doc_tokens[doc] + self.kept_extra,
                                   self.page_tokens)
            if need_pages > self.free_pages:
                # pages are granted in order: put it back and stop
                # claiming pages behind it
                self.pending.appendleft(doc)
                blocked_pages = True
                break
            cost = self.stage_tokens[0] + (
                0 if doc in self.restored else self.doc_tokens[doc])
            if cost > room:
                skipped.append(doc)   # chunk room only; retry next chunk
                continue
            self.free_pages -= need_pages
            self.resident[doc] = need_pages
            groups.append((doc, 0, True))
            self.in_flight.add(doc)
            room -= cost
        while skipped:
            self.pending.appendleft(skipped.pop())
        return groups

    # ---- gating --------------------------------------------------------

    def report(self, doc, stage, passed, release=True):
        """One landed answer. Frees pages on FALSE or on the last
        stage; otherwise the next-stage suffix becomes ready.

        release=False keeps a leaving document's pages held (the store
        is copying them out); the caller returns them with release()
        when the copy completes."""
        self.in_flight.discard(doc)
        self.answers.setdefault(doc, []).append(1 if passed else 0)
        last = stage == len(self.stage_tokens) - 1
        if passed and last:
            self._survivor_count += 1
        if passed and not last:
            self.ready.append((doc, stage + 1))
            return
        if release:
            self.free_pages += self.resident.pop(doc)

    def release(self, doc):
        """Return a document's pages after a deferred store save."""
        self.free_pages += self.resident.pop(doc)

    # ---- progress ------------------------------------------------------

    def _limit_reached(self):
        return (self.limit is not None
                and self._survivor_count >= self.limit)

    def done(self):
        if self._limit_reached():
            return not self.in_flight
        return (not self.pending and not self.ready
                and not self.in_flight)

    def survivors(self):
        """Documents that answered TRUE at every stage."""
        n = len(self.stage_tokens)
        return sorted(d for d, row in self.answers.items()
                      if len(row) == n and all(row))
