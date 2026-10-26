"""KV pages that grow as documents write, and the admission that sizes them.

A document that decodes an answer gets pages for its prompt and its
first request when it is admitted, and more pages as its requests
need them. Two rules decide who gets pages:

* Admission projects the pool's occupancy forward. Each running document
  is expected to write its remaining answer, at a quantile of the
  answer lengths settled so far, one row per pass, and to free its
  pages when it ends. A fresh document is admitted when the peak of
  that projection, with the fresh document added, fits the free
  pages. Documents end at different times, so the pages one frees go
  to the ones still writing.
* Growth keeps the pool safe: every running document could still write
  its longest answer if the ones admitted before it finish first. A
  document grows only while that stays true; otherwise it waits a
  pass. The oldest running document can always grow, so the run always
  makes progress, and nothing is preempted.

The growth model is the operator's view of its documents; see
StageGrowth.
"""

from typing import Protocol

import numpy as np


class StageGrowth(Protocol):
    """What the stage executor asks of an operator whose KV grows.

    Attributes:
        initial_tokens: Rows past a document's prefix its pages cover
            when it is admitted: its first request and the rounds that
            can be in flight before it first grows.
    """

    initial_tokens: int

    def remaining(self, anchors: np.ndarray) -> np.ndarray:
        """Return the rows each document is expected to still write."""

    def capacities(self, anchors: np.ndarray) -> np.ndarray:
        """Return the rows each document's pages must cover for its next request."""


class KVGrowth:
    """The running documents' pages, grown on demand and admitted by projection.

    Args:
        arena: The KV arena holding the documents' keys.
        keys: Per anchor, its arena key.
        model: The operator's StageGrowth.
        prefix_tokens: Per anchor, its prefix length.
        final_tokens: Per anchor, the most rows its key can ever need.
        initial_tokens: Rows past the prefix an admitted document's
            pages cover; the model's initial_tokens when None.
    """

    def __init__(self, arena, keys, model: StageGrowth, prefix_tokens, final_tokens,
                 initial_tokens=None):
        self.arena = arena
        self.keys = keys
        self.model = model
        self.page = arena.page_tokens
        self.prefix = np.asarray(prefix_tokens, dtype=np.int64)
        self.final = np.asarray(final_tokens, dtype=np.int64)
        self.initial = (model.initial_tokens if initial_tokens is None
                        else int(initial_tokens))
        num_anchors = len(self.final)
        horizon = int((self.final - self.prefix).max()) if num_anchors else 0
        # the projection's time axis, in rows written, a page apart
        self.grid = np.arange(0, horizon + self.page, self.page, dtype=np.int64)
        # pages the projection leaves free at its peak: the most one
        # document can still grow, so the oldest grows without
        # stalling the others
        self.reserve = int(self._pages(horizon))
        # rows each anchor's pages cover, and pages its finish does not
        # free: the ones it borrowed and the ones it lent to borrowers
        self.capacity = np.zeros(num_anchors, dtype=np.int64)
        self.kept_pages = np.zeros(num_anchors, dtype=np.int64)
        self.running = {}      # anchor -> None, in admission order
        self._margins = None
        self._projection = None
        self.stalls = 0
        self.grown_pages = 0
        self.max_running = 0

    def _pages(self, rows):
        return -(-np.asarray(rows, dtype=np.int64) // self.page)

    def stats(self) -> dict:
        """Return the stalls, the pages grown, and the most running documents."""
        return {"stalls": self.stalls, "grown_pages": self.grown_pages,
                "max_running": self.max_running}

    # ---- the running documents ----------------------------------------------

    def admitted(self, anchor: int, lent_pages: int = 0) -> None:
        """Record an admitted document's pages.

        Args:
            anchor: The document.
            lent_pages: Leading pages its borrowers will read, which
                its finish does not free.
        """
        accounting = self.arena.accounting
        key = self.keys[anchor]
        self.capacity[anchor] = len(accounting.table_pages(key)) * self.page
        self.kept_pages[anchor] = len(accounting.borrowed[key]) + lent_pages
        self.running[anchor] = None
        self.max_running = max(self.max_running, len(self.running))
        # admits() already counted this document in the projection; it
        # is the youngest, so its margin joins the end of the others
        if self._margins is not None:
            rank, base, total_frees = self._margins
            held = int(self._pages(self.capacity[anchor]))
            frees = max(held - int(self.kept_pages[anchor]), 0)
            need = max(int(self._pages(self.final[anchor])) - held, 0)
            rank[anchor] = len(base)
            self._margins = (rank, np.append(base, total_frees - need),
                             total_frees + frees)

    def settled(self, anchor: int) -> None:
        """Forget a document whose pages were freed or retained."""
        if anchor in self.running:
            del self.running[anchor]
            self._margins = None
            self._projection = None

    def _state(self):
        """The running anchors in admission order, the pages each frees, and needs."""
        running = np.fromiter(self.running, dtype=np.int64, count=len(self.running))
        held = self._pages(self.capacity[running])
        frees = np.maximum(held - self.kept_pages[running], 0)
        need = np.maximum(self._pages(self.final[running]) - held, 0)
        return running, held, frees, need

    def _margin_base(self):
        """Return each running anchor's rank and its margin less the free pages.

        Document i's margin is the free pages, plus the pages the
        documents admitted before it free when they finish, less the
        pages its own longest answer still needs. The pool is safe
        while every margin is at least zero.
        """
        if self._margins is None:
            running, _, frees, need = self._state()
            before = np.cumsum(frees) - frees
            rank = np.full(len(self.final), len(running), dtype=np.int64)
            rank[running] = np.arange(len(running))
            self._margins = rank, before - need, int(frees.sum())
        return self._margins

    # ---- growth before a pass ----------------------------------------------

    def grow(self, anchors, free_pages: int) -> set:
        """Grow the documents whose next request needs more rows.

        Documents grow oldest first. One grows only while every older
        document's margin stays at least zero; the others wait.

        Args:
            anchors: The documents whose next request may be packed.
            free_pages: The arena's free pages.

        Returns:
            The anchors that could not grow, to hold back this pass.
        """
        anchors = np.asarray(anchors, dtype=np.int64)
        if not len(anchors):
            return set()
        wanted = self.model.capacities(anchors)
        short = wanted > self.capacity[anchors]
        if not short.any():
            return set()
        anchors, wanted = anchors[short], wanted[short]
        rank, base, _ = self._margin_base()
        ranks = rank[anchors]
        stalled = set()
        for index in np.argsort(ranks, kind="stable").tolist():
            anchor, rows = int(anchors[index]), int(wanted[index])
            position = int(ranks[index])
            gain = int(self._pages(rows) - self._pages(self.capacity[anchor]))
            older = base[:position]
            unsafe = len(older) and free_pages + int(older.min()) < gain
            if gain > free_pages or unsafe:
                stalled.add(anchor)
                continue
            key = self.keys[anchor]
            if self.arena.activate(key, self.arena.accounting.tokens[key],
                                   capacity_tokens=rows) is None:
                stalled.add(anchor)
                continue
            pages = self.arena.accounting.table_pages(key)
            self.capacity[anchor] = len(pages) * self.page
            free_pages -= gain
            # the grown document needs less and frees more, so the
            # margins from it on keep their value as the free pages fall
            base[position:] += gain
            self.grown_pages += gain
        self.stalls += len(stalled)
        self._projection = None
        return stalled

    # ---- admission ---------------------------------------------------------

    def _projected(self):
        """Return the running documents' projected growth, in pages, over the grid.

        Entry t is the pages the running documents are expected to hold
        grid[t] rows from now past what they hold now: each grows a
        row per pass until its expected answer ends, then frees the
        pages its finish frees.
        """
        if self._projection is None:
            running, held, frees, _ = self._state()
            if not len(running):
                self._projection = np.zeros(len(self.grid), dtype=np.int64)
            else:
                remaining = self.model.remaining(running)
                rows = self.capacity[running][None, :] + self.grid[:, None]
                writing = self.grid[:, None] < remaining[None, :]
                growth = np.where(writing, np.maximum(self._pages(rows) - held, 0),
                                  -frees[None, :])
                self._projection = growth.sum(axis=1)
        return self._projection

    def admits(self, anchor: int, pages: int, free_pages: int) -> bool:
        """Return whether a fresh document fits, and count it if so.

        It fits when the pool stays safe with it admitted last and the
        projected growth, with it added, never comes within reserve
        pages of the pages left.

        Args:
            anchor: The fresh document.
            pages: The pages its admission takes now.
            free_pages: The free pages before its admission.
        """
        _, base, total_frees = self._margin_base()
        left = free_pages - pages
        if len(base) and left + int(base.min()) < 0:
            return False
        if left + total_frees < int(self._pages(self.final[anchor])) - pages:
            return False
        growth = self._projected()
        remaining = int(self.model.remaining(np.array([anchor], dtype=np.int64))[0])
        rows = self.prefix[anchor] + self.initial + self.grid
        own = np.where(self.grid < remaining, self._pages(rows) - pages, -pages)
        if int((growth + own).max()) > left - self.reserve:
            return False
        self._projection = growth + own
        return True
