"""Page accounting for the KV arena: a free list, page lists per key, retention."""

import heapq
from collections import deque

import numpy as np


class PageArena:
    """Page accounting: a free list and per-document page lists.

    A key may borrow the leading pages of another resident key's
    prefix: it reads them and never writes them. A page returns to
    the free list when its last holder frees it.

    By default the free list is a stack, and a free page's KV is
    forgotten. keep_freed_contents makes it vLLM's free block queue: a
    key's pages return last page first, the page freed longest ago is
    taken first, and a free page keeps a label for the KV it holds
    until it is taken, so revive can make it a key again.
    """

    def __init__(self, num_pages: int, page_tokens: int, num_slots: int = 0):
        self.num_pages = num_pages
        self.page_tokens = page_tokens
        self.num_slots = num_slots
        self.free_slots = list(range(num_slots - 1, -1, -1))    # stack
        self.slots = {}    # key -> the checkpoint slot it holds
        self.slot_tokens = {}    # key -> tokens its checkpoint covers
        self.ext_slots = {}    # key -> slot of the state past retained suffix
        self.ext_tokens = {}    # key -> tokens that second checkpoint covers
        self.share_slots = {}    # (key, tokens) -> slot of the state there
        self.free = list(range(num_pages - 1, -1, -1))    # stack
        self.queue = None     # (page, its free count) once freed contents stay
        self.frees = None    # page id -> times it went free
        self.labels = {}      # label -> the free page holding that KV
        self.label_of = {}    # page id -> its labels
        self.owned = {}       # key -> list of page ids it alone writes
        self.borrowed = {}    # key -> leading page ids read from a parent
        self.holds = np.zeros(num_pages, dtype=np.int32)    # page id -> its holders
        self.tokens = {}      # key -> resident token count, borrowed included
        self.pinned = set()    # current operators depend on these keys
        self._retained_sizes = {}
        self._retained_pages = 0
        self.retained = {}     # key -> reusable prefix tokens
        self._retained_heap = []
        self._retained_versions = {}
        self._retention_version = 0
        self.retention_policy = None
        self.retention_cap_pages = None

    def pages_needed(self, tokens: int) -> int:
        return -(-tokens // self.page_tokens)

    def alloc(self, key, tokens: int, capacity_tokens: int | None = None,
              borrowed=()):
        """The document's own pages, or None when the free list is short.

        The admission scheduler treats None as "wait". borrowed are
        resident pages of another key the new key reads as its
        leading pages (see shared_pages); tokens and capacity_tokens
        count them.
        """
        if key in self.owned:
            raise KeyError(f"{key!r} already resident")
        capacity_tokens = tokens if capacity_tokens is None else capacity_tokens
        if capacity_tokens < tokens:
            raise ValueError("capacity_tokens must cover logical tokens")
        borrowed = list(borrowed)
        if len(borrowed) * self.page_tokens > tokens:
            raise ValueError("a key cannot borrow past its own tokens")
        need = self.pages_needed(capacity_tokens) - len(borrowed)
        if need > self.free_pages:
            return None
        pages = self._take(need)
        self.owned[key] = pages
        self.borrowed[key] = borrowed
        self.tokens[key] = tokens
        self.holds[borrowed + pages] += 1
        return pages

    def shared_pages(self, parent, shared: int, start: int = 0) -> list:
        """The parent's pages for its tokens [start, shared), to borrow.

        Both bounds are whole pages. The parent's first page must hold
        token `start` or an earlier one.
        """
        if (shared - start) % self.page_tokens or start % self.page_tokens:
            raise ValueError("a borrowed prefix is a whole number of pages")
        if parent not in self.owned:
            raise KeyError(f"borrow parent {parent!r} is not resident")
        if shared > self.tokens[parent]:
            raise ValueError(f"{parent!r} holds fewer tokens than borrowed")
        first = start // self.page_tokens
        return self.table_pages(parent)[first:shared // self.page_tokens]

    def drop_leading(self, key, pages: int) -> int:
        """Drop a key's first pages, borrowed ones first; returns pages freed."""
        borrowed = self.borrowed[key]
        from_borrowed = min(pages, len(borrowed))
        dropped = borrowed[:from_borrowed] + self.owned[key][:pages - from_borrowed]
        self.borrowed[key] = borrowed[from_borrowed:]
        self.owned[key] = self.owned[key][pages - from_borrowed:]
        return self.release_pages(dropped)

    def table_pages(self, key) -> list:
        """The pages a key's block table reads: borrowed, then its own."""
        return self.borrowed[key] + self.owned[key]

    def borrowed_tokens(self, key) -> int:
        return len(self.borrowed[key]) * self.page_tokens

    def release_pages(self, pages) -> int:
        """Drop one hold on each page; returns how many went free."""
        pages = np.asarray(pages, dtype=np.int64)
        self.holds[pages] -= 1
        freed = pages[self.holds[pages] == 0]
        if self.queue is None:
            self.free.extend(freed.tolist())
            return len(freed)
        # the last page first, so a prefix's first pages are taken last
        freed = freed[::-1]
        self.frees[freed] += 1
        self.queue.extend(zip(freed.tolist(), self.frees[freed].tolist()))
        self.num_free += len(freed)
        return len(freed)

    def keep_freed_contents(self) -> None:
        """Take the page freed longest ago first, and keep free pages' labels."""
        self.frees = np.zeros(self.num_pages, dtype=np.int64)
        # the stack's next page first
        self.queue = deque((page, 0) for page in reversed(self.free))
        self.num_free = len(self.free)
        self.free = []

    def forget_freed_contents(self) -> None:
        """Return to a stack of free pages that keeps no labels."""
        if self.queue is None:
            return
        free = list(dict.fromkeys(
            page for page, freed in self.queue
            if not self.holds[page] and freed == self.frees[page]))
        self.free = free[::-1]
        self.queue = self.frees = None
        self.labels.clear()
        self.label_of.clear()

    def _take(self, count: int) -> list:
        """Remove count free pages; a taken page's KV is overwritten."""
        if self.queue is None:
            taken = self.free[len(self.free) - count:][::-1]
            del self.free[len(self.free) - count:]
            return taken
        taken = []
        while len(taken) < count:
            page, freed = self.queue.popleft()
            # an entry is stale once its page was revived or freed again
            if self.holds[page] or freed != self.frees[page]:
                continue
            taken.append(page)
            for label in self.label_of.pop(page, ()):
                if self.labels.get(label) == page:
                    del self.labels[label]
        self.num_free -= count
        return taken

    def label(self, pages, tag) -> None:
        """Name the KV of pages a later key may revive: (tag, page index)."""
        if self.queue is None:
            return
        for index, page in enumerate(pages):
            self.labels[(tag, index)] = page
            self.label_of.setdefault(page, set()).add((tag, index))

    def free_run(self, tag, count: int) -> list:
        """The free pages holding (tag, 0), (tag, 1), ... up to the first gap."""
        run = []
        for index in range(count):
            page = self.labels.get((tag, index))
            if page is None or self.holds[page]:
                break
            run.append(page)
        return run

    def revive(self, key, pages) -> None:
        """Make free pages a resident key that holds the KV they kept."""
        if key in self.owned:
            raise KeyError(f"{key!r} already resident")
        pages = list(pages)
        if self.holds[pages].any():
            raise ValueError("revived pages must be free")
        self.owned[key] = pages
        self.borrowed[key] = []
        self.tokens[key] = len(pages) * self.page_tokens
        self.holds[pages] += 1
        self.num_free -= len(pages)

    def alloc_slot(self, key):
        """A checkpoint slot for a resident key, or None when none is free.

        A key that already holds a slot keeps it.
        """
        if key not in self.owned:
            raise KeyError(key)
        if key in self.slots:
            return self.slots[key]
        if not self.free_slots:
            return None
        self.slots[key] = self.free_slots.pop()
        return self.slots[key]

    def alloc_extension(self, key):
        """A second checkpoint slot for a resident key, or None when none is free.

        Holds the state past tokens the key's KV retains beyond its
        document, so the document-end checkpoint stays for a later
        operator that asks at the document's length.
        """
        if key not in self.owned:
            raise KeyError(key)
        if key in self.ext_slots:
            return self.ext_slots[key]
        if not self.free_slots:
            return None
        self.ext_slots[key] = self.free_slots.pop()
        return self.ext_slots[key]

    def alloc_share(self, key, tokens: int):
        """A slot for the key's state after `tokens` tokens, or None.

        A borrower of the key's first `tokens` tokens starts from it.
        The slot lives until the key is freed.
        """
        if key not in self.owned:
            raise KeyError(key)
        point = (key, tokens)
        if point in self.share_slots:
            return self.share_slots[point]
        if not self.free_slots:
            return None
        self.share_slots[point] = self.free_slots.pop()
        return self.share_slots[point]

    def drop_extension(self, key) -> None:
        """Free the key's second checkpoint slot, if it has one."""
        slot = self.ext_slots.pop(key, None)
        self.ext_tokens.pop(key, None)
        if slot is not None:
            self.free_slots.append(slot)

    def free_key(self, key) -> int:
        """Drop a document's pages and slots; returns the pages that went free."""
        slot = self.slots.pop(key, None)
        self.slot_tokens.pop(key, None)
        self.drop_extension(key)
        for point in [p for p in self.share_slots if p[0] == key]:
            self.free_slots.append(self.share_slots.pop(point))
        if slot is not None:
            self.free_slots.append(slot)
        pages = self.owned.pop(key)
        borrowed = self.borrowed.pop(key)
        self.tokens.pop(key)
        self.pinned.discard(key)
        self._forget_retained(key)
        return self.release_pages(borrowed + pages)

    def pin(self, key) -> None:
        """Protect a resident key while an operator uses it."""
        if key not in self.owned:
            raise KeyError(key)
        self._forget_retained(key)
        self.pinned.add(key)

    def _forget_retained(self, key):
        self._retained_pages -= self._retained_sizes.pop(key, 0)
        self.retained.pop(key, None)
        self._retained_versions.pop(key, None)

    def retain(self, key, priority=None) -> None:
        """Make a resident key evictable after its current use."""
        if key not in self.owned:
            raise KeyError(key)
        self.pinned.discard(key)
        prefix_tokens = self.tokens[key]
        self._forget_retained(key)
        # the pages evicting the key would free: those it alone holds
        # now; a child freed later leaves more, so the count is a
        # lower bound, and admission short of pages evicts anyway
        pages = int((self.holds[self.table_pages(key)] == 1).sum())
        self.retained[key] = prefix_tokens
        self._retained_sizes[key] = pages
        self._retained_pages += pages
        self._retention_version += 1
        version = self._retention_version
        self._retained_versions[key] = version
        if priority is None:
            # a key whose pages its children all share frees none alone
            priority = (self.retention_policy.priority(
                key, prefix_tokens, max(1, pages))
                if self.retention_policy
                else (prefix_tokens / max(1, pages),))
        heapq.heappush(
            self._retained_heap,
            (priority, version, key, pages, prefix_tokens),
        )

    def configure_retention(self, policy, cap_pages: int) -> None:
        """Update priorities without making active prefixes evictable."""
        if not 0 <= cap_pages <= self.num_pages:
            raise ValueError("retention capacity must fit inside the arena")
        self.retention_policy = policy
        self.retention_cap_pages = cap_pages
        keys = list(self.retained)
        self._retained_heap.clear()
        for key in keys:
            self.retain(key)

    def pop_retained_victim(self, keep=frozenset()):
        """Remove the retained prefix with the lowest planned reuse priority.

        Keys in keep stay retained and are passed over.
        """
        passed = []
        victim = None
        while self._retained_heap:
            entry = heapq.heappop(self._retained_heap)
            _, version, key, pages, prefix_tokens = entry
            if self._retained_versions.get(key) != version:
                continue
            if key in keep:
                passed.append(entry)
                continue
            self._forget_retained(key)
            victim = key, pages, prefix_tokens
            break
        for entry in passed:
            heapq.heappush(self._retained_heap, entry)
        return victim

    def rewind(self, key, tokens: int) -> int:
        """Keep the first tokens and return unused trailing pages."""
        if key not in self.owned:
            raise KeyError(key)
        borrowed = self.borrowed_tokens(key)
        capacity = borrowed + len(self.owned[key]) * self.page_tokens
        if tokens < borrowed or tokens > capacity:
            raise ValueError("rewind tokens leave the resident capacity")
        keep = self.pages_needed(tokens) - len(self.borrowed[key])
        released = self.owned[key][keep:]
        self.owned[key] = self.owned[key][:keep]
        self.tokens[key] = tokens
        freed = self.release_pages(released)
        if key in self.retained:
            self.retain(key)
        return freed

    def grow(self, key, capacity_tokens: int) -> int | None:
        """Add pages for capacity_tokens without changing logical tokens."""
        if key not in self.owned:
            raise KeyError(key)
        need = (self.pages_needed(capacity_tokens) - len(self.borrowed[key])
                - len(self.owned[key]))
        if need <= 0:
            return 0
        if need > self.free_pages:
            return None
        added = self._take(need)
        self.owned[key].extend(added)
        self.holds[added] += 1
        return need

    def grow_pages(self, keys) -> list | None:
        """Give each key one more page; None when the free list is short."""
        if len(keys) > self.free_pages:
            return None
        added = self._take(len(keys))
        for key, page in zip(keys, added):
            self.owned[key].append(page)
        self.holds[added] += 1
        return added

    def row_indices(self, key, tokens=None):
        """Flat row positions of the document's tokens inside a pool.

        The pool is viewed as (num_pages * page_tokens, ...): page p holds
        rows p*page_tokens .. p*page_tokens + page_tokens.
        """
        rows = []
        left = self.tokens[key] if tokens is None else tokens
        for p in self.table_pages(key):
            take = min(left, self.page_tokens)
            base = p * self.page_tokens
            rows.extend(range(base, base + take))
            left -= take
        return rows

    @property
    def free_pages(self) -> int:
        return len(self.free) if self.queue is None else self.num_free

    @property
    def resident_tokens(self) -> int:
        return sum(self.tokens.values())

    @property
    def retained_pages(self) -> int:
        return self._retained_pages

    @property
    def retained_prefix_tokens(self) -> int:
        return sum(self.tokens[key] for key in self.retained)


def _evict_event(key, prefix_tokens):
    """The answer-sink entry for one evicted document prefix.

    Returns None when the key is not an alias and a document index.
    """
    if not (isinstance(key, tuple) and len(key) == 2):
        return None
    alias, document = key
    if not isinstance(alias, str):
        return None
    try:
        document = int(document)
    except (TypeError, ValueError):
        return None
    return {"kind": "evict", "alias": alias, "document": document,
            "tokens": int(prefix_tokens)}
