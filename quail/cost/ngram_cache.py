"""The continuations seen after each run of recent tokens, with counts.

A map's draft needs a guess at the model's next tokens even where the
document does not continue the answer. The cache indexes every run of
the last 2 to 4 committed tokens to the tokens that followed it in
this query run: the answers written so far, across documents. Counts
seed the acceptance estimate, and the document that last inserted a
continuation says whether a match comes from the drafting document's
own output or a sibling's.

The cache is an open-addressing hash table in arrays, so a chunk's
documents look up and insert together in a few array passes. A key is
the 64-bit hash of its order and tokens; two keys with one hash share
an entry, which a verified draft tolerates.
"""

from dataclasses import dataclass

import numpy as np

# longest match first: a guess from more context is more reliable
MATCH_ORDERS = (4, 3, 2)
# distinct continuations kept per key; a fuller key stops growing
MAX_CONTINUATIONS = 4
# an empty continuation column, and an empty key
_EMPTY = -1
_NO_KEY = np.uint64(0)
# slots probed before a lookup gives up; the table grows at half full,
# so a run this long is rare
_PROBES = 64
_FNV_OFFSET = np.uint64(1469598103934665603)
_FNV_PRIME = np.uint64(1099511628211)


@dataclass(frozen=True)
class NgramMatch:
    """One lookup's best continuation and the features that price it.

    Attributes:
        token: The continuation seen most often after the key.
        order: The key's tokens.
        fan_out: Distinct continuations seen after the key.
        share: The continuation's count over the key's count, the
            seed of the acceptance estimate.
        own: Whether the drafting document's own output last inserted
            the continuation.
    """

    token: int
    order: int
    fan_out: int
    share: float
    own: bool


@dataclass
class NgramMatches:
    """A chunk's lookups at once, one row per query; found says which hit."""

    found: np.ndarray
    token: np.ndarray
    order: np.ndarray
    fan_out: np.ndarray
    share: np.ndarray
    own: np.ndarray


def _hashes(order: int, contexts: np.ndarray) -> np.ndarray:
    """Return a 64-bit hash per row of the (rows, order) contexts, never 0."""
    out = np.full(len(contexts), _FNV_OFFSET ^ np.uint64(order), dtype=np.uint64)
    tokens = contexts.astype(np.uint64) + np.uint64(1)
    for column in range(order):
        out = (out ^ tokens[:, column]) * _FNV_PRIME
    return out | np.uint64(1)


class NgramCache:
    """Continuation counts for one query run, shared across documents."""

    def __init__(self, slots: int = 1 << 16):
        self._allocate(slots)
        self.used = 0

    def _allocate(self, slots: int) -> None:
        self.slots = slots
        self.mask = np.uint64(slots - 1)
        self.keys = np.zeros(slots, dtype=np.uint64)
        self.total = np.zeros(slots, dtype=np.int64)
        shape = (slots, MAX_CONTINUATIONS)
        self.continuation = np.full(shape, _EMPTY, dtype=np.int64)
        self.count = np.zeros(shape, dtype=np.int64)
        self.last = np.full(shape, _EMPTY, dtype=np.int64)

    # ---- the table -------------------------------------------------------

    def _find(self, hashes: np.ndarray) -> tuple:
        """Return each hash's slot and whether its key is there.

        A key not in the table gets the empty slot its probe reached,
        or, after _PROBES slots, the last one with found False.
        """
        slot = (hashes & self.mask).astype(np.int64)
        found = np.zeros(len(hashes), dtype=bool)
        done = np.zeros(len(hashes), dtype=bool)
        for _ in range(_PROBES):
            keys = self.keys[slot]
            hit = ~done & (keys == hashes)
            empty = ~done & (keys == _NO_KEY)
            found |= hit
            done |= hit | empty
            if done.all():
                break
            slot = np.where(done, slot, (slot + 1) & self.mask.astype(np.int64))
        return slot, found

    def _claim(self, hashes: np.ndarray) -> np.ndarray:
        """Return a slot per distinct hash, adding the keys not yet in the table."""
        if self.used + len(hashes) > self.slots // 2:
            self._grow(self.used + len(hashes))
        slot = np.empty(len(hashes), dtype=np.int64)
        pending = np.arange(len(hashes))
        while len(pending):
            at, found = self._find(hashes[pending])
            slot[pending] = at
            new = pending[~found]
            if not len(new):
                break
            # distinct new keys that probed to one empty slot: the first
            # takes it, the rest probe again past it
            _, first = np.unique(at[~found], return_index=True)
            winners = new[first]
            self.keys[slot[winners]] = hashes[winners]
            self.used += len(winners)
            pending = np.setdiff1d(new, winners, assume_unique=True)
        return slot

    def _grow(self, needed: int) -> None:
        """Double the table until needed keys fit at half full, keeping entries."""
        slots = self.slots
        while needed > slots // 2:
            slots *= 2
        kept = np.flatnonzero(self.keys != _NO_KEY)
        keys, total = self.keys[kept], self.total[kept]
        continuation, count, last = (self.continuation[kept], self.count[kept],
                                     self.last[kept])
        self._allocate(slots)
        self.used = 0
        slot = self._claim(keys)
        self.total[slot] = total
        self.continuation[slot] = continuation
        self.count[slot] = count
        self.last[slot] = last

    # ---- inserting ---------------------------------------------------------

    def add_many(self, order: int, contexts: np.ndarray, tokens: np.ndarray,
                 documents: np.ndarray) -> None:
        """Insert the continuations seen after keys of one order.

        Args:
            order: The keys' tokens.
            contexts: (rows, order) key tokens.
            tokens: The token that followed each context.
            documents: The document each row was written by.
        """
        if not len(tokens):
            return
        hashes = _hashes(order, np.asarray(contexts, dtype=np.int64))
        tokens = np.asarray(tokens, dtype=np.int64)
        documents = np.asarray(documents, dtype=np.int64)
        # one row per distinct (key, token), with its count and the
        # document that wrote it last
        order_by = np.lexsort((tokens, hashes))
        hashes, tokens = hashes[order_by], tokens[order_by]
        documents = documents[order_by]
        starts = np.flatnonzero(np.concatenate((
            [True], (hashes[1:] != hashes[:-1]) | (tokens[1:] != tokens[:-1]))))
        ends = np.concatenate((starts[1:], [len(hashes)]))
        hashes, tokens = hashes[starts], tokens[starts]
        documents = documents[ends - 1]
        counts = ends - starts
        distinct, inverse = np.unique(hashes, return_inverse=True)
        slot = self._claim(distinct)[inverse]
        column = self._columns(slot, tokens)
        placed = column >= 0
        slot, column, tokens = slot[placed], column[placed], tokens[placed]
        self.continuation[slot, column] = tokens
        np.add.at(self.count, (slot, column), counts[placed])
        np.add.at(self.total, slot, counts[placed])
        self.last[slot, column] = documents[placed]

    def _columns(self, slot: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """Return each (slot, token)'s continuation column, claiming empty ones.

        Returns -1 where the key already holds MAX_CONTINUATIONS other
        tokens. The rows are distinct (slot, token) pairs.
        """
        held = self.continuation[slot]
        matches = held == tokens[:, None]
        column = np.where(matches.any(axis=1), matches.argmax(axis=1), -1)
        pending = np.flatnonzero(column < 0)
        while len(pending):
            # one new token per slot per pass takes the first empty
            # column; the others wait for the next pass
            _, first = np.unique(slot[pending], return_index=True)
            rows = pending[first]
            empty = self.continuation[slot[rows]] == _EMPTY
            free = empty.any(axis=1)
            chosen = np.where(free, empty.argmax(axis=1), -1)
            column[rows] = chosen
            taking = rows[free]
            self.continuation[slot[taking], chosen[free]] = tokens[taking]
            # a token the key cannot hold is dropped, as a full key stops growing
            pending = np.setdiff1d(pending, rows, assume_unique=True)
            if not len(rows[free]):
                pending = pending[:0]
        return column

    def add(self, tokens, document: int, start: int = 0) -> None:
        """Insert the continuations the tokens show from start on.

        Args:
            tokens: One document's committed answer tokens so far.
            document: The document that wrote them.
            start: The first position not inserted yet.
        """
        tokens = np.asarray(tokens, dtype=np.int64)
        positions = np.arange(max(start, MATCH_ORDERS[-1]), len(tokens))
        for order in MATCH_ORDERS:
            at = positions[positions >= order]
            if not len(at):
                continue
            contexts = tokens[at[:, None] - order + np.arange(order)[None, :]]
            self.add_many(order, contexts, tokens[at],
                          np.full(len(at), document, dtype=np.int64))

    # ---- looking up ---------------------------------------------------------

    def lookup_many(self, contexts: np.ndarray, lengths: np.ndarray,
                    documents: np.ndarray) -> NgramMatches:
        """Return the longest match's best continuation for each query.

        Args:
            contexts: (rows, max order) tokens, most recent last; a
                query shorter than the width is left-padded.
            lengths: How many of each row's tokens are context.
            documents: The drafting document of each row.
        """
        rows = len(lengths)
        width = contexts.shape[1]
        out = NgramMatches(
            found=np.zeros(rows, dtype=bool),
            token=np.full(rows, _EMPTY, dtype=np.int64),
            order=np.zeros(rows, dtype=np.int64),
            fan_out=np.zeros(rows, dtype=np.int64),
            share=np.zeros(rows, dtype=np.float64),
            own=np.zeros(rows, dtype=bool))
        for order in MATCH_ORDERS:
            query = np.flatnonzero(~out.found & (lengths >= order))
            if not len(query):
                continue
            slot, found = self._find(_hashes(order, contexts[query, width - order:]))
            hit = query[found]
            slot = slot[found]
            if not len(hit):
                continue
            best = np.argmax(self.count[slot], axis=1)
            out.found[hit] = True
            out.token[hit] = self.continuation[slot, best]
            out.order[hit] = order
            out.fan_out[hit] = (self.continuation[slot] != _EMPTY).sum(axis=1)
            out.share[hit] = self.count[slot, best] / self.total[slot]
            out.own[hit] = self.last[slot, best] == documents[hit]
        return out

    def lookup(self, context, document: int) -> NgramMatch | None:
        """Return the longest match's best continuation, or None.

        Args:
            context: The answer's last tokens, most recent last.
            document: The drafting document.
        """
        width = max(MATCH_ORDERS)
        tokens = np.asarray(context, dtype=np.int64)[-width:]
        row = np.full((1, width), _EMPTY, dtype=np.int64)
        row[0, width - len(tokens):] = tokens
        found = self.lookup_many(row, np.array([len(tokens)]), np.array([document]))
        if not found.found[0]:
            return None
        return NgramMatch(token=int(found.token[0]), order=int(found.order[0]),
                          fan_out=int(found.fan_out[0]),
                          share=float(found.share[0]), own=bool(found.own[0]))

    def continuations(self, key) -> dict:
        """Return a key's continuation counts, token -> count, for inspection."""
        key = np.asarray(key, dtype=np.int64)
        slot, found = self._find(_hashes(len(key), key[None, :]))
        if not found[0]:
            return {}
        held = self.continuation[slot[0]]
        counts = self.count[slot[0]]
        return {int(token): int(count) for token, count in zip(held, counts)
                if token != _EMPTY}

    def __len__(self) -> int:
        """The keys in the table."""
        return self.used
