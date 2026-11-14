"""KV of document prefixes kept in host memory, keyed by the prefix's tokens.

The arena saves a document's prefix KV here when it frees the key, and
restores it when a later operator or query admits a document with the
same prefix tokens. A restore replaces the forward pass over the prefix
with a copy of the same bytes: the store is lossless.

Host memory is pinned in segments up to capacity_bytes. Pinning a
gibibyte takes a few hundred milliseconds, so a thread pins segments
ahead of the saves, and a save that finds no pinned room is skipped: the
prefix is saved the next time a query frees it. A saved prefix holds
extents of token records, one record per token: its K and V of every
layer. When the store is full and a save needs room, the prefix with
the lowest credit leaves (GreedyDual-Size): a prefix's credit is the
forward-pass seconds it saves per byte, so a long document outlasts
short ones, plus a clock that rises with each eviction, so a prefix no
query reads ages out.

Saves and restores are queued and run by flush, which the loop calls
before a forward pass writes the arena. Every copy runs on the current
CUDA stream, so a restore reads what an earlier save wrote.
"""

import hashlib
import heapq
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from quail.common.tokens import token_array

# tokens one gather and copy moves; bounds the device memory a flush
# takes beside the arena
FLUSH_TOKENS = 1024
# one pinned allocation; small enough that pinning one stalls no pass
SEGMENT_BYTES = 1 << 28
# pinned bytes the pinning thread keeps free ahead of the saves
SPARE_BYTES = 8 << 30
# a worker's store pins at most this share of the host's memory, and at
# most HOST_BYTES
HOST_FRACTION = 0.5
HOST_BYTES = 64 << 30


def host_memory_bytes() -> int:
    """Return the memory this process may use: the container's limit, or the host's."""
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.is_file():
        text = limit.read_text().strip()
        if text.isdigit():
            return min(physical, int(text))
    return physical


def attach_host_store(arena, prefill_seconds=None) -> bool:
    """Give the arena a host store when its pools can use one.

    Args:
        arena: The KVArena.
        prefill_seconds: Callable(tokens) -> forward-pass seconds of a
            prefix, which orders the store's evictions.

    Returns:
        Whether a store was attached. An arena whose layers differ, slide,
        or keep checkpoints gets none.
    """
    keys, _ = arena.layer_kv(0)
    store = KVStore(
        arena.torch, (2 * len(arena.shapes), *keys.shape[1:]), arena.dtype,
        min(HOST_BYTES, int(HOST_FRACTION * host_memory_bytes())),
        device=arena.device, prefill_seconds=prefill_seconds)
    try:
        arena.attach_store(store)
    except ValueError:
        store.close()
        return False
    return True


def content_id(tokens) -> bytes:
    """Return the store key of a prefix: a digest of its token ids."""
    return hashlib.blake2b(token_array(tokens).tobytes(), digest_size=16).digest()


@dataclass
class _Saved:
    """One saved prefix: its tokens and where its records are."""

    tokens: int
    extents: list     # (segment index, first record, record count)
    queued: int = 0    # queued copies that read or write the extents
    credit: float = 0.0    # the eviction order: the lowest leaves first


@dataclass
class _Segment:
    """One pinned host tensor of records and its free extents."""

    records: object
    free: list = field(default_factory=list)    # sorted (first record, count)

    @property
    def free_records(self) -> int:
        return sum(count for _, count in self.free)

    def take(self, count: int):
        """Remove and return up to count records as (first record, count)."""
        if not self.free:
            return None
        # the largest free extent keeps a save in the fewest copies
        index = max(range(len(self.free)), key=lambda i: self.free[i][1])
        first, available = self.free[index]
        taken = min(count, available)
        if taken == available:
            del self.free[index]
        else:
            self.free[index] = (first + taken, available - taken)
        return first, taken

    def give(self, first: int, count: int) -> None:
        """Return records to the free list, joined with their neighbors."""
        self.free.append((first, count))
        self.free.sort()
        merged = [self.free[0]]
        for start, length in self.free[1:]:
            last_start, last_length = merged[-1]
            if last_start + last_length == start:
                merged[-1] = (last_start, last_length + length)
            else:
                merged.append((start, length))
        self.free = merged


class KVStore:
    """Host memory for document-prefix KV, filled and read by a KVArena.

    Args:
        torch: The torch module.
        record_shape: (K and V tensors of all layers, KV heads, head
            dimension) of one token.
        dtype: The pool's element type.
        capacity_bytes: Host bytes the store may pin.
        device: The pool's device. For a CUDA device host memory is
            pinned, by a thread; otherwise segments are made as saves
            need them.
        prefill_seconds: Callable(tokens) -> the seconds a forward pass
            over a prefix of that length takes; None counts tokens.
    """

    def __init__(self, torch, record_shape, dtype, capacity_bytes: int,
                 device="cuda", prefill_seconds=None):
        self.torch = torch
        self.record_shape = tuple(record_shape)
        self.dtype = dtype
        self.device = device
        self.pinned = str(device).startswith("cuda")
        self.record_bytes = (int(np.prod(self.record_shape))
                             * torch.empty(0, dtype=dtype).element_size())
        self.segment_records = max(
            FLUSH_TOKENS, min(SEGMENT_BYTES, capacity_bytes) // self.record_bytes)
        self.max_segments = capacity_bytes // (
            self.segment_records * self.record_bytes)
        self.segments = []
        self.prefill_seconds = prefill_seconds or float
        self.saved = {}        # content id -> _Saved
        self._credits = []     # heap of (credit, content id), stale ones too
        self._clock = 0.0      # the credit of the last prefix dropped
        self._saves = []       # (segment index, first record, count, rows)
        self._restores = []    # the same, for records to write into the pool
        self._queued = []      # _Saved whose extents the queued copies use
        self.reset_stats()
        # the pinning thread appends to segments under the lock, and
        # wakes when the free pinned records fall under the spare
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._thread = None
        if self.pinned and self.max_segments:
            self._thread = threading.Thread(
                target=self._pin_ahead, name="quail-kv-store", daemon=True)
            self._thread.start()
            self._wake.set()

    def close(self) -> None:
        """Stop the pinning thread."""
        self._closed = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def reset_stats(self) -> None:
        """Zero the counters a query's report reads."""
        self.saved_tokens = 0
        self.restored_tokens = 0
        self.dropped_tokens = 0
        self.skipped_tokens = 0    # freed prefixes that found no pinned room
        self.flush_seconds = 0.0   # host seconds queueing the copies

    def stats(self) -> dict:
        """Return the counters since reset_stats and what the store holds."""
        return {"saved_tokens": self.saved_tokens,
                "restored_tokens": self.restored_tokens,
                "dropped_tokens": self.dropped_tokens,
                "skipped_tokens": self.skipped_tokens,
                "flush_seconds": self.flush_seconds,
                "held_tokens": self.held_tokens, "held_bytes": self.held_bytes,
                "pinned_bytes": len(self.segments) * self.segment_records
                * self.record_bytes,
                "capacity_bytes": self.max_segments * self.segment_records
                * self.record_bytes}

    # ---- what the store holds -------------------------------------------

    def has(self, content: bytes, tokens: int) -> bool:
        """Whether the store holds the prefix with this id and length."""
        saved = self.saved.get(content)
        return saved is not None and saved.tokens == tokens

    @property
    def held_tokens(self) -> int:
        return sum(saved.tokens for saved in self.saved.values())

    @property
    def held_bytes(self) -> int:
        return self.held_tokens * self.record_bytes

    def _new_segment(self) -> _Segment:
        # a tensor made in inference mode could not be written outside
        # it, where a query boundary flushes
        with self.torch.inference_mode(False):
            records = self.torch.empty(
                (self.segment_records, *self.record_shape), dtype=self.dtype,
                pin_memory=self.pinned)
        return _Segment(records, [(0, self.segment_records)])

    def _free_records(self) -> int:
        return sum(segment.free_records for segment in self.segments)

    def _pin_ahead(self) -> None:
        """Pin segments while the free pinned records are under the spare."""
        spare = SPARE_BYTES // self.record_bytes
        while True:
            self._wake.wait()
            self._wake.clear()
            while not self._closed and len(self.segments) < self.max_segments:
                with self._lock:
                    if self._free_records() >= spare:
                        break
                segment = self._new_segment()
                with self._lock:
                    self.segments.append(segment)
            if self._closed:
                return

    def _allocate(self, tokens: int):
        """Extents for tokens records, or None when no pinned room is ready.

        A full store drops its lowest-credit prefixes to make room. One
        still pinning segments drops nothing: the save is skipped.
        """
        extents, left = [], tokens
        with self._lock:
            while left:
                taken = None
                for index, segment in enumerate(self.segments):
                    taken = segment.take(left)
                    if taken is not None:
                        break
                if (taken is None and self._thread is None
                        and len(self.segments) < self.max_segments):
                    index = len(self.segments)
                    self.segments.append(self._new_segment())
                    taken = self.segments[index].take(left)
                if taken is None and (len(self.segments) < self.max_segments
                                      or not self._drop_lowest()):
                    for segment, first, count in extents:
                        self.segments[segment].give(first, count)
                    extents = None
                    break
                if taken is not None:
                    extents.append((index, *taken))
                    left -= taken[1]
        self._wake.set()
        return extents

    def _credit(self, content: bytes, saved) -> None:
        """Set a prefix's credit on a save or a restore."""
        saved.credit = self._clock + (
            self.prefill_seconds(saved.tokens) / (saved.tokens * self.record_bytes))
        heapq.heappush(self._credits, (saved.credit, content))

    def _drop_lowest(self) -> bool:
        """Free the prefix with the lowest credit that no queued copy uses."""
        queued, dropped = [], False
        while self._credits and not dropped:
            credit, content = heapq.heappop(self._credits)
            saved = self.saved.get(content)
            if saved is None or saved.credit != credit:
                continue        # dropped, or credited again since
            if saved.queued:
                queued.append((credit, content))
                continue
            del self.saved[content]
            for segment, first, count in saved.extents:
                self.segments[segment].give(first, count)
            self.dropped_tokens += saved.tokens
            self._clock = credit
            dropped = True
        for entry in queued:
            heapq.heappush(self._credits, entry)
        return dropped

    # ---- queueing, called by the arena ----------------------------------

    def save(self, content: bytes, rows) -> None:
        """Queue a copy of the pool rows of a prefix into the store.

        Args:
            content: The prefix's content id.
            rows: CPU tensor of the prefix's pool rows, in token order.
        """
        tokens = int(rows.numel())
        saved = self.saved.get(content)
        if saved is not None or not tokens:
            return
        extents = self._allocate(tokens)
        if extents is None:
            self.skipped_tokens += tokens
            return
        saved = self.saved[content] = _Saved(tokens, extents)
        self._credit(content, saved)
        self._queue(self._saves, saved, rows)
        self.saved_tokens += tokens

    def restore(self, content: bytes, rows) -> None:
        """Queue a copy of a saved prefix into the pool rows."""
        saved = self.saved[content]
        self._credit(content, saved)
        self._queue(self._restores, saved, rows)
        self.restored_tokens += saved.tokens

    def _queue(self, copies, saved, rows) -> None:
        offset = 0
        for segment, first, count in saved.extents:
            copies.append((segment, first, count, rows[offset:offset + count]))
            offset += count
        saved.queued += 1
        self._queued.append(saved)

    # ---- the copies -------------------------------------------------------

    def flush(self, pool) -> None:
        """Run the queued saves, then the queued restores.

        Args:
            pool: The arena's K and V of every layer as one tensor of
                shape (tensors, rows, KV heads, head dimension).
        """
        started = time.perf_counter()
        with self.torch.inference_mode():
            self._copy(pool)
        self.flush_seconds += time.perf_counter() - started
        for saved in self._queued:
            saved.queued -= 1
        self._queued = []

    def _copy(self, pool) -> None:
        saves, self._saves = self._saves, []
        restores, self._restores = self._restores, []
        for segment, first, count, rows in self._runs(saves):
            staged = pool.index_select(1, self._device_rows(rows))
            self.segments[segment].records[first:first + count].copy_(
                staged.permute(1, 0, 2, 3).contiguous(), non_blocking=True)
        for segment, first, count, rows in self._runs(restores):
            staged = self.segments[segment].records[first:first + count].to(
                self.device, non_blocking=True)
            pool.index_copy_(1, self._device_rows(rows), staged.permute(1, 0, 2, 3))

    def _runs(self, copies):
        """Join copies of adjacent records into runs of at most FLUSH_TOKENS."""
        torch = self.torch
        pieces = []
        for segment, first, count, rows in sorted(copies, key=lambda c: c[:2]):
            for start in range(0, count, FLUSH_TOKENS):
                size = min(FLUSH_TOKENS, count - start)
                pieces.append((segment, first + start, size,
                               rows[start:start + size]))
        runs = []
        for segment, first, count, rows in pieces:
            if runs:
                run_segment, run_first, held, run_rows = runs[-1]
                if (run_segment == segment and run_first + held == first
                        and held + count <= FLUSH_TOKENS):
                    run_rows.append(rows)
                    runs[-1] = (segment, run_first, held + count, run_rows)
                    continue
            runs.append((segment, first, count, [rows]))
        return [(segment, first, count, torch.cat(rows))
                for segment, first, count, rows in runs]

    def _device_rows(self, rows):
        if not self.pinned:
            return rows
        return rows.pin_memory().to(self.device, non_blocking=True)

