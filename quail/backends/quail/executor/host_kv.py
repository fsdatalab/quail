"""Host memory that keeps document KV between queries.

The arena frees a document's pages when an operator finishes with it,
when retention evicts it, and when the query ends. With a store
attached, the arena first copies the document's prefix rows, the
prompt head and the document, into pinned host memory. The next
operator that reads the same document, in this query or a later one,
allocates pages and copies the rows back instead of computing them.

Entries are keyed by a hash of the prefix token ids, so a saved prefix
matches only a document shown under the same head, tokenized the same
way. The host pool is a ring over token rows: a new entry takes the
rows after the newest one and drops the oldest entries in its way.
Every entry is one contiguous run of rows in each pool, so a save is
one gather on the compute stream and one copy to the host per layer
group, and a restore is one copy from the host and one scatter.

A key's rows are saved only once an operator computed or restored them
(mark_written); pages reserved for a chunk that never ran hold nothing.
"""

import hashlib
import time
from collections import deque

import numpy as np

from quail.catalog.tokens import token_array

# device staging buffers a layer group may hold while their copies to
# the host are in flight; a 300-token document of a 4B model takes
# about 45 MB
MAX_STAGING = 16

def prefix_hash(prefix) -> bytes:
    """Hash a prefix's token ids: the document head, then the document."""
    ids = np.ascontiguousarray(token_array(prefix), dtype=np.int32)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.int64(len(ids)).tobytes())
    digest.update(ids.tobytes())
    return digest.digest()


class _Ring:
    """A ring of rows: entries are appended in order and the oldest make room.

    Attributes:
        num_rows: Rows in the ring.
        head: The row the next entry starts at.
        entries: (start, rows) of every entry, oldest first.
    """

    def __init__(self, num_rows: int):
        self.num_rows = num_rows
        self.head = 0
        self.entries = deque()

    def plan(self, rows: int):
        """Where `rows` contiguous rows would go: (start, entries in the way).

        The start is None when the ring is too small for the run.
        Nothing changes until take().
        """
        if rows > self.num_rows or rows <= 0:
            return None, ()
        head = self.head
        evicted = []
        index = 0
        while True:
            if head + rows > self.num_rows:
                # the run would cross the end: the rows up to it go
                # unused, and the entries still ahead of the head are
                # the oldest, so they go
                while index < len(self.entries) and self.entries[index][0] >= head:
                    evicted.append(self.entries[index])
                    index += 1
                head = 0
                continue
            if index >= len(self.entries):
                break
            oldest = self.entries[index][0]
            # free rows run from the head to the oldest entry when it
            # sits ahead of the head, else to the end of the ring
            if oldest >= head and oldest - head < rows:
                evicted.append(self.entries[index])
                index += 1
                continue
            break
        return head, tuple(evicted)

    def take(self, rows: int):
        """Claim `rows` contiguous rows, returning (start, evicted entries)."""
        start, evicted = self.plan(rows)
        if start is None:
            return None, ()
        for _ in evicted:
            self.entries.popleft()
        self.head = start + rows
        self.entries.append((start, rows))
        return start, evicted

    def pop_oldest(self):
        """Drop the oldest entry, returning it or None."""
        return self.entries.popleft() if self.entries else None

    def drop(self, start: int) -> None:
        """Forget the entry starting at `start`."""
        for index, entry in enumerate(self.entries):
            if entry[0] == start:
                del self.entries[index]
                return


class HostKV:
    """Pinned host pools mirroring an arena's layers, and the entries in them.

    One pool per layer group: the full-attention layers of one KV
    shape and the sliding layers of one shape each form a group. A
    group's pool is (rows, layers in the group, 2, heads * d_head), so
    an entry's rows hold every layer of the group as one block. When
    the group's layers sit at aligned offsets of the arena's storage,
    the block gathers from it in one launch; otherwise one per layer.

    Args:
        arena: The KVArena whose layers the pools mirror.
        capacity_bytes: Host bytes for KV rows and checkpoints.
        torch: The torch module.
        pinned: Allocate pinned (page-locked) host memory, which the
            copy engine reads directly, and copy on a side stream.
            None pins on a CUDA arena; False for CPU tests.
    """

    def __init__(self, arena, capacity_bytes: int, torch, *, pinned=None):
        self.torch = torch
        self.arena = arena
        self.device = arena.device
        self.pinned = (str(self.device).startswith("cuda")
                       if pinned is None else pinned)
        self.capacity_bytes = int(capacity_bytes)
        self._groups = self._layer_groups(arena)
        row_bytes = sum(2 * len(layers) * width * arena.dtype.itemsize
                        for layers, width, _ in self._groups)
        state_bytes = 0
        if arena.gdn_state is not None:
            state_bytes = (arena.gdn_state[0].numel()
                           * arena.gdn_state.element_size()
                           + arena.conv_state[0].numel()
                           * arena.conv_state.element_size())
        # a hybrid model keeps one checkpoint beside each entry, so the
        # rows per slot follow the arena's own ratio of rows to slots
        rows_per_slot = 0
        if state_bytes:
            rows_per_slot = max(1, arena.num_pages * arena.page_tokens
                                // max(1, arena.num_state_slots))
        slot_share = state_bytes / rows_per_slot if rows_per_slot else 0.0
        self.num_rows = int(self.capacity_bytes // max(1.0, row_bytes + slot_share))
        self.num_slots = (-(-self.num_rows // rows_per_slot)
                          if rows_per_slot else 0)
        self.row_bytes = row_bytes
        self.pools = [
            torch.empty((self.num_rows, len(layers), 2, width),
                        dtype=arena.dtype, device="cpu", pin_memory=self.pinned)
            for layers, width, _ in self._groups]
        self.gdn_state = self.conv_state = None
        if self.num_slots:
            self.gdn_state = torch.empty(
                (self.num_slots, *arena.gdn_state.shape[1:]),
                dtype=arena.gdn_state.dtype, device="cpu", pin_memory=self.pinned)
            self.conv_state = torch.empty(
                (self.num_slots, *arena.conv_state.shape[1:]),
                dtype=arena.conv_state.dtype, device="cpu", pin_memory=self.pinned)
        self._free_slots = list(range(self.num_slots - 1, -1, -1))
        self.ring = _Ring(self.num_rows)
        self.entries = {}       # prefix hash -> (start, tokens, slot or None)
        self._by_start = {}     # ring start -> prefix hash
        self._keys = {}         # arena key -> (prefix hash, prefix tokens)
        self._written = set()   # keys whose prefix rows hold computed KV
        self._fresh = set()     # hashes saved during this query
        self._staging = {}      # (group, side) -> device staging buffers
        self._flat = [self._flat_view(layers, width)
                      for layers, width, _ in self._groups]
        self.stream = (torch.cuda.Stream(device=self.device)
                       if self.pinned else None)
        self.reset_stats()

    @staticmethod
    def _layer_groups(arena):
        """Group layers by pool and KV shape: (layer ids, row width, sliding)."""
        groups = {}
        for layer, (heads, dim) in enumerate(arena.shapes):
            sliding = layer in arena.sliding_layers
            groups.setdefault((sliding, heads * dim), []).append(layer)
        return [(tuple(layers), width, sliding)
                for (sliding, width), layers in sorted(groups.items())]

    def _flat_view(self, layers, width):
        """The storage as rows of `width`, and each layer's K and V row base.

        None when a layer of the group does not start on a row boundary.
        """
        arena = self.arena
        storage = getattr(arena, "storage", None)
        if storage is None or storage.numel() % width:
            return None
        bases = []
        for layer in layers:
            for tensor in arena.layer_kv(layer):
                offset = tensor.storage_offset()
                if offset % width:
                    return None
                bases.append(offset // width)
        # the bases live on the device: a copy per save would wait on
        # the stream's queued passes
        return storage.view(-1, width), self.torch.tensor(
            bases, dtype=self.torch.int64, device=self.device)

    def rebind(self) -> None:
        """Find the layers again after the arena rebuilt its pools."""
        self._flat = [self._flat_view(layers, width)
                      for layers, width, _ in self._groups]

    @property
    def bytes(self) -> int:
        """Host bytes the pools and checkpoint slots hold."""
        return (sum(pool.numel() * pool.element_size() for pool in self.pools)
                + (0 if self.gdn_state is None else
                   self.gdn_state.numel() * self.gdn_state.element_size()
                   + self.conv_state.numel() * self.conv_state.element_size()))

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0
        self.saved_keys = 0
        self.saved_tokens = 0
        self.restored_keys = 0
        self.restored_tokens = 0
        self.skipped_keys = 0
        self.host_s = 0.0

    def report(self) -> dict:
        """The store's counters for the query so far, and its size."""
        return {
            "host_kv_hits": self.hits,
            "host_kv_misses": self.misses,
            "host_kv_saved_keys": self.saved_keys,
            "host_kv_saved_tokens": self.saved_tokens,
            "host_kv_restored_keys": self.restored_keys,
            "host_kv_restored_tokens": self.restored_tokens,
            "host_kv_skipped_keys": self.skipped_keys,
            "host_kv_s": round(self.host_s, 4),
            "host_kv_entries": len(self.entries),
            "host_kv_rows": self.num_rows,
        }

    # ---- the keys of one query -----------------------------------------

    def begin_query(self) -> None:
        """Forget the last query's keys: a key names a document per query."""
        self._keys.clear()
        self._written.clear()
        self._fresh.clear()
        self.reset_stats()

    def register(self, key, prefix, prefix_tokens: int) -> bytes:
        """Name the prefix a key holds; returns its hash.

        A key renamed to another prefix holds no saved KV until an
        operator writes it again.
        """
        started = time.perf_counter()
        digest = prefix_hash(prefix)
        named = self._keys.get(key)
        if named is not None and named[0] != digest:
            self._written.discard(key)
        self._keys[key] = (digest, prefix_tokens)
        self.host_s += time.perf_counter() - started
        return digest

    def mark_written(self, key) -> None:
        """The key's prefix rows now hold KV an operator computed."""
        if key in self._keys:
            self._written.add(key)

    def holds(self, digest: bytes) -> bool:
        return digest in self.entries

    # ---- save ----------------------------------------------------------

    def before_free(self, key) -> bool:
        """Save a registered key's prefix rows before the arena frees them.

        Returns whether the rows were saved. The key must be resident.
        """
        named = self._keys.get(key)
        if named is None or key not in self._written:
            return False
        digest, prefix_tokens = named
        self._written.discard(key)
        arena = self.arena
        if arena.accounting.tokens[key] < prefix_tokens or digest in self.entries:
            return False
        if self.num_slots and arena.checkpoint_tokens(key) != prefix_tokens:
            return False
        started = time.perf_counter()
        start, evicted = self.ring.plan(prefix_tokens)
        # a store smaller than the query's documents keeps the first
        # ones it saved: a save that would drop one of them would move
        # the rows for nothing
        if start is None or any(self._by_start[old_start] in self._fresh
                                for old_start, _ in evicted):
            self.skipped_keys += 1
            return False
        slot = None
        if self.num_slots and not self._free_slots and not any(
                self.entries[self._by_start[old_start]][2] is not None
                for old_start, _ in evicted):
            oldest = self.ring.entries[0] if self.ring.entries else None
            if oldest is None or self._by_start[oldest[0]] in self._fresh:
                self.skipped_keys += 1
                return False
            self.ring.pop_oldest()
            self._forget(oldest[0])
        self.ring.take(prefix_tokens)
        for old_start, _ in evicted:
            self._forget(old_start)
        if self.num_slots:
            slot = self._free_slots.pop()
        self._save_rows(key, start, prefix_tokens, slot)
        self.entries[digest] = (start, prefix_tokens, slot)
        self._by_start[start] = digest
        self._fresh.add(digest)
        self.saved_keys += 1
        self.saved_tokens += prefix_tokens
        self.host_s += time.perf_counter() - started
        return True

    def _forget(self, start):
        digest = self._by_start.pop(start, None)
        if digest is None:
            return
        _, _, slot = self.entries.pop(digest)
        if slot is not None:
            self._free_slots.append(slot)

    def _entry_rows(self, key, tokens, sliding):
        """Device row indices of the key's prefix in one pool, and its offset.

        A sliding pool holds the rows from the window origin on, so the
        offset says where in the entry those rows sit.
        """
        arena = self.arena
        if sliding:
            origin = arena.origin(tokens)
            offset = origin - arena.sliding_start(key)
            rows = arena.capacity_rows_sliding(key)[offset:offset + tokens - origin]
            return self._to_device(rows), origin
        return self._to_device(arena.capacity_rows(key)[:tokens]), 0

    def _to_device(self, rows):
        if not self.pinned:
            return rows.to(self.device)
        # a pageable copy would wait on the stream's queued work
        return rows.pin_memory().to(self.device, non_blocking=True)

    def _save_rows(self, key, start, tokens, slot):
        """Copy the key's first `tokens` rows of every layer to the host."""
        torch = self.torch
        arena = self.arena
        compute = torch.cuda.current_stream() if self.pinned else None
        for group, ((layers, width, sliding), pool, flat) in enumerate(zip(
                self._groups, self.pools, self._flat)):
            rows, origin = self._entry_rows(key, tokens, sliding)
            staging, buffer = self._staging_for(
                group, (len(rows), len(layers), 2, width))
            if flat is not None:
                view, bases = flat
                index = (rows[:, None] + bases[None, :]).reshape(-1)
                torch.index_select(view, 0, index, out=staging.view(-1, width))
            else:
                for place, layer in enumerate(layers):
                    for side, tensor in enumerate(arena.layer_kv(layer)):
                        staging[:, place, side] = tensor.view(-1, width)[rows]
            self._to_host(pool[start + origin:start + tokens], staging, compute,
                          buffer)
        if slot is not None:
            source = arena.accounting.slots[key]
            self._to_host(self.gdn_state[slot], arena.gdn_state[source], compute)
            self._to_host(self.conv_state[slot], arena.conv_state[source], compute)

    def _staging_for(self, group, shape):
        """A device buffer of `shape` whose last copy to the host has landed.

        Buffers are reused once their copy's event has fired, checked
        without waiting, and a new one is allocated while fewer than
        MAX_STAGING are in flight. Past that the host waits for the
        oldest copy, which paces saves at the copy engine's rate
        instead of filling the device with staging buffers.
        """
        torch = self.torch
        buffers = self._staging.setdefault(group, deque())
        if buffers and (buffers[0][1] is None or buffers[0][1].query()):
            entry = buffers.popleft()
        elif len(buffers) >= MAX_STAGING:
            entry = buffers.popleft()
            entry[1].synchronize()
        else:
            entry = [None, None]
        buffers.append(entry)
        numel = 1
        for size in shape:
            numel *= size
        if entry[0] is None or entry[0].numel() < numel:
            entry[0] = torch.empty(numel, dtype=self.arena.dtype, device=self.device)
        return entry[0][:numel].view(shape), entry

    def _to_host(self, destination, source, compute, staging=None):
        """Copy a device tensor to the host on the copy stream.

        staging is the buffer entry the source came from, which records
        the copy's event so the buffer is reused only after it.
        """
        if compute is None:
            destination.copy_(source)
            return
        # the gather is in stream order after the pass that wrote the
        # rows, so the arena may reuse them as soon as it is queued
        self.stream.wait_stream(compute)
        with self.torch.cuda.stream(self.stream):
            destination.copy_(source, non_blocking=True)
        if staging is not None:
            staging[1] = self.torch.cuda.Event()
            staging[1].record(self.stream)
        source.record_stream(self.stream)

    def flush(self) -> None:
        """Wait for every copy in flight."""
        if self.stream is not None:
            self.stream.synchronize()

    # ---- restore -------------------------------------------------------

    def restore(self, key, digest: bytes) -> bool:
        """Copy a saved prefix into the key's reserved pages.

        The key holds pages for the prefix; under a sliding pool its
        window is trimmed here, as a computed key's is after its pass.
        Returns False, leaving the pages as they were, when the entry
        is gone.
        """
        entry = self.entries.get(digest)
        if entry is None:
            self.misses += 1
            return False
        start, tokens, slot = entry
        arena = self.arena
        if arena.accounting.tokens[key] != tokens:
            raise ValueError(
                f"{key!r} holds {arena.accounting.tokens[key]} tokens; the "
                f"saved prefix has {tokens}")
        started = time.perf_counter()
        torch = self.torch
        compute = torch.cuda.current_stream() if self.pinned else None
        if arena.has_sliding:
            arena.trim_window(key)
        for (layers, width, sliding), pool, flat in zip(
                self._groups, self.pools, self._flat):
            rows, origin = self._entry_rows(key, tokens, sliding)
            staging = self._from_host(pool[start + origin:start + tokens], compute)
            if flat is not None:
                view, bases = flat
                index = (rows[:, None] + bases[None, :]).reshape(-1)
                view.index_copy_(0, index, staging.view(-1, width))
            else:
                for place, layer in enumerate(layers):
                    for side, tensor in enumerate(arena.layer_kv(layer)):
                        tensor.view(-1, width).index_copy_(
                            0, rows, staging[:, place, side])
        if slot is not None:
            target = arena.accounting.slots[key]
            arena.gdn_state[target].copy_(
                self._from_host(self.gdn_state[slot], compute))
            arena.conv_state[target].copy_(
                self._from_host(self.conv_state[slot], compute))
            arena.record_checkpoint(key, tokens)
        self._written.add(key)
        self.hits += 1
        self.restored_keys += 1
        self.restored_tokens += tokens
        self.host_s += time.perf_counter() - started
        return True

    def _from_host(self, source, compute):
        """Copy a host tensor to a device staging tensor the compute stream reads."""
        if compute is None:
            return source.clone()
        torch = self.torch
        with torch.cuda.stream(self.stream):
            staging = source.to(self.device, non_blocking=True)
        compute.wait_stream(self.stream)
        staging.record_stream(compute)
        return staging
