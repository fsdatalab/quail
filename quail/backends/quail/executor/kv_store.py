"""KV of document prefixes kept in host memory, keyed by the prefix's tokens.

The arena saves a document's prefix KV here when it frees the key, and
restores it when a later operator or query admits a document with the
same prefix tokens. A restore replaces the forward pass over the prefix
with a copy of the same bytes: the store is lossless.

On a CUDA device with nvCOMP installed, the store entropy-codes the KV
on the GPU with nvCOMP's ANS coder for 16-bit floats. A bf16 value's
sign and exponent byte takes few distinct values and its mantissa byte
is close to random, so the coder saves about 30% of the bytes; the copy
to and from the host moves the coded bytes. Without nvCOMP the store
keeps the pool's bytes.

Host memory is pinned in segments up to capacity_bytes. Pinning a
gibibyte takes a few hundred milliseconds and stalls other CUDA calls,
so a thread pins every segment while the worker boots, and the first
save waits for it. A saved prefix is cut into pieces of at most
PIECE_TOKENS tokens, each coded alone and kept in byte extents of the
segments. When the store is full and a save needs room, the prefix with
the lowest credit leaves (GreedyDual-Size): a prefix's credit is the
forward-pass seconds it saves per byte, so a long document outlasts
short ones, plus a clock that rises with each eviction, so a prefix no
query reads ages out.

Saves and restores are queued and run by flush, which the loop calls
before it admits documents and before a forward pass writes the arena.
A saved prefix can be restored once a flush has stored it. Every copy
runs on the current CUDA stream, so a restore reads what an earlier
save wrote. Coding a save waits for the GPU, to read the coded sizes;
a restore does not wait.
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

# tokens of one coded piece of a prefix
PIECE_TOKENS = 1024
# tokens one gather, coding, and copy moves; bounds the device memory a
# flush takes beside the arena
GROUP_TOKENS = 2048
# one pinned allocation
SEGMENT_BYTES = 1 << 28
# a worker's store pins at most this share of the host's memory, and at
# most HOST_BYTES
HOST_FRACTION = 0.5
HOST_BYTES = 64 << 30
# bytes nvCOMP's ANS coder models together
ANS_CHUNK_BYTES = 1 << 20
# nvCOMP reads and writes buffers at this alignment
ANS_ALIGNMENT = 16


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
    """One saved prefix: its tokens and where its coded pieces are."""

    tokens: int
    # per piece: (tokens, coded bytes, [(segment index, first byte, bytes)])
    pieces: list = field(default_factory=list)
    queued: int = 0    # queued restores that read the pieces
    credit: float = 0.0    # the eviction order: the lowest leaves first

    @property
    def stored_bytes(self) -> int:
        return sum(size for _, size, _ in self.pieces)


@dataclass
class _Segment:
    """One pinned host tensor of bytes and its free extents."""

    octets: object
    free: list = field(default_factory=list)    # sorted (first byte, bytes)

    def take(self, count: int):
        """Remove and return up to count bytes as (first byte, bytes)."""
        if not self.free:
            return None
        # the largest free extent keeps a piece in the fewest copies
        index = max(range(len(self.free)), key=lambda i: self.free[i][1])
        first, available = self.free[index]
        taken = min(count, available)
        if taken == available:
            del self.free[index]
        else:
            self.free[index] = (first + taken, available - taken)
        return first, taken

    def give(self, first: int, count: int) -> None:
        """Return bytes to the free list, joined with their neighbors."""
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


class _RawCoder:
    """Keeps the pool's bytes."""

    name = "raw"

    def stride(self, size: int) -> int:
        """Bytes between the starts of two coded pieces in one buffer."""
        return size

    def encode(self, values, sizes):
        """Code pieces of a byte buffer.

        Args:
            values: The pieces' bytes, one after another.
            sizes: Each piece's bytes.

        Returns:
            (buffer of coded pieces, each piece's offset, each piece's
            coded bytes).
        """
        offsets = np.concatenate(([0], np.cumsum(sizes)[:-1])).tolist()
        return values, offsets, list(sizes)

    def decode(self, buffer, offsets, sizes, value_sizes):
        """Return the pieces' bytes, one after another, from their coded buffer."""
        return buffer


class _AnsCoder:
    """Codes 16-bit floats with nvCOMP's ANS coder, on the GPU."""

    name = "ans"

    def __init__(self, torch, nvcomp, device):
        self.torch = torch
        self.nvcomp = nvcomp
        self.device = device
        self.stream = torch.cuda.current_stream(device).cuda_stream
        self.codec = nvcomp.Codec(
            algorithm="ANS", data_type="<f2", uncomp_chunk_size=ANS_CHUNK_BYTES,
            cuda_stream=self.stream)

    def stride(self, size: int) -> int:
        return -(-size // ANS_ALIGNMENT) * ANS_ALIGNMENT

    def _batch(self, buffer, offsets, sizes):
        return self.nvcomp.as_arrays(buffer, [int(offset) for offset in offsets],
                                     [int(size) for size in sizes],
                                     cuda_stream=self.stream)

    def encode(self, values, sizes):
        torch = self.torch
        halves = values.view(torch.float16)
        offsets = np.concatenate(([0], np.cumsum(sizes)[:-1])).tolist()
        largest = self.nvcomp.as_array(halves[:max(sizes) // 2],
                                       cuda_stream=self.stream)
        capacity = self.stride(self.codec.get_max_comp_buffer_size(largest))
        # the coder writes into memory torch owns, which a later pass can
        # reuse only after the copies queued behind the coding
        buffer = torch.empty(capacity * len(sizes), dtype=torch.uint8,
                             device=self.device)
        coded_offsets = [index * capacity for index in range(len(sizes))]
        coded = self.codec.encode(
            self._batch(halves, offsets, sizes),
            out=self._batch(buffer, coded_offsets, [capacity] * len(sizes)),
            compression_config=self.codec.compression_config(
                [int(size) for size in sizes]))
        # reading the sizes waits for the coding
        return buffer, coded_offsets, list(coded.buffer_sizes)

    def decode(self, buffer, offsets, sizes, value_sizes):
        torch = self.torch
        halves = torch.empty(sum(value_sizes) // 2, dtype=torch.float16,
                             device=self.device)
        value_offsets = np.concatenate(([0], np.cumsum(value_sizes)[:-1])).tolist()
        value_sizes = [int(size) for size in value_sizes]
        # a config made from the sizes parses no header, so nothing waits
        config = self.codec.decompression_config(
            self.codec.compression_config(value_sizes))
        self.codec.decode(
            self._batch(buffer, offsets, sizes),
            out=self._batch(halves, value_offsets, value_sizes),
            decompression_config=config)
        return halves.view(torch.uint8)


def _coder(torch, dtype, device):
    """The ANS coder for 16-bit pools on CUDA with nvCOMP, else the raw one."""
    if (str(device).startswith("cuda")
            and torch.empty(0, dtype=dtype).element_size() == 2):
        try:
            from nvidia import nvcomp
        except ImportError:
            return _RawCoder()
        return _AnsCoder(torch, nvcomp, device)
    return _RawCoder()


class KVStore:
    """Host memory for document-prefix KV, filled and read by a KVArena.

    Args:
        torch: The torch module.
        record_shape: (K and V tensors of all layers, KV heads, head
            dimension) of one token.
        dtype: The pool's element type.
        capacity_bytes: Host bytes the store may pin.
        device: The pool's device. For a CUDA device a thread pins the
            host memory at once; otherwise segments are made as saves
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
        self.coder = _coder(torch, dtype, device)
        self.segment_bytes = min(SEGMENT_BYTES, capacity_bytes)
        self.max_segments = (capacity_bytes // self.segment_bytes
                             if self.segment_bytes else 0)
        self.segments = []
        self.prefill_seconds = prefill_seconds or float
        self.saved = {}        # content id -> _Saved, once a flush stored it
        self._credits = []     # heap of (credit, content id), stale ones too
        self._clock = 0.0      # the credit of the last prefix dropped
        self._saves = {}       # content id -> pool rows, freed since the flush
        self._restores = []    # (_Saved, pool rows) to write into the pool
        self.reset_stats()
        self._closed = False
        self._thread = None
        if self.pinned and self.max_segments:
            self._thread = threading.Thread(
                target=self._pin, name="quail-kv-store", daemon=True)
            self._thread.start()

    def close(self) -> None:
        """Stop the pinning thread."""
        self._closed = True
        self._pinned()

    def _pin(self) -> None:
        """Pin every segment; runs on the pinning thread."""
        while not self._closed and len(self.segments) < self.max_segments:
            self.segments.append(self._new_segment())

    def _pinned(self) -> None:
        """Wait for the pinning thread, which no query runs beside."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def reset_stats(self) -> None:
        """Zero the counters a query's report reads."""
        self.saved_tokens = 0
        self.restored_tokens = 0
        self.dropped_tokens = 0
        self.skipped_tokens = 0    # freed prefixes the store had no room for
        self.flush_seconds = 0.0   # host seconds in flush, coding waits included

    def stats(self) -> dict:
        """Return the counters since reset_stats and what the store holds."""
        return {"coder": self.coder.name,
                "saved_tokens": self.saved_tokens,
                "restored_tokens": self.restored_tokens,
                "dropped_tokens": self.dropped_tokens,
                "skipped_tokens": self.skipped_tokens,
                "flush_seconds": self.flush_seconds,
                "held_tokens": self.held_tokens, "held_bytes": self.held_bytes,
                "pinned_bytes": len(self.segments) * self.segment_bytes,
                "capacity_bytes": self.max_segments * self.segment_bytes}

    # ---- what the store holds -------------------------------------------

    def has(self, content: bytes, tokens: int) -> bool:
        """Whether a flush has stored the prefix with this id and length."""
        saved = self.saved.get(content)
        return saved is not None and saved.tokens == tokens

    @property
    def held_tokens(self) -> int:
        return sum(saved.tokens for saved in self.saved.values())

    @property
    def held_bytes(self) -> int:
        return sum(saved.stored_bytes for saved in self.saved.values())

    def _new_segment(self) -> _Segment:
        # a tensor made in inference mode could not be written outside
        # it, where a query boundary flushes
        with self.torch.inference_mode(False):
            octets = self.torch.empty(self.segment_bytes, dtype=self.torch.uint8,
                                      pin_memory=self.pinned)
        return _Segment(octets, [(0, self.segment_bytes)])

    def _allocate(self, size: int):
        """Extents for size bytes, dropping low-credit prefixes; None if short."""
        extents, left = [], size
        while left:
            taken = None
            for index, segment in enumerate(self.segments):
                taken = segment.take(left)
                if taken is not None:
                    break
            if (taken is None and not self.pinned
                    and len(self.segments) < self.max_segments):
                index = len(self.segments)
                self.segments.append(self._new_segment())
                taken = self.segments[index].take(left)
            if taken is None:
                if not self._drop_lowest():
                    self._give(extents)
                    return None
                continue
            extents.append((index, *taken))
            left -= taken[1]
        return extents

    def _give(self, extents) -> None:
        for segment, first, count in extents:
            self.segments[segment].give(first, count)

    def _credit(self, content: bytes, saved) -> None:
        """Set a prefix's credit on a save or a restore."""
        saved.credit = self._clock + (
            self.prefill_seconds(saved.tokens) / max(1, saved.stored_bytes))
        heapq.heappush(self._credits, (saved.credit, content))

    def _drop_lowest(self) -> bool:
        """Free the prefix with the lowest credit that no queued restore reads."""
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
            for _, _, extents in saved.pieces:
                self._give(extents)
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
        if rows.numel() and content not in self.saved:
            self._saves.setdefault(content, rows)

    def restore(self, content: bytes, rows) -> None:
        """Queue a copy of a saved prefix into the pool rows."""
        saved = self.saved[content]
        self._credit(content, saved)
        saved.queued += 1
        self._restores.append((saved, rows))
        self.restored_tokens += saved.tokens

    # ---- the copies -------------------------------------------------------

    def flush(self, pool) -> None:
        """Run the queued saves, then the queued restores.

        Args:
            pool: The arena's K and V of every layer as one tensor of
                shape (tensors, rows, KV heads, head dimension).
        """
        if not self._saves and not self._restores:
            return
        started = time.perf_counter()
        with self.torch.inference_mode():
            self._save(pool)
            self._restore(pool)
        self.flush_seconds += time.perf_counter() - started

    @staticmethod
    def _groups(pieces):
        """Split (owner, tokens, rows) pieces into groups of GROUP_TOKENS."""
        groups, tokens = [[]], 0
        for piece in pieces:
            if groups[-1] and tokens + piece[1] > GROUP_TOKENS:
                groups.append([])
                tokens = 0
            groups[-1].append(piece)
            tokens += piece[1]
        return groups if groups[-1] else []

    @staticmethod
    def _pieces(owner, rows):
        return [(owner, min(PIECE_TOKENS, rows.numel() - start),
                 rows[start:start + PIECE_TOKENS])
                for start in range(0, rows.numel(), PIECE_TOKENS)]

    def _device_rows(self, pieces):
        rows = self.torch.cat([rows for _, _, rows in pieces])
        if not self.pinned:
            return rows
        return rows.pin_memory().to(self.device, non_blocking=True)

    def _save(self, pool) -> None:
        saves, self._saves = self._saves, {}
        if not saves:
            return
        self._pinned()
        torch = self.torch
        building = {content: _Saved(int(rows.numel()))
                    for content, rows in saves.items()}
        failed = set()
        pieces = [piece for content, rows in saves.items()
                  for piece in self._pieces(content, rows)]
        for group in self._groups(pieces):
            staged = pool.index_select(1, self._device_rows(group))
            values = staged.permute(1, 0, 2, 3).contiguous().view(torch.uint8)
            buffer, offsets, sizes = self.coder.encode(
                values.reshape(-1), [tokens * self.record_bytes
                                     for _, tokens, _ in group])
            for (content, tokens, _), offset, size in zip(group, offsets, sizes):
                if content in failed:
                    continue
                extents = self._allocate(int(size))
                if extents is None:
                    failed.add(content)
                    continue
                building[content].pieces.append((tokens, int(size), extents))
                for segment, first, count in extents:
                    self.segments[segment].octets[first:first + count].copy_(
                        buffer[offset:offset + count], non_blocking=True)
                    offset += count
        for content, saved in building.items():
            if content in failed:
                for _, _, extents in saved.pieces:
                    self._give(extents)
                self.skipped_tokens += saved.tokens
                continue
            self.saved[content] = saved
            self._credit(content, saved)
            self.saved_tokens += saved.tokens

    def _restore(self, pool) -> None:
        restores, self._restores = self._restores, []
        torch = self.torch
        pieces = []
        for saved, rows in restores:
            start = 0
            for tokens, size, extents in saved.pieces:
                pieces.append(((size, extents), tokens, rows[start:start + tokens]))
                start += tokens
        for group in self._groups(pieces):
            sizes = [size for (size, _), _, _ in group]
            strides = [self.coder.stride(size) for size in sizes]
            offsets = np.concatenate(([0], np.cumsum(strides)[:-1])).tolist()
            buffer = torch.empty(sum(strides), dtype=torch.uint8, device=self.device)
            for ((_, extents), _, _), offset in zip(group, offsets):
                for segment, first, count in extents:
                    buffer[offset:offset + count].copy_(
                        self.segments[segment].octets[first:first + count],
                        non_blocking=True)
                    offset += count
            tokens = sum(tokens for _, tokens, _ in group)
            values = self.coder.decode(
                buffer, offsets, sizes,
                [tokens * self.record_bytes for _, tokens, _ in group])
            staged = values.view(self.dtype).view(tokens, *self.record_shape)
            pool.index_copy_(1, self._device_rows(group), staged.permute(1, 0, 2, 3))
        for saved, _ in restores:
            saved.queued -= 1
