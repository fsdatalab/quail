"""KV of document prefixes kept in host memory, keyed by the prefix's tokens.

The arena saves a document's prefix KV here when it frees the key, and
restores it when a later operator or query admits a document with the
same prefix tokens. A restore replaces the forward pass over the prefix
with a copy.

Host memory is pinned and grows in segments up to capacity_bytes. A
saved prefix holds extents of token records; the least recently used
prefixes leave when a save needs room. A record is one token's K and V
of every layer, encoded by the codec:

- "raw" keeps the pool's bytes;
- "int8" keeps one signed byte per element and one fp16 scale per head
  vector, about half of bf16;
- "int4" keeps four bits per element and one fp16 scale per 32
  elements, about 28% of bf16.

Saves and restores are queued and run by flush, which the loop calls
before a forward pass writes the arena. Every copy runs on the current
CUDA stream, so a restore reads what an earlier save wrote.
"""

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from quail.common.tokens import token_array

CODECS = ("raw", "int8", "int4")
# tokens one gather, encode, and copy moves; bounds the device memory a
# flush takes beside the arena
FLUSH_TOKENS = 1024
SEGMENT_BYTES = 1 << 30
INT4_GROUP = 32
# the share of the host's memory a worker's store may pin
HOST_FRACTION = 0.5
CODEC = "int8"


def host_memory_bytes() -> int:
    """Return the memory this process may use: the container's limit, or the host's."""
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.is_file():
        text = limit.read_text().strip()
        if text.isdigit():
            return min(physical, int(text))
    return physical


def attach_host_store(arena) -> bool:
    """Give the arena a host store when its pools can use one.

    Returns:
        Whether a store was attached. An arena whose layers differ, slide,
        or keep checkpoints gets none.
    """
    keys, _ = arena.layer_kv(0)
    # the store pins no memory until its first save
    store = KVStore(
        arena.torch, (2 * len(arena.shapes), *keys.shape[1:]), arena.dtype,
        int(HOST_FRACTION * host_memory_bytes()), codec=CODEC,
        device=arena.device)
    try:
        arena.attach_store(store)
    except ValueError:
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


@dataclass
class _Segment:
    """One pinned host tensor of records and its free extents."""

    records: object
    free: list = field(default_factory=list)    # sorted (first record, count)

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
        codec: One of CODECS.
        device: The pool's device; host memory is pinned for a CUDA device.
    """

    def __init__(self, torch, record_shape, dtype, capacity_bytes: int,
                 codec: str = "int8", device="cuda"):
        if codec not in CODECS:
            raise ValueError(f"codec must be one of {CODECS}, got {codec!r}")
        self.torch = torch
        self.record_shape = tuple(record_shape)
        self.dtype = dtype
        self.codec = codec
        self.device = device
        self.pinned = str(device).startswith("cuda")
        elements = int(np.prod(self.record_shape))
        element_bytes = torch.empty(0, dtype=dtype).element_size()
        dim = self.record_shape[-1]
        self.group = dim if codec == "int8" else min(INT4_GROUP, dim)
        if codec != "raw" and dim % self.group:
            raise ValueError(f"head dimension {dim} is not a multiple of "
                             f"{self.group}")
        if codec == "int4" and self.group % 2:
            raise ValueError("int4 packs elements in pairs")
        self.record_bytes = {
            "raw": elements * element_bytes,
            "int8": elements + 2 * elements // self.group,
            "int4": elements // 2 + 2 * elements // self.group,
        }[codec]
        self.segment_records = max(
            FLUSH_TOKENS, min(SEGMENT_BYTES, capacity_bytes) // self.record_bytes)
        self.max_segments = capacity_bytes // (
            self.segment_records * self.record_bytes)
        self.segments = []
        self.saved = OrderedDict()    # content id -> _Saved, least recent first
        self._saves = []       # (segment index, first record, count, rows)
        self._restores = []    # the same, for records to write into the pool
        self._queued = []      # _Saved whose extents the queued copies use
        self.reset_stats()

    def reset_stats(self) -> None:
        """Zero the counters a query's report reads."""
        self.saved_tokens = 0
        self.restored_tokens = 0
        self.dropped_tokens = 0

    def stats(self) -> dict:
        """Return the counters since reset_stats and what the store holds."""
        return {"codec": self.codec, "saved_tokens": self.saved_tokens,
                "restored_tokens": self.restored_tokens,
                "dropped_tokens": self.dropped_tokens,
                "held_tokens": self.held_tokens, "held_bytes": self.held_bytes,
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

    def _allocate(self, tokens: int):
        """Extents for tokens records, dropping old prefixes; None if short."""
        extents, left = [], tokens
        while left:
            taken = None
            for index, segment in enumerate(self.segments):
                taken = segment.take(left)
                if taken is not None:
                    break
            if taken is None and len(self.segments) < self.max_segments:
                index = len(self.segments)
                # a tensor made in inference mode could not be written
                # outside it, where a query boundary flushes
                with self.torch.inference_mode(False):
                    records = self.torch.empty(
                        (self.segment_records, self.record_bytes),
                        dtype=self.torch.uint8, pin_memory=self.pinned)
                self.segments.append(_Segment(records, [(0, self.segment_records)]))
                taken = self.segments[index].take(left)
            if taken is None:
                if not self._drop_oldest():
                    for segment, first, count in extents:
                        self.segments[segment].give(first, count)
                    return None
                continue
            extents.append((index, *taken))
            left -= taken[1]
        return extents

    def _drop_oldest(self) -> bool:
        """Free the least recently used prefix no queued copy uses."""
        for content, saved in self.saved.items():
            if saved.queued:
                continue
            del self.saved[content]
            for segment, first, count in saved.extents:
                self.segments[segment].give(first, count)
            self.dropped_tokens += saved.tokens
            return True
        return False

    # ---- queueing, called by the arena ----------------------------------

    def save(self, content: bytes, rows) -> None:
        """Queue a copy of the pool rows of a prefix into the store.

        Args:
            content: The prefix's content id.
            rows: CPU tensor of the prefix's pool rows, in token order.
        """
        tokens = int(rows.numel())
        saved = self.saved.get(content)
        if saved is not None and saved.tokens == tokens:
            self.saved.move_to_end(content)
            return
        if saved is not None or not tokens:
            return
        extents = self._allocate(tokens)
        if extents is None:
            return
        saved = self.saved[content] = _Saved(tokens, extents)
        self._queue(self._saves, saved, rows)
        self.saved_tokens += tokens

    def restore(self, content: bytes, rows) -> None:
        """Queue a copy of a saved prefix into the pool rows."""
        saved = self.saved[content]
        self.saved.move_to_end(content)
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
        with self.torch.inference_mode():
            self._copy(pool)
        for saved in self._queued:
            saved.queued -= 1
        self._queued = []

    def _copy(self, pool) -> None:
        saves, self._saves = self._saves, []
        restores, self._restores = self._restores, []
        for segment, first, count, rows in self._runs(saves):
            staged = pool.index_select(1, self._device_rows(rows))
            records = self._encode(staged.permute(1, 0, 2, 3))
            self.segments[segment].records[first:first + count].copy_(
                records, non_blocking=True)
        for segment, first, count, rows in self._runs(restores):
            records = self.segments[segment].records[first:first + count].to(
                self.device, non_blocking=True)
            staged = self._decode(records)
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

    def _encode(self, values):
        """Records, as (tokens, record_bytes) bytes, of (tokens, ...) values."""
        torch = self.torch
        tokens = values.shape[0]
        if self.codec == "raw":
            return values.contiguous().view(torch.uint8).view(tokens, -1)
        levels = 127 if self.codec == "int8" else 7
        groups = values.float().reshape(tokens, -1, self.group)
        # the scale is rounded to fp16 first, so decoding multiplies by
        # the number encoding divided by
        scales = (groups.abs().amax(-1, keepdim=True) / levels).to(torch.float16)
        codes = (groups / scales.float().clamp_min(1e-30)).round_().clamp_(
            -levels, levels)
        if self.codec == "int8":
            codes = codes.to(torch.int8).view(torch.uint8)
        else:
            nibbles = (codes + 8).to(torch.uint8).view(tokens, -1, 2)
            codes = nibbles[..., 0] << 4 | nibbles[..., 1]
        return torch.cat((codes.reshape(tokens, -1),
                          scales.view(torch.uint8).view(tokens, -1)), dim=1)

    def _decode(self, records):
        """Values of shape (tokens, *record_shape) from their records."""
        torch = self.torch
        tokens = records.shape[0]
        shape = (tokens, *self.record_shape)
        if self.codec == "raw":
            return records.contiguous().view(self.dtype).view(shape)
        elements = int(np.prod(self.record_shape))
        code_bytes = elements if self.codec == "int8" else elements // 2
        scales = (records[:, code_bytes:].contiguous().view(torch.float16)
                  .view(tokens, -1, 1).float())
        packed = records[:, :code_bytes]
        if self.codec == "int8":
            codes = packed.contiguous().view(torch.int8).float()
        else:
            codes = torch.stack((packed >> 4, packed & 15), dim=-1).float() - 8
        values = codes.view(tokens, -1, self.group) * scales
        return values.view(shape).to(self.dtype)
