"""Per-document feature maxima for the pooling 'max'."""

import numpy as np

from quail.backends.quail.executor.chunk import InputStaging
from quail.backends.quail.executor.sae.encoder import SaeEncoder


class SaeMaxReadout:
    """Wait for the forward hook's maxima without encoding the final token again."""

    dtype = None
    trailing_rows = 1
    normalize = False

    def __init__(self, torch):
        self.torch = torch

    def submit(self, stream):
        event = self.torch.cuda.Event()
        event.record()
        return event, stream.shape[0]

    @staticmethod
    def result(handle):
        event, num_rows = handle
        event.synchronize()
        return [True] * num_rows


class _PinnedSlabs:
    """Pinned host rows handed out from slab allocations.

    A settle runs on the thread that feeds the GPU, and allocating
    pinned memory there costs milliseconds per call, so one slab serves
    many settles. A handle names its slab and row, and the harvest
    reads a slab's rows in one pass over a view of it.
    """

    def __init__(self, torch, width: int, rows_per_slab: int = 256):
        self.torch = torch
        self.width = width
        self.rows_per_slab = rows_per_slab
        self.slabs = []
        self.last_event = []    # per slab, the event of its latest copy
        self._next = rows_per_slab

    def take(self) -> tuple:
        """Return (slab, row) of a fresh pinned row."""
        if self._next == self.rows_per_slab:
            self.slabs.append(self.torch.empty(
                (self.rows_per_slab, self.width), dtype=self.torch.float32,
                pin_memory=True))
            self.last_event.append(None)
            self._next = 0
        self._next += 1
        return len(self.slabs) - 1, self._next - 1

    def complete(self, slab: int) -> bool:
        """Whether the slab is full and its latest copy has landed.

        Copies queue on one stream, so the latest event's completion
        covers every row of the slab.
        """
        full = slab < len(self.slabs) - 1 or self._next == self.rows_per_slab
        return full and self.last_event[slab].query()


class _SlotMatrix:
    """Per-document maxima as rows (slots) of one GPU matrix.

    One matrix lets a chunk fold every span in one scatter, and lets
    the slots that settles freed re-zero in one index_fill per chunk.
    """

    def __init__(self, torch, width: int, capacity: int = 512):
        self.torch = torch
        self.width = width
        self.matrix = torch.zeros((capacity, width), device="cuda",
                                  dtype=torch.float32)
        self.free = list(range(capacity - 1, -1, -1))
        self.dirty = []    # freed slots, zeroed before the next scatter

    def take(self) -> int:
        if not self.free:
            capacity = self.matrix.shape[0]
            grown = self.torch.zeros((2 * capacity, self.width),
                                     device="cuda", dtype=self.torch.float32)
            grown[:capacity] = self.matrix
            self.matrix = grown
            self.free.extend(range(2 * capacity - 1, capacity - 1, -1))
        return self.free.pop()

    def release(self, slot: int) -> None:
        self.free.append(slot)
        self.dirty.append(slot)


class SaeMaxAccumulator:
    """Per-document maxima of per-token top-k strengths, for the pooling 'max'.

    observe() runs inside forward_chunk over every chunk row. A row
    whose position is inside its document's prefix folds into the
    document's slot, which the pipeline's feature calls share; the
    other rows belong to the running call's question and fold into a
    slot consumed when the document's stage settles. A document's slot
    is freed once it has settled every registered feature call. Every
    step is one operation over the whole chunk, or over all settled
    documents, so the host work does not grow with the document count.
    """

    def __init__(self, torch, encoder: SaeEncoder, document_tokens):
        """Hold the encoder and the key-to-document-length lookup.

        Args:
            torch: The torch module.
            encoder: The model's SAE encoder.
            document_tokens: Callable from arena key to the document
                prefix's token count (the head and the document).
        """
        self.torch = torch
        self.encoder = encoder
        self.document_tokens = document_tokens
        self.document_slot = {}
        self.tail_slot = {}
        self.calls = 0        # feature calls registered on this pipeline
        self.settled = {}     # key -> calls settled so far
        self._slots = _SlotMatrix(torch, encoder.num_features)
        self._staging = InputStaging(torch)
        self._staging_index = 0
        self._pinned = _PinnedSlabs(torch, encoder.num_features)

    def register_call(self) -> None:
        """Count one feature call of the pipeline, before execution."""
        self.calls += 1

    def _slot(self, slots: dict, key) -> int:
        slot = slots.get(key)
        if slot is None:
            slot = slots[key] = self._slots.take()
        return slot

    def observe(self, stream, chunk) -> None:
        """Fold one chunk's rows into the document and question maxima.

        The chunk encodes in one call. numpy builds each row's document
        and question slot from the spans, one pinned copy carries them
        to the GPU, and one scatter folds every row's strengths into the
        slot its position selects. Nothing reads a data-dependent
        shape, so the hook never synchronizes with the forward pass.
        """
        spans = [span for span in chunk.row_spans if span[3] > span[1]]
        if not spans:
            return
        torch = self.torch
        indices, strengths = self.encoder.top_features(stream)
        count = len(spans)
        keys = [span[0] for span in spans]
        row0 = np.fromiter((span[1] for span in spans), np.int64, count)
        row1 = np.fromiter((span[3] for span in spans), np.int64, count)
        per_span = np.stack([
            np.fromiter((self._slot(self.document_slot, key) for key in keys),
                        np.int64, count),
            np.fromiter((self._slot(self.tail_slot, key) for key in keys),
                        np.int64, count),
            np.fromiter((self.document_tokens(key) for key in keys),
                        np.int64, count)])
        lengths = row1 - row0
        total = int(lengths.sum())
        per_row = np.repeat(per_span, lengths, axis=1)
        contiguous = (total == stream.shape[0] and row0[0] == 0
                      and np.array_equal(row0[1:], row1[:-1]))
        dirty = np.asarray(self._slots.dirty, dtype=np.int64)
        self._slots.dirty = []
        parts = [per_row.reshape(-1), dirty]
        if not contiguous:
            parts.insert(0, np.repeat(row0 - np.cumsum(lengths) + lengths,
                                      lengths) + np.arange(total))
        array = np.concatenate(parts)
        index = self._staging_index
        self._staging_index ^= 1
        host = self._staging.host(index, max(array.size, 1 << 16),
                                  self.torch.int64)
        host[:array.size].numpy()[:] = array
        block = self._staging.upload(index, array.size)
        at = 0
        if not contiguous:
            rows = block[:total]
            at = total
            indices = indices[rows]
            strengths = strengths[rows]
            positions = chunk.positions[rows]
        else:
            positions = chunk.positions
        document_slots = block[at:at + total]
        tail_slots = block[at + total:at + 2 * total]
        tokens = block[at + 2 * total:at + 3 * total]
        matrix = self._slots.matrix
        if dirty.size:
            matrix.index_fill_(0, block[at + 3 * total:], 0.0)
        slot = torch.where(positions < tokens, document_slots, tail_slots)
        flat = slot.unsqueeze(1) * self._slots.width + indices.long()
        matrix.view(-1).scatter_reduce_(0, flat.view(-1), strengths.reshape(-1),
                                        reduce="amax")

    def settle(self, key, *, collect: bool = True) -> tuple | None:
        """Return the handle of one document's combined maxima for one call.

        The question's slot is consumed; the document's stays for the
        pipeline's later feature calls and is freed after the last one.
        The handle names the pinned slab and row the maxima copy into.

        Args:
            key: The document's arena key.
            collect: Whether this call reads maxima. False discards the
                question's maxima after a call with last-token pooling.
        """
        torch = self.torch
        matrix = self._slots.matrix
        tail = self.tail_slot.pop(key, None)
        document = self.document_slot.get(key)
        handle = None
        if collect:
            source = tail if tail is not None else document
            if tail is not None and document is not None:
                torch.maximum(matrix[document], matrix[tail], out=matrix[tail])
            slab, row = self._pinned.take()
            host = self._pinned.slabs[slab][row]
            if source is None:
                host.zero_()
            else:
                host.copy_(matrix[source], non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            self._pinned.last_event[slab] = event
            handle = event, slab, row
        if tail is not None:
            self._slots.release(tail)
        self.settled[key] = self.settled.get(key, 0) + 1
        if self.settled[key] >= self.calls and document is not None:
            self._slots.release(self.document_slot.pop(key))
        return handle

    def harvest(self, handles, *, completed_only: bool) -> tuple:
        """Return (taken, lengths, indices, strengths) of settled handles.

        Each slab's rows unpack in one pass over a view of the slab, so
        the results come out slab by slab: ``taken`` lists the handle
        positions in that sequence, and ``lengths`` gives each one's
        feature count in the same sequence. With ``completed_only``,
        only slabs that are full and whose copies have landed are read,
        so the call never waits and can run while chunks are in
        flight; without it, every handle is read, waiting on each
        slab's latest copy.
        """
        count = len(handles)
        slabs = np.fromiter((handle[1] for handle in handles), np.int64, count)
        rows = np.fromiter((handle[2] for handle in handles), np.int64, count)
        order = np.lexsort((rows, slabs))
        slabs, rows = slabs[order], rows[order]
        taken, lengths, indices, strengths = [], [], [], []
        starts = np.flatnonzero(np.diff(slabs, prepend=-1))
        for start, stop in zip(starts, (*starts[1:], count)):
            slab = slabs[start]
            if completed_only:
                if not self._pinned.complete(slab):
                    continue
            else:
                self._pinned.last_event[slab].synchronize()
            block = self._pinned.slabs[slab].numpy()
            group = rows[start:stop]
            if group[-1] - group[0] == stop - start - 1:
                view = block[group[0]:group[-1] + 1]
            else:
                view = block[group]
            row_ids, columns = np.nonzero(view > 0.0)
            taken.append(order[start:stop])
            lengths.append(np.bincount(row_ids, minlength=stop - start))
            indices.append(columns.astype(np.int32))
            strengths.append(view[row_ids, columns])
        return tuple(
            np.concatenate(parts) if parts else np.empty(0, dtype)
            for parts, dtype in ((taken, np.int64), (lengths, np.int64),
                                 (indices, np.int32), (strengths, np.float32)))
