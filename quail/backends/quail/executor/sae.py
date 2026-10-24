"""The SAE encoder for AI.SAE_FEATURES: weights, encode, and poolings.

The encode follows the published SAE: per token with hidden state h,
pre = h @ W_enc.T + b_enc, keep the sae_keep largest values, and set
the rest and any negatives to 0. The pooling 'last' encodes each
answer row through a readout; 'max' encodes every chunk row through
the pipeline's row observer and keeps each document's per-feature
maxima across the pipeline's feature calls.
"""


import numpy as np

from quail.backends.quail.executor.readout import _HostReadout

# bytes of fp32 pre-activation one torch encode block materializes: it
# lives in the device memory the arena leaves unclaimed (POOL_FRACTION
# in quail.cost.budgets), and fewer, larger blocks mean fewer launches
ENCODE_BLOCK_BYTES = 256 * 2**20


# the fused path's histogram bins and candidate buffer capacity: bins
# are quarter octaves of the positive pre-activations, and a row's
# candidates are the values in and above its threshold bin, at most
# keep plus the threshold bin's own count
HISTOGRAM_BINS = 256
CANDIDATE_CAP = 512


class _SaeKernels:
    """The fused encode kernels, compiled once per process.

    Both kernels tile the encode GEMM (rows x hidden) x (hidden x
    features) and never write the pre-activation out. The first builds
    a per-row histogram of the positive values; the second emits the
    values at or above each row's threshold into a bounded candidate
    buffer, which torch cuts to the exact top k. Triton imports lazily
    so the torch path works without it, and its compile cache keys on
    the kernel objects, so the class holds exactly one of each. The
    layout follows the conventions of the attention kernels
    (quail.backends.quail.executor.attention).
    """

    histogram = None
    emit = None

    @classmethod
    def compiled(cls):
        if cls.histogram is not None:
            return cls
        import triton
        import triton.language as tl

        @triton.jit
        def _tile(h_ptr, w_ptr, b_ptr, stride_h, stride_w, row0, feat0, rows,
                  T: tl.constexpr, F: tl.constexpr, K: tl.constexpr,
                  H: tl.constexpr):
            """One (T x F) tile of relu(h @ W.T + b), accumulated in fp32."""
            row = row0 + tl.arange(0, T)
            feature = feat0 + tl.arange(0, F)
            acc = tl.zeros((T, F), dtype=tl.float32)
            for k in range(0, H, K):
                cols = k + tl.arange(0, K)
                h = tl.load(h_ptr + row[:, None] * stride_h + cols[None, :],
                            mask=(row < rows)[:, None], other=0.0)
                w = tl.load(w_ptr + feature[:, None] * stride_w + cols[None, :])
                acc += tl.dot(h, tl.trans(w))
            acc += tl.load(b_ptr + feature)[None, :]
            return row, feature, tl.maximum(acc, 0.0)

        @triton.jit
        def sae_histogram(h_ptr, w_ptr, b_ptr, hist_ptr, stride_h, stride_w,
                          rows, T: tl.constexpr, F: tl.constexpr,
                          K: tl.constexpr, H: tl.constexpr, BINS: tl.constexpr):
            row, _feature, acc = _tile(
                h_ptr, w_ptr, b_ptr, stride_h, stride_w,
                tl.program_id(0) * T, tl.program_id(1) * F, rows,
                T=T, F=F, K=K, H=H)
            positive = (acc > 0.0) & (row < rows)[:, None]
            safe = tl.where(positive, acc, 1.0)
            bucket = tl.floor((tl.math.log2(safe) + 32.0) * 4.0).to(tl.int32)
            bucket = tl.minimum(tl.maximum(bucket, 0), BINS - 1)
            tl.atomic_add(hist_ptr + row[:, None] * BINS + bucket, 1,
                          mask=positive)

        @triton.jit
        def sae_emit(h_ptr, w_ptr, b_ptr, thr_ptr, index_ptr, value_ptr,
                     count_ptr, stride_h, stride_w, rows,
                     T: tl.constexpr, F: tl.constexpr, K: tl.constexpr,
                     H: tl.constexpr, CAP: tl.constexpr):
            row, feature, acc = _tile(
                h_ptr, w_ptr, b_ptr, stride_h, stride_w,
                tl.program_id(0) * T, tl.program_id(1) * F, rows,
                T=T, F=F, K=K, H=H)
            threshold = tl.load(thr_ptr + row, mask=row < rows, other=0.0)
            kept = (acc > 0.0) & (acc >= threshold[:, None]) & (
                row < rows)[:, None]
            counters = count_ptr + tl.broadcast_to(row[:, None],
                                                   acc.shape)
            slot = tl.atomic_add(counters, 1, mask=kept)
            fits = kept & (slot < CAP)
            at = row[:, None] * CAP + slot
            features = tl.broadcast_to(feature[None, :], acc.shape)
            tl.store(index_ptr + at, features, mask=fits)
            tl.store(value_ptr + at, acc, mask=fits)

        cls.histogram = sae_histogram
        cls.emit = sae_emit
        return cls


class SaeEncoder:
    """W_enc and b_enc on the GPU, with the per-token top-k encode.

    The GEMM runs in bf16 and the bias adds in fp32; the agreement
    check against the fp32 reference covers the cast. With
    kernels="quail" the encode runs fused and never writes the
    pre-activation out; "vllm" runs the blocked torch path.
    """

    def __init__(self, torch, F, weight, bias, keep: int, engine=None,
                 kernels: str | None = None):
        self.torch = torch
        self.F = F
        # the checkpoint stores W_enc as a transposed view; the fused
        # kernels index it with an inner stride of one
        self.weight = weight.contiguous()    # (sae_num_features, hidden) bf16
        self.bias = bias                     # (sae_num_features,) fp32
        self.keep = keep
        self.num_features = weight.shape[0]
        self.engine = engine
        self.kernels = (kernels if kernels is not None
                        else getattr(engine, "kernels", "vllm"))
        # rows the last fused call handed to the torch path; stays 0 on
        # the torch-only path
        self.overflow_rows = 0

    @classmethod
    def load(cls, torch, F, spec, engine=None):
        """Load the spec's SAE tensors onto the GPU.

        Raises:
            ValueError: The spec names no SAE, or the tensors do not
                match its width.
        """
        if not spec.sae_hf_name or not spec.sae_file:
            raise ValueError(f"{spec.name!r} names no SAE checkpoint")
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(spec.sae_hf_name, spec.sae_file,
                               revision=spec.sae_revision or None)
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        weight = tensors["W_enc"]
        bias = tensors["b_enc"]
        if tuple(weight.shape) != (spec.sae_num_features, spec.hidden):
            raise ValueError(
                f"{spec.sae_file} W_enc is {tuple(weight.shape)}, expected "
                f"({spec.sae_num_features}, {spec.hidden})")
        return cls(torch,
                   F,
                   weight.to(device="cuda", dtype=torch.bfloat16),
                   bias.to(device="cuda", dtype=torch.float32),
                   spec.sae_keep,
                   engine=engine)

    def _pre(self, rows):
        """The nonnegative pre-activations of a block of rows, in fp32."""
        pre = self.F.linear(rows.to(self.torch.bfloat16), self.weight).float()
        pre += self.bias
        return pre.clamp_(min=0.0)

    def top_features(self, rows):
        """Return (indices int32, strengths fp32), the keep largest per row."""
        torch = self.torch
        if self.kernels == "quail" and rows.shape[0]:
            try:
                return self._fused_top_features(rows)
            except ImportError:
                # without triton, the blocked torch path serves
                pass
        indices, strengths = [], []
        block = max(1, ENCODE_BLOCK_BYTES // (4 * self.num_features))
        for start in range(0, rows.shape[0], block):
            # the fp32 bias promotes the bf16 GEMM output in one pass; a
            # negative value is selected only when a row has fewer than
            # keep positives, and the clamp on the selection zeroes it
            pre = self.F.linear(rows[start:start + block].to(torch.bfloat16),
                                self.weight) + self.bias
            values, index = torch.topk(pre, self.keep, dim=1)
            indices.append(index.to(torch.int32))
            strengths.append(values.clamp_(min=0.0))
        return torch.cat(indices), torch.cat(strengths)

    def _fused_top_features(self, rows):
        """The fused encode: histogram, per-row threshold, emit, exact cut.

        The pre-activation never reaches memory: the first kernel keeps
        a per-row histogram of the positive values, torch picks each
        row's threshold one bin under the top-k boundary, and the second
        kernel emits the values at or above it into a bounded candidate
        buffer that torch cuts to the exact top k. A row whose
        candidates overflow the buffer falls back to the torch path.
        """
        import triton

        torch = self.torch
        kernels = _SaeKernels.compiled()
        count = rows.shape[0]
        hidden = rows.shape[1]
        h = rows.to(torch.bfloat16).contiguous()
        histogram = torch.zeros((count, HISTOGRAM_BINS), device="cuda",
                                dtype=torch.int32)
        tile_rows, tile_features, tile_k = 64, 128, 64
        grid = (triton.cdiv(count, tile_rows),
                self.num_features // tile_features)
        kernels.histogram[grid](
            h, self.weight, self.bias, histogram,
            h.stride(0), self.weight.stride(0), count,
            T=tile_rows, F=tile_features, K=tile_k, H=hidden,
            BINS=HISTOGRAM_BINS)
        # per row, the count of values in each bin or above, and the
        # largest bin that still reaches keep values; one bin lower
        # keeps boundary values in despite float rounding
        above = histogram.flip(1).cumsum(1).flip(1)
        reached = (above >= self.keep).sum(1) - 2
        thresholds = torch.where(
            reached >= 0, torch.exp2(reached.float() / 4.0 - 32.0),
            torch.zeros((), device="cuda"))
        counters = torch.zeros(count, dtype=torch.int32, device="cuda")
        out_index = torch.zeros((count, CANDIDATE_CAP), dtype=torch.int32,
                                device="cuda")
        out_value = torch.zeros((count, CANDIDATE_CAP), dtype=torch.float32,
                                device="cuda")
        kernels.emit[grid](
            h, self.weight, self.bias, thresholds, out_index, out_value,
            counters, h.stride(0), self.weight.stride(0), count,
            T=tile_rows, F=tile_features, K=tile_k, H=hidden,
            CAP=CANDIDATE_CAP)
        strengths, position = torch.topk(out_value, self.keep, dim=1)
        indices = torch.gather(out_index, 1, position)
        overflowed = (counters > CANDIDATE_CAP).nonzero(as_tuple=True)[0]
        self.overflow_rows = int(overflowed.numel())
        if overflowed.numel():
            pre = self._pre(rows[overflowed])
            values, index = torch.topk(pre, self.keep, dim=1)
            strengths[overflowed] = values
            indices[overflowed] = index.to(torch.int32)
        return indices, strengths

    def packed_top_features(self, rows):
        """One fp32 tensor per block of rows: the indices, then the strengths.

        Feature indices are below 2**24, so fp32 carries them exactly.
        """
        indices, strengths = self.top_features(rows)
        return self.torch.cat([indices.float(), strengths], dim=1)


class SaeFeaturesReadout(_HostReadout):
    """Encode the answer rows' residual stream, for the pooling 'last'.

    result() returns a (rows, 2 * keep) fp32 array: each row's feature
    indices, then their strengths. The dtype carries the row width, so
    the scheduler's answer array holds one vector per answer.
    """

    trailing_rows = 1

    def __init__(self, torch, encoder: SaeEncoder):
        super().__init__(torch)
        self.encoder = encoder
        self.dtype = np.dtype((np.float32, 2 * encoder.keep))

    def submit(self, stream):
        return self._copy(self.encoder.packed_top_features(stream))


def feature_lists_of_rows(packed, keep: int) -> tuple:
    """Return (lengths, indices, strengths) of packed readout rows.

    Each row is a readout's packed vector: keep feature indices, then
    their strengths, zero past the kept ones. Every row unpacks in one
    pass, with no per-feature or per-row Python work.
    """
    packed = np.asarray(packed, dtype=np.float32).reshape(-1, 2 * keep)
    indices = packed[:, :keep]
    strengths = packed[:, keep:]
    row_ids, columns = np.nonzero(strengths > 0.0)
    lengths = np.bincount(row_ids, minlength=packed.shape[0])
    return (lengths, indices[row_ids, columns].astype(np.int32),
            strengths[row_ids, columns])


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
        self._next = rows_per_slab

    def take(self) -> tuple:
        """Return (slab, row) of a fresh pinned row."""
        if self._next == self.rows_per_slab:
            self.slabs.append(self.torch.empty(
                (self.rows_per_slab, self.width), dtype=self.torch.float32,
                pin_memory=True))
            self._next = 0
        self._next += 1
        return len(self.slabs) - 1, self._next - 1


class _Staging:
    """Pinned host staging for a chunk's index arrays, two buffers deep.

    A pageable host-to-device copy synchronizes; a pinned one does not.
    A buffer is reused only once its earlier copy's event has completed.
    """

    def __init__(self, torch):
        self.torch = torch
        self._buffers = [None, None]
        self._events = [None, None]
        self._turn = 0

    def upload(self, array: np.ndarray):
        """Copy a one-dimensional int64 array to the GPU asynchronously."""
        torch = self.torch
        index = self._turn
        self._turn ^= 1
        if self._events[index] is not None:
            self._events[index].synchronize()
        buffer = self._buffers[index]
        if buffer is None or buffer.numel() < array.size:
            buffer = torch.empty(max(array.size, 1 << 16), dtype=torch.int64,
                                 pin_memory=True)
            self._buffers[index] = buffer
        host = buffer[:array.size]
        host.numpy()[:] = array
        device = host.to("cuda", non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._events[index] = event
        return device


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
        self._staging = _Staging(torch)
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
        shape, so the observer never synchronizes with the forward pass.
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
        block = self._staging.upload(np.concatenate(parts))
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

    def settle(self, key) -> tuple:
        """Return the handle of one document's combined maxima for one call.

        The question's slot is consumed; the document's stays for the
        pipeline's later feature calls and is freed after the last one.
        The handle names the pinned slab and row the maxima copy into.
        """
        torch = self.torch
        matrix = self._slots.matrix
        tail = self.tail_slot.pop(key, None)
        document = self.document_slot.get(key)
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
        if tail is not None:
            self._slots.release(tail)
        self.settled[key] = self.settled.get(key, 0) + 1
        if self.settled[key] >= self.calls and document is not None:
            self._slots.release(self.document_slot.pop(key))
        return event, slab, row

    def harvest(self, handles) -> tuple:
        """Return (order, lengths, indices, strengths) of settled handles.

        Each slab's rows unpack in one pass over a view of the slab, so
        the results come out slab by slab: ``order`` lists the handle
        positions in that sequence, and ``lengths`` gives each one's
        feature count in the same sequence. Call it once the pipeline
        is done; a handle whose copy is still in flight is waited on.
        """
        count = len(handles)
        slabs = np.fromiter((handle[1] for handle in handles), np.int64, count)
        rows = np.fromiter((handle[2] for handle in handles), np.int64, count)
        for event, _slab, _row in handles:
            event.synchronize()
        order = np.lexsort((rows, slabs))
        slabs, rows = slabs[order], rows[order]
        lengths = np.zeros(count, dtype=np.int64)
        indices, strengths = [], []
        starts = np.flatnonzero(np.diff(slabs, prepend=-1))
        for start, stop in zip(starts, (*starts[1:], count)):
            block = self._pinned.slabs[slabs[start]].numpy()
            group = rows[start:stop]
            if group[-1] - group[0] == stop - start - 1:
                view = block[group[0]:group[-1] + 1]
            else:
                view = block[group]
            row_ids, columns = np.nonzero(view > 0.0)
            lengths[start:stop] = np.bincount(row_ids, minlength=stop - start)
            indices.append(columns.astype(np.int32))
            strengths.append(view[row_ids, columns])
        return (order, lengths,
                np.concatenate(indices) if indices else np.empty(0, np.int32),
                np.concatenate(strengths) if strengths
                else np.empty(0, np.float32))


def sae_encoder(state) -> SaeEncoder:
    """Return the loaded model's SAE encoder, loading it on first use.

    The engine encodes on the blocked torch path: measured on 4,096-row
    chunks, the fused path takes 8.7 ms per chunk against torch's
    4.4 ms (the emit pass recomputes a compute-bound GEMM), and its
    overflow check reads the GPU, which stalls the forward pass. The
    fused kernels stay available through kernels="quail".
    """
    loaded = state.loaded_model
    if getattr(loaded, "sae", None) is None:
        import torch.nn.functional as F

        loaded.sae = SaeEncoder.load(state.torch, F, loaded.model_spec)
    return loaded.sae
