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

# rows per encode block, bounding the pre-activation the torch path
# materializes (BLOCK_ROWS x sae_num_features floats)
BLOCK_ROWS = 512


# the fused path's histogram bins and candidate buffer capacity: bins
# are quarter octaves of the positive pre-activations, and a row's
# candidates are the values in and above its threshold bin, at most
# keep plus the threshold bin's own count
HISTOGRAM_BINS = 256
CANDIDATE_CAP = 512


def _sae_kernels():
    """Compile and cache the fused encode kernels.

    Both kernels tile the encode GEMM (rows x hidden) x (hidden x
    features) and never write the pre-activation out. The first builds
    a per-row histogram of the positive values; the second emits the
    values at or above each row's threshold into a bounded candidate
    buffer, which torch cuts to the exact top k. The layout follows
    the conventions of the attention kernels
    (quail.backends.quail.executor.attention).
    """
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
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

    _KERNELS = {"hist": sae_histogram, "emit": sae_emit}
    return _KERNELS


_KERNELS = None


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
        for start in range(0, rows.shape[0], BLOCK_ROWS):
            pre = self._pre(rows[start:start + BLOCK_ROWS])
            values, index = torch.topk(pre, self.keep, dim=1)
            indices.append(index.to(torch.int32))
            strengths.append(values)
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
        kernels = _sae_kernels()
        count = rows.shape[0]
        hidden = rows.shape[1]
        h = rows.to(torch.bfloat16).contiguous()
        histogram = torch.zeros((count, HISTOGRAM_BINS), device="cuda",
                                dtype=torch.int32)
        tile_rows, tile_features, tile_k = 64, 128, 64
        grid = (triton.cdiv(count, tile_rows),
                self.num_features // tile_features)
        kernels["hist"][grid](
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
        kernels["emit"][grid](
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


def unpack_features(row, keep: int) -> list:
    """Return one packed readout row as (feature, strength) mappings.

    Zero strengths are left out: the SAE keeps at most ``keep`` features,
    and a padded row carries zeros past the nonzero ones.
    """
    indices = row[:keep].astype(np.int64)
    strengths = row[keep:2 * keep]
    kept = strengths > 0.0
    return [{"feature": int(feature), "strength": float(strength)}
            for feature, strength in zip(indices[kept], strengths[kept])]


class _PinnedRows:
    """Pinned host rows handed out from slab allocations.

    A settle runs on the thread that feeds the GPU, and allocating
    pinned memory there costs milliseconds per call, so one slab serves
    many settles. Handed-out rows are views that keep their slab alive.
    """

    def __init__(self, torch, width: int, rows_per_slab: int = 256):
        self.torch = torch
        self.width = width
        self.rows_per_slab = rows_per_slab
        self._slab = None
        self._next = 0

    def take(self):
        """Return one pinned (width,) fp32 row."""
        if self._slab is None or self._next == self.rows_per_slab:
            self._slab = self.torch.empty(
                (self.rows_per_slab, self.width), dtype=self.torch.float32,
                pin_memory=True)
            self._next = 0
        row = self._slab[self._next]
        self._next += 1
        return row


class SaeMaxAccumulator:
    """Per-document maxima of per-token top-k strengths, for the pooling 'max'.

    observe() runs inside forward_chunk over every chunk row. A row
    whose position is inside its document's prefix folds into the
    document's buffer, which the pipeline's feature calls share; the
    other rows belong to the running call's question and are consumed
    when the document's stage settles. The buffers are freed once a
    document has settled every registered feature call.
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
        self.document_max = {}
        self.tail_max = {}
        self.calls = 0        # feature calls registered on this pipeline
        self.settled = {}     # key -> calls settled so far
        self._buffers = []
        self._pinned = _PinnedRows(torch, encoder.num_features)

    def register_call(self) -> None:
        """Count one feature call of the pipeline, before execution."""
        self.calls += 1

    def _buffer(self):
        if self._buffers:
            return self._buffers.pop().zero_()
        return self.torch.zeros(self.encoder.num_features, device="cuda",
                                dtype=self.torch.float32)

    def observe(self, stream, chunk) -> None:
        """Fold one chunk's rows into the document and question maxima.

        The whole chunk encodes in one call, and each span's strengths
        fold under a mask instead of a boolean-indexed subset: a masked
        strength contributes 0, which amax over nonnegative strengths
        ignores. Nothing here reads a data-dependent shape, so the
        observer never synchronizes with the forward pass.
        """
        indices, strengths = self.encoder.top_features(stream)
        indices = indices.to(self.torch.int64)
        positions = chunk.positions
        for key, row0, _suffix_row, row1 in chunk.row_spans:
            if row1 == row0:
                continue
            inside = (positions[row0:row1]
                      < self.document_tokens(key)).unsqueeze(1)
            span_indices = indices[row0:row1].reshape(-1)
            span_strengths = strengths[row0:row1]
            for buffers, mask in ((self.document_max, inside),
                                  (self.tail_max, ~inside)):
                if key not in buffers:
                    buffers[key] = self._buffer()
                buffers[key].scatter_reduce_(
                    0, span_indices, (span_strengths * mask).reshape(-1),
                    reduce="amax")

    def settle(self, key):
        """Return the handle of one document's combined maxima for one call.

        The question's buffer is consumed; the document's stays for the
        pipeline's later feature calls and is freed after the last one.
        """
        torch = self.torch
        tail = self.tail_max.pop(key, None)
        document = self.document_max.get(key)
        if tail is not None and document is not None:
            combined = torch.maximum(document, tail)
        elif tail is not None:
            combined = tail.clone()
        elif document is not None:
            combined = document.clone()
        else:
            combined = self._buffer()
        host = self._pinned.take()
        host.copy_(combined, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        if tail is not None:
            self._buffers.append(tail)
        self._buffers.append(combined)
        self.settled[key] = self.settled.get(key, 0) + 1
        if self.settled[key] >= self.calls and document is not None:
            self._buffers.append(self.document_max.pop(key))
        return event, host

    @staticmethod
    def features_of(handle) -> list:
        """Return a settled handle's nonzero (feature, strength) mappings."""
        event, host = handle
        event.synchronize()
        values = host.numpy()
        kept = np.flatnonzero(values > 0.0)
        return [{"feature": int(feature), "strength": float(values[feature])}
                for feature in kept]


def sae_encoder(state) -> SaeEncoder:
    """Return the loaded model's SAE encoder, loading it on first use."""
    loaded = state.loaded_model
    if getattr(loaded, "sae", None) is None:
        import torch.nn.functional as F

        loaded.sae = SaeEncoder.load(state.torch, F, loaded.model_spec,
                                     engine=loaded.pipeline.engine)
    return loaded.sae
