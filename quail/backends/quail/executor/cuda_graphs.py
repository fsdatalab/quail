"""Capture the work between token mixers, leaving the mixers and their metadata eager.

A token mixer (attention, or a Gated DeltaNet layer's convolution and
recurrence) reads per-chunk metadata of variable size, so it runs
eagerly between captured pieces, as vLLM's piecewise graphs split at
the same ops.
"""

import time
from dataclasses import dataclass

from quail.common.budgets import graph_memory_bytes, graph_token_buckets
from quail.progress import logger


def chunk_writes(chunk):
    """Return the chunk's fresh and partial-page write mappings."""
    if chunk.attention_mode == "tree":
        return dict(src=chunk.meta["kv_src"], dst=chunk.meta["kv_dst"],
                    tail_src=None, tail_dst=None)
    return chunk.meta["unified"]


class KVWriteBuffers:
    """Fixed scatter mappings with masked destinations for unused entries.

    Given storage, a larger bucket's buffers, the mappings are prefixes
    of that bucket's tensors.
    """

    def __init__(self, torch, num_tokens, page_tokens, *, device, storage=None):
        self.meta = {}
        self.has_tails = False
        self._state = (storage._state if storage is not None
                       else {"src": 0, "dst": 0, "tail_src": 0, "tail_dst": 0})
        for name in ("src", "dst", "tail_src", "tail_dst"):
            count = num_tokens * (page_tokens if name.startswith("tail_") else 1)
            if storage is not None:
                self.meta[name] = storage.meta[name][:count]
                continue
            self.meta[name] = torch.full(
                (count,), -1 if name.endswith("dst") else 0,
                dtype=torch.int64, device=device)

    def supports(self, writes):
        """Whether all mappings fit without dropping any KV writes."""
        return writes is None or all(
            writes[name] is None or writes[name].numel() <= target.numel()
            for name, target in self.meta.items())

    def copy(self, writes):
        """Stage mappings and mask only previously written unused entries."""
        self.has_tails = (writes is not None and writes["tail_src"] is not None
                          and writes["tail_src"].numel() > 0)
        for name, target in self.meta.items():
            if (name.startswith("tail_") and not self.has_tails
                    and not self._state[name]):
                continue
            source = writes[name] if writes is not None else None
            count = 0 if source is None else source.numel()
            previous = min(self._state[name], target.numel())
            if count:
                target[:count].copy_(source)
            if previous > count:
                target[count:previous].fill_(-1 if name.endswith("dst") else 0)
            self._state[name] = count


class PiecewiseBuffers:
    """Persistent boundary tensors for one padded token count.

    The model pipeline names the tensors its pieces and token mixers
    pass between them (ModelPipeline.boundary); these are shared by
    every model: input ids and positions, the running residual, the
    final rows, and the fixed KV write mappings a piece writes through.

    Only real rows reach a token mixer. Padding therefore has no KV
    writes, sequence boundaries, state, or reads. GEMM tiling can change
    rounding with the padded row count; replay parity is checked
    against eager pieces with the same bucket shape.

    Given storage, a larger bucket's buffers, every row-major tensor is
    a prefix of that bucket's, so all buckets share one allocation; one
    stream replays one bucket at a time. Tensors that cannot be sliced
    by row, such as column-major activation scales, are the pipeline's
    bucket_boundary, which each bucket allocates for itself.
    """

    def __init__(self, pipeline, num_tokens, *, device="cuda", storage=None):
        engine = pipeline.engine
        torch = engine.torch
        self.num_tokens = num_tokens
        self.copy_tail = False
        if storage is not None and storage.num_tokens < num_tokens:
            raise ValueError("storage must hold at least num_tokens rows")
        if storage is None:
            width = pipeline.final_norm.weight.numel()
            ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
            residual = torch.zeros((num_tokens, width), dtype=torch.bfloat16,
                                   device=device)
            rows = dict(ids=ids, positions=torch.zeros_like(ids),
                        residual=residual, output=torch.empty_like(residual),
                        **pipeline.boundary(num_tokens, device))
        else:
            rows = {name: tensor[:num_tokens]
                    for name, tensor in storage.rows.items()}
        self.rows = rows
        for name, tensor in (rows | pipeline.bucket_boundary(
                num_tokens, device)).items():
            setattr(self, name, tensor)
        page_tokens = engine.arena.page_tokens
        self.writes = KVWriteBuffers(
            torch, num_tokens, page_tokens, device=device,
            storage=None if storage is None else storage.writes)
        self.sliding_writes = (KVWriteBuffers(
            torch, num_tokens, page_tokens, device=device,
            storage=None if storage is None else storage.sliding_writes)
            if engine.arena.has_sliding else None)
        self.kv_meta = dict(layer=0, unified=dict(self.writes.meta))
        if self.sliding_writes is not None:
            self.kv_meta["unified"]["sliding"] = self.sliding_writes.meta

    @property
    def has_tails(self):
        """Whether either KV pool has retained partial pages to copy."""
        return (self.writes.has_tails
                or (self.sliding_writes is not None and self.sliding_writes.has_tails))

    def supports(self, chunk):
        """Whether the full and sliding KV mappings fit this token bucket."""
        writes = chunk_writes(chunk)
        return (self.writes.supports(writes)
                and (self.sliding_writes is None or self.sliding_writes.supports(
                    writes.get("sliding") or writes if writes is not None else None)))

    def copy(self, chunk):
        """Stage real input rows and clear padding from the previous pass."""
        self.ids.zero_()
        self.positions.zero_()
        self.ids[:chunk.tokens].copy_(chunk.input_ids)
        rotary = getattr(chunk, "rotary", None)
        self.positions[:chunk.tokens].copy_(
            chunk.positions if rotary is None else rotary)
        writes = chunk_writes(chunk)
        self.writes.copy(writes)
        if self.sliding_writes is not None:
            self.sliding_writes.copy(
                writes.get("sliding") or writes if writes is not None else None)

    def write_kv(self, engine, k, v, layer):
        """Write a layer's fresh KV rows through the staged mappings."""
        self.kv_meta["layer"] = layer
        engine.write_kv(
            k.view(self.num_tokens, engine.num_kv_heads, engine.head_dim),
            v.view(self.num_tokens, engine.num_kv_heads, engine.head_dim),
            self.kv_meta, copy_tail=self.copy_tail)

    def piece(self, pipeline, index, *, copy_tail=None):
        """Run the pipeline's work from token mixer index - 1 up to mixer index."""
        self.copy_tail = self.has_tails if copy_tail is None else copy_tail
        pipeline.piece(index, self)

    def mixer(self, pipeline, index, chunk):
        """Run layer index's token mixer eagerly over the chunk's real rows."""
        chunk.meta["layer"] = index
        pipeline.mixer(index, self, chunk)


@dataclass
class CapturedPieces:
    """A bucket's boundary buffers and ordered graphs between token mixers."""

    buffers: PiecewiseBuffers
    graphs: list | None
    tail_graphs: list | None = None

    def forward(self, pipeline, chunk):
        """Replay pieces separated by eager token mixers; copy selected outputs."""
        self.buffers.copy(chunk)
        graphs = (self.tail_graphs if self.buffers.has_tails
                  and self.tail_graphs is not None else self.graphs)
        chunk.meta["layer"] = 0
        previous = chunk.meta.get("kv_written")
        chunk.meta["kv_written"] = True
        try:
            with pipeline.forward_context(self.buffers.num_tokens):
                graphs[0].replay()
                for index, graph in enumerate(graphs[1:]):
                    self.buffers.mixer(pipeline, index, chunk)
                    graph.replay()
            return self.buffers.output.index_select(0, chunk.final_indices)
        finally:
            if previous is None:
                chunk.meta.pop("kv_written")
            else:
                chunk.meta["kv_written"] = previous


_PRECAPTURE_BUCKETS = (4096, 2048, 1024, 512, 256, 192, 128, 64, 16)
_TILE_ROUND_MAX_TOKENS = 256
_TILE_ROUND_STEP = 64


class PiecewiseGraphs:
    """Capture and replay the pieces of a model pipeline on the current stream.

    Captured writes reference the arena, so resizing it invalidates the
    graphs. The two tail-copy variants share each bucket's boundary
    buffers. All temporary tensors die inside their piece, allowing the
    shared pool to reuse their storage across layers and token buckets.
    """

    def __init__(self, pipeline, *, token_buckets, memory_bytes):
        if (not isinstance(token_buckets, tuple) or not token_buckets
                or any(type(n) is not int or n < 1 for n in token_buckets)
                or any(a >= b for a, b in zip(token_buckets, token_buckets[1:]))):
            raise ValueError("token_buckets must be increasing positive integers")
        self.pipeline = pipeline
        self.token_buckets = token_buckets
        self.memory_bytes = memory_bytes
        self.enabled = True
        self.graphs = []
        self.storage = None
        self.replays = 0
        self.passes = 0
        self.capture_s = 0.0
        self.captured_bytes = 0
        self.buffer_bytes = 0
        self.growth_bytes = 0
        self.token_bytes = 0.0
        self.pool = None
        self.stream = None

    @classmethod
    def for_specs(cls, pipeline, model, device):
        """Build graphs only for the model families budgeted for capture."""
        buckets = graph_token_buckets(model, device)
        if not buckets:
            return None
        return cls(pipeline, token_buckets=buckets,
                   memory_bytes=graph_memory_bytes(model, device))

    def invalidate(self):
        """Release captures before replacing the KV allocation they write."""
        self.graphs.clear()
        self.storage = None
        self.pool = None
        self.stream = None

    def prepare(self):
        """Build every bucket's buffers and capture the largest bucket.

        Each smaller bucket is captured on the first pass that needs it,
        so a new KV allocation costs one bucket's capture, not every
        bucket's. A bucket captured later grows the shared pool by its
        own blocks, so later captures stop before the pool outgrows its
        reserve, and the remaining buckets run eagerly.

        Returns:
            Whether this call built new graphs for the current KV allocation.

        Raises:
            RuntimeError: The largest bucket alone exceeds the reserve.
        """
        engine = self.pipeline.engine
        if not self.enabled or (self.graphs and self.storage is engine.arena.k):
            return False
        self.invalidate()
        torch = engine.torch
        self.capture_s = 0.0
        before = torch.cuda.memory_allocated()
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream()
        storage = None
        for num_tokens in reversed(self.token_buckets):
            buffers = PiecewiseBuffers(self.pipeline, num_tokens, storage=storage)
            storage = storage or buffers
            self.graphs.append(CapturedPieces(buffers, None))
        self.graphs.reverse()
        self.buffer_bytes = torch.cuda.memory_allocated() - before
        self.storage = engine.arena.k
        self.capture(self.graphs[-1])
        if self.captured_bytes > self.memory_bytes:
            self.invalidate()
            raise RuntimeError(
                f"graphs allocated {self.captured_bytes} bytes, "
                f"exceeding their {self.memory_bytes}-byte reserve")
        self.token_bytes = ((self.captured_bytes - self.buffer_bytes)
                            / self.token_buckets[-1])
        self.growth_bytes = (self.memory_bytes - self.captured_bytes) // 8
        by_tokens = {item.buffers.num_tokens: item for item in self.graphs}
        for size in _PRECAPTURE_BUCKETS:
            target = by_tokens.get(size)
            if target is not None and target.graphs is None:
                if not self.capture_within_reserve(target):
                    break
        return True

    def capture_within_reserve(self, captured, *, copy_tail=False):
        """Capture a smaller bucket or tail variant if the pool has room.

        A bucket grows the pool by its tokens' share of the largest
        bucket's blocks, plus a fixed part: the largest such part seen,
        and at first an eighth of the room the largest bucket left.

        Returns:
            Whether the requested graphs are now captured.
        """
        scaled = captured.buffers.num_tokens * self.token_bytes
        if self.captured_bytes + scaled + self.growth_bytes > self.memory_bytes:
            return False
        before = self.captured_bytes
        if copy_tail:
            self.capture(captured, copy_tail=True)
        else:
            self.capture(captured)
        self.growth_bytes = max(self.growth_bytes,
                                self.captured_bytes - before - scaled)
        if self.captured_bytes > self.memory_bytes:
            logger.warning("CUDA graphs: %.0f MiB exceeds the %.0f MiB reserve; "
                           "smaller buckets run eagerly",
                           self.captured_bytes / 2**20, self.memory_bytes / 2**20)
        return True

    def capture(self, captured, *, copy_tail=False):
        """Capture one bucket's pieces into the shared pool.

        Each piece runs once eagerly before its capture, so the KV write
        mappings are masked first; they may still hold a real pass's
        pages. The partial-page tail variant is captured only when a
        pass needs it.
        """
        engine = self.pipeline.engine
        torch = engine.torch
        buffers = captured.buffers
        started = time.perf_counter()
        # The allocator reuses a pool's freed blocks only on the stream
        # that freed them, so every capture shares one stream.
        stream = self.stream
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        try:
            buffers.writes.copy(None)
            if buffers.sliding_writes is not None:
                buffers.sliding_writes.copy(None)
            num_pieces = len(self.pipeline.layers) + 1
            with (torch.cuda.stream(stream), torch.inference_mode(),
                  self.pipeline.forward_context(buffers.num_tokens)):
                for index in range(num_pieces):
                    buffers.piece(self.pipeline, index, copy_tail=copy_tail)
                graphs = []
                for index in range(num_pieces):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                        buffers.piece(self.pipeline, index, copy_tail=copy_tail)
                    graphs.append(graph)
            current.wait_stream(stream)
            stream.synchronize()
            # The pool's segments stay reserved until its graphs die. A
            # query's own tensors live outside the pool, so a capture
            # during a query counts only the pool and the buffers.
            self.captured_bytes = self.buffer_bytes + sum(
                segment["total_size"]
                for segment in torch.cuda.memory_snapshot()
                if tuple(segment["segment_pool_id"]) == self.pool)
        except Exception:
            self.invalidate()
            raise
        if copy_tail:
            captured.tail_graphs = graphs
        else:
            captured.graphs = graphs
        seconds = time.perf_counter() - started
        self.capture_s += seconds
        logger.info("CUDA graphs: captured %d tokens%s in %.1f s, using "
                    "%.0f of %.0f MiB", buffers.num_tokens,
                    " (tails)" if copy_tail else "", seconds,
                    self.captured_bytes / 2**20, self.memory_bytes / 2**20)

    def supports(self, chunk):
        """Whether a chunk fits capture without a model-specific forward hook."""
        if (not self.enabled or not self.graphs
                or self.storage is not self.pipeline.engine.arena.k
                or not 0 < chunk.tokens <= self.token_buckets[-1]
                or chunk.meta.get("canvas") is not None
                or self.pipeline.forward_hook is not None):
            return False
        return next(c.buffers for c in self.graphs
                    if c.buffers.num_tokens >= chunk.tokens).supports(chunk)

    def _select_bucket(self, chunk):
        """Return the bucket to replay for chunk, reusing a tile-aligned capture."""
        first = next(c for c in self.graphs
                     if c.buffers.num_tokens >= chunk.tokens)
        if first.graphs is not None:
            return first
        if len(self.graphs) > 16 and chunk.tokens <= _TILE_ROUND_MAX_TOKENS:
            ceiling = max(16, ((chunk.tokens + _TILE_ROUND_STEP - 1)
                               // _TILE_ROUND_STEP) * _TILE_ROUND_STEP)
            ready = next((c for c in self.graphs
                          if chunk.tokens <= c.buffers.num_tokens <= ceiling
                          and c.graphs is not None
                          and c.buffers.supports(chunk)), None)
            if ready is not None:
                return ready
        return first

    def forward(self, chunk):
        """Replay dense pieces or run a larger chunk eagerly."""
        self.passes += 1
        if self.supports(chunk):
            captured = self._select_bucket(chunk)
            if (captured.graphs is not None
                    or self.capture_within_reserve(captured)):
                writes = chunk_writes(chunk)
                needs_tails = (
                    writes is not None and (
                        (writes["tail_src"] is not None
                         and writes["tail_src"].numel() > 0)
                        or (writes.get("sliding") is not None
                            and writes["sliding"]["tail_src"] is not None
                            and writes["sliding"]["tail_src"].numel() > 0)
                    )
                )
                if (not needs_tails or captured.tail_graphs is not None
                        or self.capture_within_reserve(captured, copy_tail=True)):
                    self.replays += 1
                    return captured.forward(self.pipeline, chunk)
        return self.pipeline.forward_chunk(chunk)
