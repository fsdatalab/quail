"""Capture dense layer work while leaving attention and its metadata eager."""

import time
from dataclasses import dataclass
from functools import partial

from quail.cost.budgets import graph_memory_bytes, graph_token_buckets
from quail.progress import logger


def chunk_writes(chunk):
    """Return the chunk's fresh and partial-page write mappings."""
    if chunk.attention_mode == "tree":
        return dict(src=chunk.meta["kv_src"], dst=chunk.meta["kv_dst"],
                    tail_src=None, tail_dst=None)
    return chunk.meta["unified"]


class KVWriteBuffers:
    """Fixed scatter mappings with masked destinations for unused entries."""

    def __init__(self, torch, num_tokens, page_tokens, *, device):
        self.meta = {}
        self.has_tails = False
        for name in ("src", "dst", "tail_src", "tail_dst"):
            count = num_tokens * (page_tokens if name.startswith("tail_") else 1)
            self.meta[name] = torch.full(
                (count,), -1 if name.endswith("dst") else 0,
                dtype=torch.int64, device=device)

    def supports(self, writes):
        """Whether all mappings fit without dropping any KV writes."""
        return writes is None or all(
            writes[name] is None or writes[name].numel() <= target.numel()
            for name, target in self.meta.items())

    def copy(self, writes):
        """Stage mappings and mask unused fresh and partial-page destinations."""
        self.has_tails = (writes is not None and writes["tail_src"] is not None
                          and writes["tail_src"].numel() > 0)
        for name, target in self.meta.items():
            target.fill_(-1 if name.endswith("dst") else 0)
            source = writes[name] if writes is not None else None
            if source is not None:
                target[:source.numel()].copy_(source)


class PiecewiseBuffers:
    """Persistent boundary tensors for one padded token count.

    Only real rows reach attention. Padding therefore has no KV writes,
    sequence boundaries, or reads, regardless of the attention path.
    GEMM tiling can change rounding with the padded row count; replay
    parity is checked against eager pieces with the same bucket shape.
    """

    def __init__(self, pipeline, num_tokens, *, device="cuda"):
        engine = pipeline.engine
        torch = engine.torch
        self.num_tokens = num_tokens
        self.ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        self.positions = torch.zeros_like(self.ids)
        self.residual = torch.zeros(
            (num_tokens, pipeline.final_norm.weight.numel()),
            dtype=torch.bfloat16, device=device)
        self.output = torch.empty_like(self.residual)
        self.q = torch.zeros((num_tokens, engine.num_q_heads * engine.head_dim),
                             dtype=torch.bfloat16, device=device)
        self.k = torch.zeros((num_tokens, engine.num_kv_heads * engine.head_dim),
                             dtype=torch.bfloat16, device=device)
        self.v = torch.zeros_like(self.k)
        self.o_in, self.o_scale = engine.quant(torch.zeros_like(self.q))
        self.writes = KVWriteBuffers(torch, num_tokens, engine.arena.page_tokens,
                                    device=device)
        self.sliding_writes = (KVWriteBuffers(
            torch, num_tokens, engine.arena.page_tokens, device=device)
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
        self.positions[:chunk.tokens].copy_(chunk.positions)
        writes = chunk_writes(chunk)
        self.writes.copy(writes)
        if self.sliding_writes is not None:
            self.sliding_writes.copy(
                writes.get("sliding") or writes if writes is not None else None)

    def attention(self, pipeline, chunk):
        """Read real rows, writing attention directly to the dense graph's inputs."""
        n = chunk.tokens
        pipeline.engine.attention(
            self.q[:n], self.k[:n], self.v[:n], chunk,
            output=(self.o_in[:n], None if self.o_scale is None else self.o_scale[:n]))

    def piece(self, pipeline, index, *, copy_tail=None):
        """Run embedding or the previous MLP, then the next QKV or final norm."""
        if index == 0:
            hidden = pipeline.embed(self.ids)
            residual = None
        else:
            residual = self.residual
            hidden = pipeline.after_attention(
                self.o_in, self.o_scale, residual, pipeline.layers[index - 1])
        if index == len(pipeline.layers):
            normed, _ = pipeline.engine.fused_add_rms_norm(
                hidden, residual, pipeline.final_norm)
            self.output.copy_(normed)
        else:
            q, k, v, residual = pipeline.before_attention(
                hidden, residual, self.positions, pipeline.layers[index])
            self.q.copy_(q)
            self.k.copy_(k)
            self.v.copy_(v)
            if index == 0:
                self.residual.copy_(residual)
            engine = pipeline.engine
            self.kv_meta["layer"] = index
            engine.write_kv(
                self.k.view(self.num_tokens, engine.num_kv_heads, engine.head_dim),
                self.v.view(self.num_tokens, engine.num_kv_heads, engine.head_dim),
                self.kv_meta,
                copy_tail=self.has_tails if copy_tail is None else copy_tail)


@dataclass
class CapturedPieces:
    """A bucket's boundary buffers and ordered non-attention graphs."""

    buffers: PiecewiseBuffers
    graphs: list
    tail_graphs: list | None = None

    def forward(self, pipeline, chunk):
        """Replay pieces separated by eager attention and copy selected outputs."""
        self.buffers.copy(chunk)
        graphs = (self.tail_graphs if self.buffers.has_tails
                  and self.tail_graphs is not None else self.graphs)
        chunk.meta["layer"] = 0
        previous = chunk.meta.get("kv_written")
        chunk.meta["kv_written"] = True
        try:
            graphs[0].replay()
            for graph in graphs[1:]:
                self.buffers.attention(pipeline, chunk)
                graph.replay()
            return self.buffers.output.index_select(0, chunk.final_indices)
        finally:
            if previous is None:
                chunk.meta.pop("kv_written")
            else:
                chunk.meta["kv_written"] = previous


class PiecewiseGraphs:
    """Capture and replay Qwen3 dense work on the worker's current stream.

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

    def prepare(self):
        """Capture dense pieces after kernel warmup, once per KV allocation."""
        engine = self.pipeline.engine
        if not self.enabled or (self.graphs and self.storage is engine.arena.k):
            return False
        self.invalidate()
        torch = engine.torch
        started = time.perf_counter()
        before = torch.cuda.memory_allocated()
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream), torch.inference_mode():
                # Larger captures establish the shared pool's high watermark.
                for num_tokens in reversed(self.token_buckets):
                    buffers = PiecewiseBuffers(self.pipeline, num_tokens)
                    variants = []
                    for copy_tail in (False, True):
                        graphs = []
                        for index in range(len(self.pipeline.layers) + 1):
                            piece = partial(buffers.piece, self.pipeline, index,
                                            copy_tail=copy_tail)
                            piece()
                            stream.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, pool=pool, stream=stream):
                                piece()
                            graphs.append(graph)
                        variants.append(graphs)
                    self.graphs.append(CapturedPieces(buffers, *variants))
            current.wait_stream(stream)
            stream.synchronize()
            self.graphs.reverse()
            # Inactive blocks in a graph pool remain reserved until its
            # graphs die; memory_allocated alone omits those blocks.
            private_unused = sum(
                segment["total_size"] - segment["allocated_size"]
                for segment in torch.cuda.memory_snapshot()
                if tuple(segment["segment_pool_id"]) == pool)
            self.captured_bytes = (
                torch.cuda.memory_allocated() - before + private_unused)
            if self.captured_bytes > self.memory_bytes:
                raise RuntimeError(
                    f"graphs allocated {self.captured_bytes} bytes, "
                    f"exceeding their {self.memory_bytes}-byte reserve")
        except Exception:
            self.invalidate()
            raise
        self.storage = engine.arena.k
        self.capture_s = time.perf_counter() - started
        logger.info("captured %d piecewise buckets in %.2f s; %.1f MiB "
                    "storage / %.1f MiB budget",
                    len(self.graphs), self.capture_s, self.captured_bytes / 2**20,
                    self.memory_bytes / 2**20)
        return True

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

    def forward(self, chunk):
        """Replay dense pieces or run a larger chunk eagerly."""
        self.passes += 1
        if self.supports(chunk):
            for captured in self.graphs:
                if captured.buffers.num_tokens >= chunk.tokens:
                    self.replays += 1
                    return captured.forward(self.pipeline, chunk)
        return self.pipeline.forward_chunk(chunk)
