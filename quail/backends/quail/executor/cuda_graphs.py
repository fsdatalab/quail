"""Capture dense layer work while leaving attention and its metadata eager."""

import time
from dataclasses import dataclass
from functools import partial

from quail.cost.budgets import graph_memory_bytes, graph_token_buckets
from quail.progress import logger


class PiecewiseBuffers:
    """Persistent boundary tensors for one padded token count.

    Only real rows reach attention. Padding therefore has no KV writes,
    sequence boundaries, or reads, regardless of the attention path.
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
        self.o_in, self.o_scale = engine.quant(self.q)

    def copy(self, chunk):
        """Stage real input rows and clear padding from the previous pass."""
        self.ids.zero_()
        self.positions.zero_()
        self.ids[:chunk.tokens].copy_(chunk.input_ids)
        self.positions[:chunk.tokens].copy_(chunk.positions)

    def attention(self, pipeline, chunk):
        """Run the ordinary attention path on real rows with original metadata."""
        n = chunk.tokens
        o_in, o_scale = pipeline.engine.attention(
            self.q[:n], self.k[:n], self.v[:n], chunk)
        self.o_in[:n].copy_(o_in)
        if o_scale is not None:
            self.o_scale[:n].copy_(o_scale)

    def piece(self, pipeline, index):
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


@dataclass
class CapturedPieces:
    """A bucket's boundary buffers and ordered non-attention graphs."""

    buffers: PiecewiseBuffers
    graphs: list

    def forward(self, pipeline, chunk):
        """Replay pieces separated by eager attention and copy selected outputs."""
        self.buffers.copy(chunk)
        chunk.meta["layer"] = 0
        self.graphs[0].replay()
        for graph in self.graphs[1:]:
            self.buffers.attention(pipeline, chunk)
            graph.replay()
        return self.buffers.output.index_select(0, chunk.final_indices)


class PiecewiseGraphs:
    """Capture and replay Qwen3 dense work on the worker's current stream.

    No piece references KV. Resizing the arena does not invalidate these
    graphs. All temporary tensors die inside their piece, allowing the
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

    def prepare(self):
        """Capture all dense pieces after kernel warmup, once per worker."""
        if not self.enabled or self.graphs:
            return False
        torch = self.pipeline.engine.torch
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
                    graphs = []
                    for index in range(len(self.pipeline.layers) + 1):
                        piece = partial(buffers.piece, self.pipeline, index)
                        piece()
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=pool, stream=stream):
                            piece()
                        graphs.append(graph)
                    self.graphs.append(CapturedPieces(buffers, graphs))
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
            self.graphs.clear()
            raise
        self.capture_s = time.perf_counter() - started
        logger.info("captured %d piecewise buckets in %.2f s; %.1f MiB "
                    "storage / %.1f MiB budget",
                    len(self.graphs), self.capture_s, self.captured_bytes / 2**20,
                    self.memory_bytes / 2**20)
        return True

    def supports(self, chunk):
        """Whether a chunk fits capture without a model-specific forward hook."""
        return (self.enabled and bool(self.graphs)
                and 0 < chunk.tokens <= self.token_buckets[-1]
                and chunk.meta.get("canvas") is None
                and self.pipeline.forward_hook is None)

    def forward(self, chunk):
        """Replay dense pieces or run a larger chunk eagerly."""
        self.passes += 1
        if self.supports(chunk):
            for captured in self.graphs:
                if captured.buffers.num_tokens >= chunk.tokens:
                    self.replays += 1
                    return captured.forward(self.pipeline, chunk)
        return self.pipeline.forward_chunk(chunk)
