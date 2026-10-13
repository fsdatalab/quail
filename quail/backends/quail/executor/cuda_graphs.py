"""Capture short tree-attention forwards in persistent CUDA buffers."""

import time
from dataclasses import dataclass

from quail.backends.quail.executor.chunk import Chunk
from quail.progress import logger

DECODE_ROWS = (32, 64, 128, 256, 512)
DECODE_CONTEXT = 65536
DECODE_QUERY = 32


class DecodeBuffers:
    """Fixed GPU inputs for a bounded tree-attention batch.

    Padding has no KV writes and no resident context. Every padded
    query forms a separate sequence so it cannot affect real rows.
    """

    def __init__(self, torch, rows, page_tokens, *, device="cuda"):
        self.rows = rows
        self.indices = torch.arange(rows + 1, dtype=torch.int32, device=device)

        def zeros(shape, dtype=torch.int64):
            return torch.zeros(shape, dtype=dtype, device=device)

        self.ids = zeros(rows)
        self.positions = zeros(rows)
        self.src = zeros(rows)
        self.dst = zeros(rows)
        self.cu_a = self.indices.clone()
        self.read_rows = self.indices[:-1].to(torch.int64)
        self.cu_q = self.indices.clone()
        self.used = zeros(rows, torch.int32)
        self.table = zeros((rows, (DECODE_CONTEXT + page_tokens - 1)
                            // page_tokens), torch.int32)
        self.source = zeros(rows, torch.int32)
        self.chunk = Chunk(
            input_ids=self.ids, positions=self.positions,
            final_indices=self.indices[:-1].to(torch.int64),
            meta=dict(layer=0, canvas=None, unified=None,
                      kv_src=self.src, kv_dst=self.dst,
                      cu_a=self.cu_a, max_a=DECODE_QUERY,
                      reads=dict(rows=self.read_rows, cu_q=self.cu_q,
                                 max_q=DECODE_QUERY, used=self.used,
                                 max_used=DECODE_CONTEXT, table=self.table,
                                 source=self.source)),
            attention_mode="tree", tokens=rows, layout=[])

    def synthetic(self, page):
        """Set independent one-token sequences reading one scratch KV row."""
        self.ids.fill_(10)
        self.positions.fill_(1)
        self.src.zero_()
        self.dst.fill_(-1)
        self.used.fill_(1)
        self.table.fill_(page)
        self.source.copy_(self.indices[:-1])

    def _bounds(self, target, source, count):
        sequences = source.numel() - 1
        padding = self.rows - count
        target.fill_(self.rows)
        target[:sequences + 1].copy_(source)
        target[sequences:sequences + padding + 1].copy_(
            self.indices[:padding + 1] + count)

    def copy(self, chunk):
        """Copy real inputs and clear the unused rows of a captured batch."""
        n = chunk.tokens
        reads = chunk.meta["reads"]
        m = reads["rows"].numel()
        sequences = reads["used"].numel()
        self.ids.zero_()
        self.ids[:n].copy_(chunk.input_ids)
        self.positions.zero_()
        self.positions[:n].copy_(chunk.positions)
        self._bounds(self.cu_a, chunk.meta["cu_a"], n)
        self._bounds(self.cu_q, reads["cu_q"], m)
        self.read_rows.zero_()
        self.read_rows[:m].copy_(reads["rows"])
        self.used.zero_()
        self.used[:sequences].copy_(reads["used"])
        # Entries beyond used/page_tokens are never read by FlashAttention.
        self.table[:sequences, :reads["table"].shape[1]].copy_(reads["table"])
        self.source.fill_(-1)
        self.source[:n].copy_(reads["source"])
        self.src.zero_()
        self.dst.fill_(-1)
        if chunk.meta["kv_src"] is not None:
            writes = chunk.meta["kv_src"].numel()
            self.src[:writes].copy_(chunk.meta["kv_src"])
            self.dst[:writes].copy_(chunk.meta["kv_dst"])


@dataclass
class CapturedDecode:
    """A graph and its persistent inputs and output."""

    buffers: DecodeBuffers
    graph: object
    output: object


class DecodeGraphs:
    """Warm and replay bounded decode graphs for one pipeline and KV arena.

    Calls use the worker's current stream. Output rows are copied before
    returning so a subsequent replay can reuse the graph's output storage.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.enabled = True
        self.graphs = []
        self.storage = None
        self.replays = 0
        self.fallbacks = 0
        self.capture_s = 0.0

    def invalidate(self):
        """Release graphs before replacing their KV allocation."""
        self.graphs.clear()
        self.storage = None

    def prepare(self):
        """Capture supported shapes using scratch KV, once per allocation."""
        engine = self.pipeline.engine
        arena, torch = engine.arena, engine.torch
        if (not self.enabled or not self.pipeline.tree_attention
                or engine.fa_version != 3 or arena.has_sliding):
            return False
        if self.storage is arena.k:
            return False
        self.invalidate()
        started = time.perf_counter()
        scratch = arena.alloc_temporary(arena.page_tokens)
        if scratch is None:
            raise RuntimeError("decode capture needs one free KV page")
        key, _ = scratch
        page = arena.owned_pages(key)[0]
        stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        try:
            for k, v in zip(arena.k, arena.v):
                start = page * arena.page_tokens
                k[start:start + arena.page_tokens].zero_()
                v[start:start + arena.page_tokens].zero_()
            stream.wait_stream(current)
            with torch.cuda.stream(stream), torch.inference_mode():
                for rows in DECODE_ROWS:
                    buffers = DecodeBuffers(torch, rows, arena.page_tokens)
                    buffers.synthetic(page)
                    for _ in range(3):
                        self.pipeline.forward_chunk(buffers.chunk)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = self.pipeline.forward_chunk(buffers.chunk)
                    self.graphs.append(CapturedDecode(buffers, graph, output))
            current.wait_stream(stream)
            stream.synchronize()
        except Exception:
            self.invalidate()
            raise
        finally:
            arena.free_key(key)
        self.storage = arena.k
        self.capture_s = time.perf_counter() - started
        logger.info("captured %d decode graphs in %.2f s",
                    len(self.graphs), self.capture_s)
        return True

    def supports(self, chunk):
        """Whether the batch fits the captured tree-attention layout."""
        if (not self.enabled or self.storage is not self.pipeline.engine.arena.k
                or chunk.attention_mode != "tree"
                or not 0 < chunk.tokens <= DECODE_ROWS[-1]
                or chunk.meta.get("canvas") is not None
                or chunk.meta["max_a"] > DECODE_QUERY):
            return False
        reads = chunk.meta["reads"]
        if (reads is None or reads.get("nodes") is not None
                or reads["max_q"] > DECODE_QUERY
                or reads["max_used"] > DECODE_CONTEXT
                or reads["table"].shape[1] > (
                    DECODE_CONTEXT + self.pipeline.engine.arena.page_tokens - 1
                ) // self.pipeline.engine.arena.page_tokens):
            return False
        return True

    def forward(self, chunk):
        """Replay a matching graph, falling back to the ordinary forward."""
        if self.supports(chunk):
            for captured in self.graphs:
                if captured.buffers.rows >= chunk.tokens:
                    captured.buffers.copy(chunk)
                    captured.graph.replay()
                    self.replays += 1
                    return captured.output.index_select(0, chunk.final_indices)
        self.fallbacks += 1
        return self.pipeline.forward_chunk(chunk)
