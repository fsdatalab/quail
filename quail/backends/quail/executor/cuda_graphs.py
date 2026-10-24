"""Capture tree-attention forwards in persistent CUDA buffers.

A chunk that fits a captured graph is copied into the graph's input
buffers, padded to the graph's rows, and replayed: the forward then
costs no kernel launches on the host. Every bucket's graph reads the
same buffers and writes the same output rows, and the graphs share
one memory pool, so the captures cost one set of inputs and one
forward's temporaries however many buckets there are.
"""

import time
from dataclasses import dataclass

from quail.backends.quail.executor.chunk import Chunk
from quail.progress import logger


def _default_buckets() -> tuple:
    """Row buckets: finer where chunks are small, so little is padded."""
    return (tuple(range(64, 1025, 64)) + tuple(range(1152, 2049, 128))
            + tuple(range(2304, 4097, 256)))


@dataclass(frozen=True)
class DecodeGraphConfig:
    """Capture shapes owned by one worker's decode graphs.

    Attributes:
        row_buckets: The rows each captured graph holds, increasing; a
            chunk replays the smallest bucket that holds its rows.
        max_context_tokens: The most resident tokens a sequence reads.
        max_query_tokens: The longest query a captured chunk may hold.
    """

    row_buckets: tuple[int, ...] = _default_buckets()
    max_context_tokens: int = 65536
    max_query_tokens: int = 4096

    def __post_init__(self):
        if (not isinstance(self.row_buckets, tuple) or not self.row_buckets
                or any(type(row) is not int or row < 1 for row in self.row_buckets)
                or any(left >= right for left, right in zip(
                    self.row_buckets, self.row_buckets[1:]))):
            raise ValueError("row_buckets must be increasing positive integers")
        if (type(self.max_context_tokens) is not int
                or self.max_context_tokens < 1
                or type(self.max_query_tokens) is not int
                or self.max_query_tokens < 1):
            raise ValueError(
                "capture context and query limits must be positive integers")

    def bucket(self, tokens: int) -> int | None:
        """Return the smallest bucket holding this many rows, or None."""
        for rows in self.row_buckets:
            if rows >= tokens:
                return rows
        return None


def configured_graphs() -> DecodeGraphConfig:
    """Return the capture limits, as the environment overrides them.

    QUAIL_DECODE_GRAPH_ROWS lists the row buckets, comma-separated;
    QUAIL_DECODE_GRAPH_QUERY_TOKENS is the longest query a captured
    chunk may hold, and QUAIL_DECODE_GRAPH_CONTEXT_TOKENS the most
    resident tokens a sequence reads; unset, the defaults apply.
    """
    import os

    settings = {}
    rows = os.environ.get("QUAIL_DECODE_GRAPH_ROWS")
    if rows:
        settings["row_buckets"] = tuple(int(part) for part in rows.split(","))
    query = os.environ.get("QUAIL_DECODE_GRAPH_QUERY_TOKENS")
    if query:
        settings["max_query_tokens"] = int(query)
    context = os.environ.get("QUAIL_DECODE_GRAPH_CONTEXT_TOKENS")
    if context:
        settings["max_context_tokens"] = int(context)
    return DecodeGraphConfig(**settings)


class DecodeBuffers:
    """Fixed GPU inputs for tree-attention batches of up to rows rows.

    A graph captured over the first n rows reads views of these
    buffers, so one set serves every bucket. Padding has no KV writes
    and no resident context. Every padded query forms a separate
    sequence so it cannot affect real rows.
    """

    def __init__(self, torch, rows, page_tokens, *,
                 config: DecodeGraphConfig | None = None, device="cuda"):
        config = DecodeGraphConfig() if config is None else config
        self.rows = rows
        self.config = config
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
        self.table = zeros((rows, (config.max_context_tokens + page_tokens - 1)
                            // page_tokens), torch.int32)
        self.source = zeros(rows, torch.int32)
        self.chunk = self.view(rows)

    def view(self, rows) -> Chunk:
        """A chunk over the first rows rows of the buffers."""
        config = self.config
        return Chunk(
            input_ids=self.ids[:rows], positions=self.positions[:rows],
            final_indices=self.indices[:rows].to(self.ids.dtype),
            meta=dict(layer=0, canvas=None, unified=None,
                      kv_src=self.src[:rows], kv_dst=self.dst[:rows],
                      cu_a=self.cu_a[:rows + 1], max_a=config.max_query_tokens,
                      reads=dict(rows=self.read_rows[:rows], cu_q=self.cu_q[:rows + 1],
                                 max_q=config.max_query_tokens, used=self.used[:rows],
                                 max_used=config.max_context_tokens,
                                 table=self.table[:rows], source=self.source[:rows])),
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

    def _bounds(self, target, source, count, rows):
        sequences = source.numel() - 1
        padding = rows - count
        target = target[:rows + 1]
        target.fill_(rows)
        target[:sequences + 1].copy_(source)
        target[sequences:sequences + padding + 1].copy_(
            self.indices[:padding + 1] + count)

    def copy(self, chunk, rows=None):
        """Copy real inputs into the first rows rows, clearing the unused ones."""
        rows = self.rows if rows is None else rows
        n = chunk.tokens
        reads = chunk.meta["reads"]
        m = reads["rows"].numel()
        sequences = reads["used"].numel()
        self.ids[:rows].zero_()
        self.ids[:n].copy_(chunk.input_ids)
        self.positions[:rows].zero_()
        self.positions[:n].copy_(chunk.positions)
        self._bounds(self.cu_a, chunk.meta["cu_a"], n, rows)
        self._bounds(self.cu_q, reads["cu_q"], m, rows)
        self.read_rows[:rows].zero_()
        self.read_rows[:m].copy_(reads["rows"])
        self.used[:rows].zero_()
        self.used[:sequences].copy_(reads["used"])
        # Entries beyond used/page_tokens are never read by FlashAttention.
        self.table[:sequences, :reads["table"].shape[1]].copy_(reads["table"])
        self.source[:rows].fill_(-1)
        self.source[:n].copy_(reads["source"])
        self.src[:rows].zero_()
        self.dst[:rows].fill_(-1)
        if chunk.meta["kv_src"] is not None:
            writes = chunk.meta["kv_src"].numel()
            self.src[:writes].copy_(chunk.meta["kv_src"])
            self.dst[:writes].copy_(chunk.meta["kv_dst"])


class DecodeGraphs:
    """Capture and replay bounded tree-attention graphs for one pipeline and KV arena.

    Calls use the worker's current stream. Output rows are copied before
    returning so a subsequent replay can reuse the graph's output storage.
    """

    def __init__(self, pipeline, *, config: DecodeGraphConfig | None = None):
        self._config = configured_graphs() if config is None else config
        self.pipeline = pipeline
        self.enabled = True
        self.graphs = {}        # bucket rows -> its captured graph
        self.views = {}         # bucket rows -> the chunk its graph reads
        self.buffers = None
        self.output = None      # every graph writes its rows here
        self.pool = None
        self.storage = None
        self.replays = 0
        self.capture_s = 0.0

    @property
    def config(self) -> DecodeGraphConfig:
        """Return the immutable limits used for capture and replay."""
        return self._config

    def invalidate(self):
        """Release graphs before replacing their KV allocation."""
        self.graphs.clear()
        self.views.clear()
        self.buffers = None
        self.output = None
        self.pool = None
        self.storage = None

    def prepare(self):
        """Capture every bucket using scratch KV, once per allocation."""
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
                self.buffers = DecodeBuffers(torch, self.config.row_buckets[-1],
                                             arena.page_tokens, config=self.config)
                self.buffers.synthetic(page)
                self.pool = torch.cuda.graph_pool_handle()
                for index, rows in enumerate(self.config.row_buckets):
                    # the graph reads the view's tensors by address, so
                    # the view lives as long as the graph
                    chunk = self.views[rows] = self.buffers.view(rows)
                    # the first capture warms every kernel; the rest
                    # need one pass at their own shape
                    for _ in range(3 if index == 0 else 1):
                        out = self.pipeline.forward_chunk(chunk)
                    if self.output is None:
                        self.output = torch.empty(
                            (self.buffers.rows, *out.shape[1:]), dtype=out.dtype,
                            device=out.device)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                        self.output[:rows].copy_(self.pipeline.forward_chunk(chunk))
                    self.graphs[rows] = graph
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
                or not 0 < chunk.tokens <= self.config.row_buckets[-1]
                or chunk.meta.get("canvas") is not None
                or chunk.meta["max_a"] > self.config.max_query_tokens):
            return False
        reads = chunk.meta["reads"]
        if (reads is None or reads.get("nodes") is not None
                or reads["max_q"] > self.config.max_query_tokens
                or reads["max_used"] > self.config.max_context_tokens
                or reads["table"].shape[1] > (
                    self.config.max_context_tokens
                    + self.pipeline.engine.arena.page_tokens - 1
                ) // self.pipeline.engine.arena.page_tokens):
            return False
        return True

    def forward(self, chunk):
        """Replay a matching graph, falling back to the ordinary forward."""
        if self.supports(chunk):
            rows = self.config.bucket(chunk.tokens)
            graph = self.graphs.get(rows)
            if graph is not None:
                self.buffers.copy(chunk, rows)
                graph.replay()
                self.replays += 1
                return self.output.index_select(0, chunk.final_indices)
        return self.pipeline.forward_chunk(chunk)
