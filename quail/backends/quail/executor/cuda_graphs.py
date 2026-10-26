"""Capture tree-attention decode forwards in persistent CUDA buffers.

Every graph reads views of one set of input buffers sized for the
largest bucket and writes the same output rows, and the graphs share
one memory pool, so the captures cost one set of inputs and one
forward's temporaries however many buckets there are.
"""

import time
from dataclasses import dataclass

from quail.backends.quail.executor.chunk import Chunk
from quail.progress import logger


def _default_buckets() -> tuple:
    """Row buckets: finer where chunks are small, so little is padded."""
    return ((32,) + tuple(range(64, 1025, 64)) + tuple(range(1152, 2049, 128))
            + tuple(range(2304, 4097, 256)))


@dataclass(frozen=True)
class DecodeGraphConfig:
    """Capture shapes owned by one worker's decode graphs."""

    row_buckets: tuple[int, ...] = _default_buckets()
    max_context_tokens: int = 65536
    max_query_tokens: int = 32
    # the rows of one document's map branches read together
    max_document_rows: int = 128

    def __post_init__(self):
        if (not isinstance(self.row_buckets, tuple) or not self.row_buckets
                or any(type(row) is not int or row < 1 for row in self.row_buckets)
                or any(left >= right for left, right in zip(
                    self.row_buckets, self.row_buckets[1:]))):
            raise ValueError("row_buckets must be increasing positive integers")
        if (type(self.max_context_tokens) is not int
                or self.max_context_tokens < 1
                or type(self.max_query_tokens) is not int
                or self.max_query_tokens < 1
                or type(self.max_document_rows) is not int
                or self.max_document_rows < 1):
            raise ValueError(
                "capture context and query limits must be positive integers")


class DecodeBuffers:
    """Fixed GPU inputs for a bounded tree-attention batch.

    Padding has no KV writes and no resident context. Every padded
    query forms a separate sequence so it cannot affect real rows.
    With documents, the batch also carries the document read of map
    branches (see Stage.document_reads); its padding merges into call
    B's last row, which a batch with documents leaves unread.

    Given a base of at least as many rows, every buffer is a view of
    the base's first rows, so buffers of several sizes share storage.
    """

    def __init__(self, torch, rows, page_tokens, *,
                 config: DecodeGraphConfig | None = None, device="cuda",
                 documents=False, base=None):
        config = DecodeGraphConfig() if config is None else config
        self.rows = rows
        width = (config.max_context_tokens + page_tokens - 1) // page_tokens

        def buffer(name, shape, dtype=torch.int64):
            if base is not None:
                return getattr(base, name)[:shape[0]]
            return torch.zeros(shape, dtype=dtype, device=device)

        self.indices = (base.indices[:rows + 1] if base is not None else
                        torch.arange(rows + 1, dtype=torch.int32, device=device))
        self.ids = buffer("ids", (rows,))
        self.positions = buffer("positions", (rows,))
        self.src = buffer("src", (rows,))
        self.dst = buffer("dst", (rows,))
        self.cu_a = buffer("cu_a", (rows + 1,), torch.int32)
        self.read_rows = buffer("read_rows", (rows,))
        self.cu_q = buffer("cu_q", (rows + 1,), torch.int32)
        self.used = buffer("used", (rows,), torch.int32)
        self.table = buffer("table", (rows, width), torch.int32)
        self.source = buffer("source", (rows,), torch.int32)
        self.finals = buffer("finals", (rows,))
        self.document_index = buffer("document_index", (rows,))
        self.document_cu = buffer("document_cu", (rows + 1,), torch.int32)
        self.document_used = buffer("document_used", (rows,), torch.int32)
        self.document_table = buffer("document_table", (rows, width), torch.int32)
        if base is None:
            for each in (self.cu_a, self.cu_q, self.document_cu):
                each.copy_(self.indices)
            for each in (self.read_rows, self.finals, self.document_index):
                each.copy_(self.indices[:-1])
        reads = dict(rows=self.read_rows, cu_q=self.cu_q,
                     max_q=config.max_query_tokens, used=self.used,
                     max_used=config.max_context_tokens, table=self.table,
                     source=self.source)
        self.documents = None
        if documents:
            self.documents = dict(
                b_index=self.document_index, cu_q=self.document_cu,
                max_q=config.max_document_rows, used=self.document_used,
                max_used=config.max_context_tokens, table=self.document_table)
            reads["documents"] = self.documents
        self.chunk = Chunk(
            input_ids=self.ids, positions=self.positions,
            final_indices=self.finals,
            meta=dict(layer=0, canvas=None, unified=None,
                      kv_src=self.src, kv_dst=self.dst,
                      cu_a=self.cu_a, max_a=config.max_query_tokens,
                      reads=reads),
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
        self.document_used.fill_(1)
        self.document_table.fill_(page)

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
        if self.documents is not None:
            documents = reads["documents"]
            count = documents["b_index"].numel()
            groups = documents["used"].numel()
            index = self.documents["b_index"]
            # call B's rows past the real ones are never merged into
            # a real row, and a batch with documents has at least one
            index.fill_(self.rows - 1)
            index[:count].copy_(documents["b_index"])
            self._bounds(self.documents["cu_q"], documents["cu_q"], count)
            self.documents["used"].zero_()
            self.documents["used"][:groups].copy_(documents["used"])
            self.documents["table"][:groups, :documents["table"].shape[1]].copy_(
                documents["table"])


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

    def __init__(self, pipeline, *, config: DecodeGraphConfig | None = None):
        self._config = DecodeGraphConfig() if config is None else config
        self.pipeline = pipeline
        self.enabled = True
        self.graphs = []
        # the same buckets with the document read of map branches
        self.document_graphs = []
        self.pool = None
        self.storage = None
        self.replays = 0
        self.capture_s = 0.0
        self.capture_bytes = 0

    @property
    def config(self) -> DecodeGraphConfig:
        """Return the immutable limits used for capture and replay."""
        return self._config

    def invalidate(self):
        """Release graphs before replacing their KV allocation."""
        self.graphs.clear()
        self.document_graphs.clear()
        self.pool = None
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
        # device-wide free memory: the caching allocator may release
        # cached blocks while capturing, which hides the pool's growth
        free, _ = torch.cuda.mem_get_info()
        try:
            for k, v in zip(arena.k, arena.v):
                start = page * arena.page_tokens
                k[start:start + arena.page_tokens].zero_()
                v[start:start + arena.page_tokens].zero_()
            stream.wait_stream(current)
            with torch.cuda.stream(stream), torch.inference_mode():
                base = DecodeBuffers(torch, self.config.row_buckets[-1],
                                     arena.page_tokens, config=self.config)
                base.synthetic(page)
                self.pool = torch.cuda.graph_pool_handle()
                output = None   # every graph writes its rows here
                for documents, graphs in ((False, self.graphs),
                                          (True, self.document_graphs)):
                    for index, rows in enumerate(self.config.row_buckets):
                        buffers = DecodeBuffers(torch, rows, arena.page_tokens,
                                                config=self.config,
                                                documents=documents, base=base)
                        # the first bucket warms every kernel; the rest
                        # need one pass at their own shape
                        for _ in range(3 if index == 0 else 1):
                            out = self.pipeline.forward_chunk(buffers.chunk)
                        if output is None:
                            output = torch.empty((base.rows, *out.shape[1:]),
                                                 dtype=out.dtype, device=out.device)
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                            output[:rows].copy_(
                                self.pipeline.forward_chunk(buffers.chunk))
                        graphs.append(CapturedDecode(buffers, graph, output[:rows]))
            current.wait_stream(stream)
            stream.synchronize()
        except Exception:
            self.invalidate()
            raise
        finally:
            arena.free_key(key)
        self.storage = arena.k
        self.capture_s = time.perf_counter() - started
        self.capture_bytes = free - torch.cuda.mem_get_info()[0]
        logger.info("captured %d decode graphs of up to %d rows in %.2f s, "
                    "holding %.2f GiB", len(self.graphs) + len(self.document_graphs),
                    self.config.row_buckets[-1], self.capture_s,
                    self.capture_bytes / 2**30)
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
        page_tokens = self.pipeline.engine.arena.page_tokens
        width = (self.config.max_context_tokens + page_tokens - 1) // page_tokens
        if (reads is None or reads.get("nodes") is not None
                or reads["max_q"] > self.config.max_query_tokens
                or reads["max_used"] > self.config.max_context_tokens
                or reads["table"].shape[1] > width):
            return False
        documents = reads.get("documents")
        if documents is not None and (
                documents["max_q"] > self.config.max_document_rows
                or documents["max_used"] > self.config.max_context_tokens
                or documents["table"].shape[1] > width
                or chunk.tokens >= self.config.row_buckets[-1]):
            return False
        return True

    def forward(self, chunk):
        """Replay a matching graph, falling back to the ordinary forward."""
        if self.supports(chunk):
            documents = chunk.meta["reads"].get("documents") is not None
            # a batch with documents leaves call B's last row unread
            for captured in (self.document_graphs if documents else self.graphs):
                if captured.buffers.rows >= chunk.tokens + documents:
                    captured.buffers.copy(chunk)
                    captured.graph.replay()
                    self.replays += 1
                    return captured.output.index_select(0, chunk.final_indices)
        return self.pipeline.forward_chunk(chunk)
