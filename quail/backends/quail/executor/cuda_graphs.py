"""Capture decode forwards in persistent CUDA buffers.

A chunk that fits a captured graph is copied into the graph's input
buffers, padded to the graph's rows, and replayed: the forward then
costs no kernel launches on the host. Every bucket's graph reads the
same buffers and writes the same output rows, and the graphs share
one memory pool, so the captures cost one set of inputs and one
forward's temporaries however many buckets there are. Graphs are
captured for the tree layout (fresh rows in one call, resident KV in
another, merged) and for the unified layout (every row scattered into
its pages, then one causal paged call), as the model allows.
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


class DecodeBuffers:
    """Fixed GPU inputs for decode batches of up to rows rows, in any layout.

    A graph captured over the first n rows reads views of these
    buffers, so one set serves every bucket, and the layouts share
    them. Padding has no KV writes and no resident context. Every
    padded query forms a separate sequence so it cannot affect real
    rows. The shared layout's stacked read pads with sequences that
    read no keys, and no real row's merge reads their results.
    """

    def __init__(self, torch, rows, page_tokens, *,
                 config: DecodeGraphConfig | None = None, device="cuda"):
        config = DecodeGraphConfig() if config is None else config
        self.rows = rows
        self.config = config
        self.indices = torch.arange(rows + 1, dtype=torch.int32, device=device)

        def zeros(shape, dtype=torch.int64):
            return torch.zeros(shape, dtype=dtype, device=device)

        width = (config.max_context_tokens + page_tokens - 1) // page_tokens
        self.ids = zeros(rows)
        self.positions = zeros(rows)
        self.src = zeros(rows)
        self.dst = zeros(rows)
        self.cu_a = self.indices.clone()
        self.read_rows = zeros(rows)
        self.cu_q = self.indices.clone()
        self.used = zeros(rows, torch.int32)
        self.table = zeros((rows, width), torch.int32)
        self.source = zeros(rows, torch.int32)
        # the shared layout's stacked read of the pages rounds share
        self.shared_rows = zeros(rows)
        self.shared_cu = self.indices.clone()
        self.shared_used = zeros(rows, torch.int32)
        self.shared_table = zeros((rows, width), torch.int32)
        self.shared_source = zeros(rows, torch.int32)
        self.chunk = self.view(rows)

    def view(self, rows) -> Chunk:
        """A tree-layout chunk over the first rows rows of the buffers."""
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

    def view_unified(self, rows) -> Chunk:
        """A unified-layout chunk over the first rows rows of the buffers."""
        config = self.config
        return Chunk(
            input_ids=self.ids[:rows], positions=self.positions[:rows],
            final_indices=self.indices[:rows].to(self.ids.dtype),
            meta=dict(layer=0, canvas=None, gdn=None, reads=None,
                      kv_src=None, kv_dst=None,
                      cu_a=self.cu_a[:rows + 1], max_a=config.max_query_tokens,
                      unified=dict(src=self.src[:rows], dst=self.dst[:rows],
                                   tail_src=None, tail_dst=None,
                                   used=self.used[:rows], table=self.table[:rows],
                                   max_used=config.max_context_tokens,
                                   cu_q=self.cu_q[:rows + 1],
                                   max_q=config.max_query_tokens)),
            attention_mode="unified", tokens=rows, layout=[])

    def view_shared(self, rows) -> Chunk:
        """A unified chunk with a stacked shared read, over the first rows rows."""
        config = self.config
        chunk = self.view_unified(rows)
        chunk.meta["unified"]["shared"] = dict(
            rows=self.shared_rows[:rows], cu_q=self.shared_cu[:rows + 1],
            max_q=config.max_query_tokens, used=self.shared_used[:rows],
            max_used=config.max_context_tokens, table=self.shared_table[:rows],
            source=self.shared_source[:rows])
        return chunk

    def synthetic(self, page):
        """Set independent one-token sequences reading one scratch KV row."""
        self.ids.fill_(10)
        self.positions.fill_(1)
        self.src.zero_()
        self.dst.fill_(-1)
        self.used.fill_(1)
        self.table.fill_(page)
        self.source.copy_(self.indices[:self.rows])
        self.shared_rows.copy_(self.indices[:self.rows])
        self.shared_used.fill_(1)
        self.shared_table.fill_(page)
        self.shared_source.copy_(self.indices[:self.rows])

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
        if chunk.attention_mode == "unified":
            self._copy_unified(chunk, rows)
            return
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

    def _copy_shared(self, shared, n, rows):
        """Copy a chunk's stacked shared read; its padding reads nothing."""
        m = shared["rows"].numel()
        sequences = shared["used"].numel()
        self._bounds(self.shared_cu, shared["cu_q"], m, rows)
        self.shared_rows[:rows].zero_()
        self.shared_rows[:m].copy_(shared["rows"])
        self.shared_used[:rows].zero_()
        self.shared_used[:sequences].copy_(shared["used"])
        self.shared_table[:sequences, :shared["table"].shape[1]].copy_(shared["table"])
        self.shared_source[:rows].fill_(-1)
        self.shared_source[:n].copy_(shared["source"])

    def _copy_unified(self, chunk, rows):
        """Copy a unified-layout chunk; padding reads one row of its page."""
        n = chunk.tokens
        unified = chunk.meta["unified"]
        sequences = unified["used"].numel()
        self.ids[:rows].zero_()
        self.ids[:n].copy_(chunk.input_ids)
        self.positions[:rows].zero_()
        self.positions[:n].copy_(chunk.positions)
        self._bounds(self.cu_a, chunk.meta["cu_a"], n, rows)
        self._bounds(self.cu_q, unified["cu_q"], n, rows)
        # a padded sequence's one query reads one resident row, so the
        # causal call has a key for it
        self.used[:rows].fill_(1)
        self.used[:sequences].copy_(unified["used"])
        self.table[:sequences, :unified["table"].shape[1]].copy_(unified["table"])
        self.src[:rows].zero_()
        self.dst[:rows].fill_(-1)
        writes = unified["src"].numel()
        self.src[:writes].copy_(unified["src"])
        self.dst[:writes].copy_(unified["dst"])
        if unified.get("shared") is not None:
            self._copy_shared(unified["shared"], n, rows)


class PieceGraphs:
    """Replay a pipeline's work between attention calls; run attention eagerly.

    Everything but attention depends only on a pass's row count, so one
    graph per piece and row bucket serves every pass of that size, of
    any operator and attention layout. A pass copies its ids into the
    padded buffers, replays piece 0, then for each layer runs attention
    on its real rows and replays the next piece. Padded rows never
    reach attention, so they read and write no KV.

    The pipeline provides num_pieces, piece_widths(), piece(),
    attention_piece(), and final_piece().
    """

    def __init__(self, pipeline, config: DecodeGraphConfig):
        self.pipeline = pipeline
        self.config = config
        self.graphs = {}        # bucket rows -> one graph per piece
        self.buffers = None
        self.pool = None
        self.replays = 0

    def _views(self, rows):
        ids, hidden, residual, qkv, out = self.buffers
        return ids[:rows], hidden[:rows], residual[:rows], qkv[:rows], out[:rows]

    def prepare(self, torch, stream):
        """Capture every piece of every bucket, once; return whether it captured."""
        if self.graphs:
            return False
        pipeline = self.pipeline
        rows = self.config.row_buckets[-1]
        width, qkv_width = pipeline.piece_widths()
        dtype = pipeline.embed.weight.dtype

        def rows_of(columns):
            return torch.zeros((rows, columns), dtype=dtype, device="cuda")

        self.buffers = (torch.zeros(rows, dtype=torch.int64, device="cuda"),
                        rows_of(width), rows_of(width), rows_of(qkv_width),
                        rows_of(width))
        self.pool = torch.cuda.graph_pool_handle()
        for index, bucket in enumerate(self.config.row_buckets):
            views = self._views(bucket)
            # the first bucket warms every kernel; the rest need one
            # pass at their own shape
            for _ in range(3 if index == 0 else 1):
                for piece in range(pipeline.num_pieces):
                    pipeline.piece(piece, *views)
            stream.synchronize()
            graphs = []
            for piece in range(pipeline.num_pieces):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                    pipeline.piece(piece, *views)
                graphs.append(graph)
            self.graphs[bucket] = graphs
        return True

    def supports(self, chunk):
        """Whether a pass's rows fit a captured bucket and its model has no state."""
        return bool(self.graphs and 0 < chunk.tokens <= self.config.row_buckets[-1]
                    and chunk.meta.get("canvas") is None
                    and chunk.meta.get("gdn") is None)

    def forward(self, chunk):
        """Run a pass: the pieces replayed, attention eager on the real rows."""
        n = chunk.tokens
        bucket = self.config.bucket(n)
        graphs = self.graphs[bucket]
        ids, hidden, residual, qkv, out = self._views(bucket)
        ids[:n].copy_(chunk.input_ids)
        ids[n:].zero_()
        chunk.meta["layer"] = 0
        graphs[0].replay()
        for index, graph in enumerate(graphs[1:]):
            hidden[:n].copy_(self.pipeline.attention_piece(index, qkv[:n], chunk))
            graph.replay()
        self.replays += 1
        return self.pipeline.final_piece(out[:n], residual[:n], chunk.final_indices)


class DecodeGraphs:
    """Capture and replay bounded decode graphs for one pipeline and KV arena.

    Calls use the worker's current stream. Output rows are copied before
    returning so a subsequent replay can reuse the graph's output storage.
    """

    def __init__(self, pipeline, *, config: DecodeGraphConfig | None = None):
        self._config = DecodeGraphConfig() if config is None else config
        self.pipeline = pipeline
        self.enabled = True
        # per layout, bucket rows -> its captured graph, and the chunk
        # its graph reads
        self.graphs = {"tree": {}, "unified": {}, "shared": {}}
        self.views = {"tree": {}, "unified": {}, "shared": {}}
        self.buffers = None
        self.output = None      # every graph writes its rows here
        self.pool = None
        self.storage = None
        self.replays = 0
        self.capture_s = 0.0
        # every other pass replays the work between attention calls
        self.pieces = (PieceGraphs(pipeline, self._config)
                       if hasattr(pipeline, "piece") else None)

    @property
    def config(self) -> DecodeGraphConfig:
        """Return the immutable limits used for capture and replay."""
        return self._config

    def invalidate(self):
        """Release graphs before replacing their KV allocation."""
        self.graphs = {"tree": {}, "unified": {}, "shared": {}}
        self.views = {"tree": {}, "unified": {}, "shared": {}}
        self.buffers = None
        self.output = None
        self.pool = None
        self.storage = None

    def layouts(self) -> tuple:
        """The attention layouts the model's kernels let the graphs capture."""
        engine = self.pipeline.engine
        arena = engine.arena
        if not self.enabled or engine.fa_version != 3 or arena.has_sliding:
            return ()
        layouts = ("tree",) if self.pipeline.tree_attention else ()
        if not getattr(arena, "num_state_slots", 0):
            layouts += ("unified", "shared")
        return layouts

    def _prepare_pieces(self) -> bool:
        """Capture the piece graphs once; they read no KV, so they outlive it."""
        if self.pieces is None or not self.enabled:
            return False
        torch = self.pipeline.engine.torch
        started = time.perf_counter()
        stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream), torch.inference_mode():
            captured = self.pieces.prepare(torch, stream)
        current.wait_stream(stream)
        stream.synchronize()
        if captured:
            logger.info("captured %d piece graphs in %.2f s",
                        sum(len(graphs) for graphs in self.pieces.graphs.values()),
                        time.perf_counter() - started)
        return captured

    def prepare(self):
        """Capture every bucket of every layout on scratch KV, once per allocation.

        The piece graphs are captured on the first call.
        """
        pieces = self._prepare_pieces()
        engine = self.pipeline.engine
        arena, torch = engine.arena, engine.torch
        layouts = self.layouts()
        if not layouts:
            return pieces
        if self.storage is arena.k:
            return pieces
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
                for layout in layouts:
                    view = {"tree": self.buffers.view,
                            "unified": self.buffers.view_unified,
                            "shared": self.buffers.view_shared}[layout]
                    for index, rows in enumerate(self.config.row_buckets):
                        # the graph reads the view's tensors by address,
                        # so the view lives as long as the graph
                        chunk = self.views[layout][rows] = view(rows)
                        # the first capture warms every kernel; the
                        # rest need one pass at their own shape
                        for _ in range(3 if index == 0 else 1):
                            out = self.pipeline.forward_chunk(chunk)
                        if self.output is None:
                            self.output = torch.empty(
                                (self.buffers.rows, *out.shape[1:]),
                                dtype=out.dtype, device=out.device)
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                            self.output[:rows].copy_(
                                self.pipeline.forward_chunk(chunk))
                        self.graphs[layout][rows] = graph
            current.wait_stream(stream)
            stream.synchronize()
        except Exception:
            self.invalidate()
            raise
        finally:
            arena.free_key(key)
        self.storage = arena.k
        self.capture_s = time.perf_counter() - started
        logger.info("captured %d decode graphs (%s) in %.2f s",
                    sum(len(graphs) for graphs in self.graphs.values()),
                    ", ".join(layouts), self.capture_s)
        return True

    def supports(self, chunk):
        """Whether the batch fits a captured layout."""
        arena = self.pipeline.engine.arena
        if (not self.enabled or self.storage is not arena.k
                or not 0 < chunk.tokens <= self.config.row_buckets[-1]
                or chunk.meta.get("canvas") is not None
                or chunk.meta.get("gdn") is not None
                or chunk.meta["max_a"] > self.config.max_query_tokens):
            return False
        table_width = ((self.config.max_context_tokens + arena.page_tokens - 1)
                       // arena.page_tokens)
        if chunk.attention_mode == "tree":
            reads = chunk.meta["reads"]
            if reads is None or reads.get("nodes") is not None:
                return False
            return not (reads["max_q"] > self.config.max_query_tokens
                        or reads["max_used"] > self.config.max_context_tokens
                        or reads["table"].shape[1] > table_width)
        unified = chunk.meta["unified"]
        shared = None if unified is None else unified.get("shared")
        if shared is not None and (
                shared["max_q"] > self.config.max_query_tokens
                or shared["max_used"] > self.config.max_context_tokens
                or shared["table"].shape[1] > table_width):
            return False
        return not (unified is None or unified.get("tail_src") is not None
                    or "sliding" in unified
                    or unified["max_q"] > self.config.max_query_tokens
                    or unified["max_used"] > self.config.max_context_tokens
                    or unified["table"].shape[1] > table_width)

    def forward(self, chunk):
        """Replay a matching graph, falling back to the ordinary forward."""
        if self.supports(chunk):
            rows = self.config.bucket(chunk.tokens)
            layout = chunk.attention_mode
            if layout == "unified" and chunk.meta["unified"].get("shared") is not None:
                layout = "shared"
            graph = self.graphs[layout].get(rows)
            if graph is not None:
                self.buffers.copy(chunk, rows)
                graph.replay()
                self.replays += 1
                return self.output.index_select(0, chunk.final_indices)
        if self.pieces is not None and self.pieces.supports(chunk):
            return self.pieces.forward(chunk)
        return self.pipeline.forward_chunk(chunk)
