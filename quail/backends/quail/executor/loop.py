"""The overlapped chunk loop.

Pack on CPU while the GPU runs, gate answers, free pages immediately.

- run_join and run_filter: the join's stages and the filter chain's
  questions as stages of run_stages, the one scheduler (stages.py).
- pack_chunk: the tensors of one chunk.

Torch is imported lazily when model execution starts.
"""

import time

import numpy as np

from quail.backends.quail.executor.attention import (
    ATTENTION_PATHS,
    Chunk,
)
from quail.progress import Progress, logger, quiet


class ArenaFullError(RuntimeError):
    """A chunk's suffix pages do not fit the free KV arena."""


def _tick(timing, key, t0):
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + time.perf_counter() - t0
    return time.perf_counter()


def _staged(torch, data, dtype, pinned=True):
    """Host data to device through pinned memory, non-blocking.

    pinned=False reverts to pageable blocking copies.
    """
    if isinstance(data, np.ndarray):
        data = torch.as_tensor(data, dtype=dtype)
    if torch.is_tensor(data):
        if pinned:
            return data.pin_memory().to("cuda", non_blocking=True)
        return data.to("cuda")
    if pinned:
        return torch.tensor(data, dtype=dtype, pin_memory=True).to(
            "cuda", non_blocking=True)
    return torch.tensor(data, dtype=dtype, device="cuda")


def _token_parts(sequence):
    parts = getattr(sequence, "token_parts", None)
    if parts is None or (len(parts) == 1 and parts[0] is sequence):
        yield sequence
        return
    for part in parts:
        yield from _token_parts(part)


def _int64(part):
    """One token part as an int64 array, without a copy where possible."""
    if hasattr(part, "arrow_array"):
        part = part.arrow_array.to_numpy(zero_copy_only=False)
    elif hasattr(part, "numpy") and not isinstance(part, np.ndarray):
        part = part.numpy()
    return np.asarray(part, dtype=np.int64)


class Suffixes:
    """A group's suffix token lists as one flat array plus lengths.

    Attributes:
        ids: Every suffix's tokens back to back.
        lengths: Tokens per suffix.
        offsets: Where each suffix starts in ids, with a final entry
            equal to the total.
    """

    __slots__ = ("ids", "lengths", "offsets")

    def __init__(self, ids, lengths):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.offsets = np.zeros(len(self.lengths) + 1, dtype=np.int64)
        np.cumsum(self.lengths, out=self.offsets[1:])
        if self.offsets[-1] != len(self.ids):
            raise ValueError(
                f"{len(self.ids)} suffix tokens for lengths summing to "
                f"{self.offsets[-1]}")

    @classmethod
    def of(cls, sequences):
        """Flatten token sequences, each possibly made of parts."""
        lengths = [len(sequence) for sequence in sequences]
        arrays = [_int64(part) for sequence in sequences
                  for part in _token_parts(sequence) if len(part)]
        ids = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
        return cls(ids, lengths)

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        bounds = self.offsets.tolist()
        for start, end in zip(bounds, bounds[1:]):
            yield self.ids[start:end]

    def lengths_at(self, indices):
        """Tokens per suffix at these indices, in that order."""
        if isinstance(indices, range) and indices.step == 1:
            return self.lengths[indices.start:indices.stop]
        return self.lengths[np.asarray(indices, dtype=np.int64)]

    def take(self, indices):
        """The suffixes at these indices, in that order."""
        if isinstance(indices, range) and indices.step == 1:
            return Suffixes(
                self.ids[self.offsets[indices.start]:self.offsets[indices.stop]],
                self.lengths[indices.start:indices.stop])
        index = np.asarray(indices, dtype=np.int64)
        lengths = self.lengths[index]
        total = int(lengths.sum())
        # each suffix's rows read from its start in ids
        starts = np.cumsum(lengths) - lengths
        rows = (np.repeat(self.offsets[index] - starts, lengths)
                + np.arange(total, dtype=np.int64))
        return Suffixes(self.ids[rows], lengths)


class InputStaging:
    """Reuse CPU and GPU input buffers on the current CUDA stream."""

    def __init__(self, torch):
        self.torch = torch
        self.buffers = {}
        self.fixed_tokens = {}

    def host(self, name, count, dtype):
        torch = self.torch
        previous = self.buffers.get(name)
        if previous is not None:
            host, device, event = previous
            # The CPU must not overwrite a buffer while its transfer is pending.
            event.synchronize()
        if previous is None or host.numel() < count or host.dtype != dtype:
            host = torch.empty(count, dtype=dtype, pin_memory=True)
            device = torch.empty(count, dtype=dtype, device="cuda")
            event = torch.cuda.Event()
        self.buffers[name] = host, device, event
        return host[:count]

    def upload(self, name, count):
        host, device, event = self.buffers[name]
        # Reusing the device buffer is ordered after its previous readers.
        device[:count].copy_(host[:count], non_blocking=True)
        event.record()
        return device[:count]

    def copy(self, name, data, dtype):
        source = self.torch.as_tensor(data)
        host = self.host(name, source.numel(), dtype)
        host.copy_(source.reshape(-1))
        return self.upload(name, source.numel())

    def fixed(self, tokens):
        key = id(tokens)
        if key not in self.fixed_tokens:
            self.fixed_tokens[key] = tokens, self.torch.as_tensor(tokens)
        return self.fixed_tokens[key][1]


def _staged_token_parts(torch, sequences, total, pinned=True, staging=None):
    """Copy the chunk's token parts into one GPU input tensor.

    The parts are joined on the host first, so the chunk takes one
    copy into pinned memory however many suffixes it packs.
    """
    host = (torch.empty(total, dtype=torch.int64, pin_memory=pinned)
            if staging is None else staging.host("tokens", total, torch.int64))
    arrays = [_int64(part) for sequence in sequences
              for part in _token_parts(sequence) if len(part)]
    ids = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
    if len(ids) != total:
        raise AssertionError(
            f"packed {len(ids)} token ids into a {total}-token chunk")
    host.copy_(torch.from_numpy(ids))
    if staging is not None:
        return staging.upload("tokens", total)
    return host.to("cuda", non_blocking=pinned)


# ------------------------------------------------------- chunk packing

class _PoolView:
    """One key's rows and pages in one pool.

    start is the logical row the pool's first page holds: 0 on the
    every-token pool, the window origin on a trimmed key's sliding
    pool.
    """

    __slots__ = ("rows", "pages", "start")

    def __init__(self, rows, pages, start):
        self.rows = rows
        self.pages = pages
        self.start = start


class _PoolBuilder:
    """The scatter map, block table, and lengths of one pool for a chunk."""

    def __init__(self):
        self.page_rows = []
        self.lengths = []
        self.src = []
        self.dst = []
        self.tail_src = []
        self.tail_dst = []

    def add_sequence(self, pages, kv_tokens):
        self.page_rows.append(pages)
        self.lengths.append(kv_tokens)

    def scatter_direct(self, view, r0, r1, logical_start):
        """Rows r0..r1 land at logical_start.. in the key's own pages.

        Rows before the pool's start are not stored there.
        """
        first = max(logical_start, view.start)
        skip = first - logical_start
        if r0 + skip >= r1:
            return
        self.src.append(np.arange(r0 + skip, r1, dtype=np.int64))
        offset = first - view.start
        self.dst.append(view.rows[offset:offset + (r1 - r0 - skip)].numpy())

    def scatter_suffix(self, view, temp, s0, s1, f, remainder, page_tokens):
        """Suffix rows go to a temporary, after a copy of the kept tail."""
        suffix_tokens = s1 - s0
        self.src.append(np.arange(s0, s1, dtype=np.int64))
        self.dst.append(temp.rows[remainder:remainder + suffix_tokens].numpy())
        if remainder:
            anchor_page = view.pages[(f - view.start) // page_tokens]
            base = anchor_page * page_tokens
            self.tail_src.append(np.arange(base, base + remainder, dtype=np.int64))
            self.tail_dst.append(temp.rows[:remainder].numpy())

    def build(self, arena, stage, torch, staging, index):
        def concatenate(parts):
            return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

        tag = "" if index == 0 else "_sliding"
        if staging is None:
            table = arena.block_table_rows(self.page_rows)
        else:
            width = max(map(len, self.page_rows))
            block_table = np.zeros((len(self.page_rows), width), dtype=np.int32)
            for row, pages in enumerate(self.page_rows):
                block_table[row, :len(pages)] = pages
            table = stage(f"block_table{tag}", block_table, torch.int32).view(
                len(self.page_rows), width)
        return dict(
            src=stage(f"unified_src{tag}", concatenate(self.src), torch.int64),
            dst=stage(f"unified_dst{tag}", concatenate(self.dst), torch.int64),
            tail_src=(stage(f"tail_src{tag}", concatenate(self.tail_src),
                            torch.int64) if self.tail_src else None),
            tail_dst=(stage(f"tail_dst{tag}", concatenate(self.tail_dst),
                            torch.int64) if self.tail_dst else None),
            used=stage(f"unified_lengths{tag}", self.lengths, torch.int32),
            table=table,
            max_used=max(self.lengths))


def pack_chunk(torch, arena, groups, timing=None, pinned=True, *,
               attention_mode, staging=None, canvas=(), answer_row=0):
    """Build tensors for one chunk from groups in chunk order.

    Each group is a dict with keys:
      key       Arena key for KV reads/writes.
      prefix    Fresh prefix token list, or None when KV is resident.
      start     Logical position of the first fresh prefix token; the
                key's pages before it are borrowed from a parent and
                already hold KV. 0 when absent.
      read_key  Under tree attention, for a group with one suffix,
                the parent whose pages the fresh rows read for
                positions before start; the rows still write the
                group's own key. Consecutive groups with one read_key
                and start share one read of it. Without it, the fresh
                prefix reads the key's own borrowed pages and each
                suffix reads the whole key as usual.
      f         Kept-context length (suffix positions start here).
      suffixes  List of suffix token lists.
      write_suffix_tokens  Leading rows of the first suffix to scatter
                into the key's pages after the prefix.
      read_all_rows  When true, every row of each suffix feeds the
                readout, not only its last; the chunk's rows_per_answer
                then says how many rows each answer has.
      single    The group's one suffix is a stage's whole request. A
                fresh, non-borrowing single group under tree attention
                is one causal segment, [prefix | suffix], whose rows
                write the key's pages without reading them back: the
                computation an unpaged filter does.
      read_rows Per suffix, how many of its last rows are read; a
                group without it reads by read_all_rows.
      canvas    The group's own canvas token ids, packed after each of
                its suffixes in place of the chunk's canvas.

    A fresh group without arena pages packs [prefix | suffix] as one
    causal segment (no scatter, no paged read). This only works with
    at most one suffix. Under unified attention a chunk is either all
    paged or all unpaged.

    A group may carry chains, from ``trie_chains``: its one suffix is
    then several causal segments, each starting at its own position
    past f, and a segment's rows also read the rows above it in
    earlier segments through call C. Its document must be resident.

    canvas is the token ids a diffusion model denoises: they follow
    every suffix as extra rows, and the answer row is canvas row
    answer_row instead of the suffix's last row; with read_all_rows
    every canvas row is read instead. Canvas KV goes wherever the
    suffix's KV goes and is never kept. Canvas rows run the unified
    path only.
    """
    def stage(name, values, dtype):
        if staging is not None:
            return staging.copy(name, values, dtype)
        return _staged(torch, values, dtype, pinned)

    def concatenate(parts):
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    if attention_mode not in ATTENTION_PATHS:
        raise ValueError(
            f"attention_mode must be one of {ATTENTION_PATHS}, "
            f"got {attention_mode!r}")
    canvas = tuple(canvas)
    if attention_mode != "unified" and (
            canvas or any(g.get("canvas") is not None for g in groups)):
        raise ValueError("canvas rows run the unified attention path")
    if canvas and not 0 <= answer_row < len(canvas):
        raise ValueError(
            f"answer_row {answer_row} is outside the {len(canvas)}-row canvas")
    t = time.perf_counter() if timing is not None else 0.0
    # unified scatters every fresh row through its own src/dst map, so
    # the reads and kv_writes bookkeeping below is tree only
    tree_path = attention_mode == "tree"
    canvas_ids = np.asarray(canvas, dtype=np.int64)
    canvas_starts = []    # per group, the first row of each canvas
    canvas_sizes = []     # per group, the rows of each canvas
    canvas_seq = []       # its sequence in the unified paged call
    id_parts, token_count = [], 0
    pos, finals = [], []
    cu_a = [np.zeros(1, dtype=np.int64)]
    rows_per_answer = []
    multi_row = False
    # rows that read resident KV in call B: a borrowing document's
    # rows reading its parent, and a kept document's tail reading its
    # own prefix (wider than the planner's readers, which are children)
    reader_rows = []
    kv_writes, layout = [], []
    fresh_keys = []
    read_keys, read_used, cu_q = [], [], [0]
    max_q = 0
    unified_groups = []
    # call C: chain rows reading their off-chain ancestor rows
    node_q, node_k, node_cu_q, node_cu_k = [], [], [0], [0]
    for g in groups:
        fresh = g.get("prefix") is not None
        f = g["f"]
        key = g["key"]
        start = g.get("start", 0)
        paged = arena.is_resident(key)
        row0 = token_count
        sufs = g["suffixes"]
        if not isinstance(sufs, Suffixes):
            sufs = Suffixes.of(sufs)
        n = len(sufs)
        # a borrowing fresh group under tree attention reads the pages
        # before start in call B. With one suffix and a read_key the
        # group is one causal segment that reads the parent, stacked
        # with its siblings; otherwise the prefix is its own segment
        # reading the key's borrowed pages
        borrowing = fresh and start and tree_path
        stacked = borrowing and n == 1 and g.get("read_key") is not None
        whole = (fresh and paged and tree_path and not start and n == 1
                 and bool(g.get("single")))
        prefix_len = 0
        if fresh:
            if not paged and n > 1:
                raise ValueError(
                    f"group {key!r}: an unpaged fresh group packs "
                    f"[prefix | suffix] as one causal segment, which "
                    f"only one suffix may join - allocate pages or "
                    f"split the group")
            if start and not paged:
                raise ValueError(
                    f"group {key!r}: a borrowed prefix needs arena pages")
            prefix_len = len(g["prefix"])
            id_parts.append(g["prefix"])
            token_count += prefix_len
            pos.append(np.arange(start, start + prefix_len, dtype=np.int64))
            if (paged or not n) and not stacked and not whole:
                cu_a.append(np.array([token_count], dtype=np.int64))
            if borrowing and not stacked:
                reader_rows.append(np.arange(row0, token_count, dtype=np.int64))
                cu_q.append(cu_q[-1] + prefix_len)
                read_keys.append(key)
                read_used.append(start)
                max_q = max(max_q, prefix_len)
            if paged and tree_path:
                kv_writes.append((key, row0, token_count, start))
                fresh_keys.append(key)
        prefix_end = token_count
        s_row0 = token_count
        # with a canvas, read_all_rows reads every canvas row instead
        # of the suffix's own rows
        read_all = bool(g.get("read_all_rows"))
        own = g.get("canvas")
        group_canvas = (canvas_ids if own is None
                        else np.asarray(own, dtype=np.int64))
        width = len(group_canvas)
        if width and not read_all and answer_row >= width:
            raise ValueError(
                f"group {key!r}: answer_row {answer_row} is outside its "
                f"{width}-row canvas")
        # every suffix's rows at once: positions restart at f for each
        # suffix, and a canvas continues its suffix's positions
        widths = sufs.lengths + width
        total = int(widths.sum())
        ends = np.cumsum(widths)
        begins = ends - widths
        within = np.arange(total, dtype=np.int64) - np.repeat(begins, widths)
        if n and width:
            # row answer_row of each canvas carries the answer
            past = within - np.repeat(sufs.lengths, widths)
            source = np.where(
                past >= 0, len(sufs.ids) + past,
                np.repeat(sufs.offsets[:-1], widths) + within)
            id_parts.append(np.concatenate([sufs.ids, group_canvas])[source])
            first = s_row0 + begins + sufs.lengths
            canvas_starts.append(first)
            canvas_sizes.append(np.full(n, width, dtype=np.int64))
            if read_all:
                finals.append(np.repeat(first, width)
                              + np.tile(np.arange(width, dtype=np.int64), n))
                rows_per_answer.append(np.full(n, width, dtype=np.int64))
                multi_row = True
            else:
                finals.append(first + answer_row)
                rows_per_answer.append(np.ones(n, dtype=np.int64))
        elif n and g.get("read_rows") is not None:
            # the last read_rows rows of each suffix carry the answers
            id_parts.append(sufs.ids)
            counts = np.asarray(g["read_rows"], dtype=np.int64)
            if len(counts) != n or (counts > sufs.lengths).any():
                raise ValueError(
                    f"group {key!r}: read_rows must give each suffix at most "
                    f"its length")
            finals.append(np.repeat(s_row0 + ends, counts)
                          - np.repeat(counts, counts)
                          + (np.arange(int(counts.sum()), dtype=np.int64)
                             - np.repeat(np.cumsum(counts) - counts, counts)))
            rows_per_answer.append(counts)
            multi_row = True
        elif n:
            id_parts.append(sufs.ids)
            if read_all:
                finals.append(np.arange(s_row0, s_row0 + total, dtype=np.int64))
                rows_per_answer.append(sufs.lengths)
                multi_row = True
            else:
                finals.append(s_row0 + ends - 1)
                rows_per_answer.append(np.ones(n, dtype=np.int64))
        chains = g.get("chains")
        if chains is not None:
            if n != 1 or not read_all or not tree_path:
                raise ValueError("chains pack one suffix, every row read, "
                                 "under tree attention")
            lengths = np.array([len(nodes) for nodes, _, _ in chains])
            starts = np.array([start for _, start, _ in chains])
            chain_ends = s_row0 + np.cumsum(lengths)
            chain_row0 = chain_ends - lengths
            pos.append(f + np.repeat(starts, lengths)
                       + (np.arange(total, dtype=np.int64)
                          - np.repeat(chain_row0 - s_row0, lengths)))
            cu_a.append(chain_ends)
            # a chain's rows read the rows above its first node, which
            # earlier chains hold, in call C
            for index, (_, _, gathers) in enumerate(chains):
                if not gathers:
                    continue
                keys = np.concatenate([
                    np.arange(chain_row0[c], chain_row0[c] + count,
                              dtype=np.int64) for c, count in gathers])
                node_q.append(np.arange(chain_row0[index], chain_ends[index],
                                        dtype=np.int64))
                node_k.append(keys)
                node_cu_q.append(node_cu_q[-1] + int(lengths[index]))
                node_cu_k.append(node_cu_k[-1] + len(keys))
            token_count += total
        elif n:
            pos.append(f + within)
            cu_a.append(s_row0 + ends)
            token_count += total
            wst = g.get("write_suffix_tokens", 0)
            if wst and paged and tree_path:
                # the shared question preamble joins the kept KV right
                # after the document rows: after the fresh prefix, or
                # after a kept document's f rows
                dest = start + prefix_len if fresh else f
                kv_writes.append((key, s_row0, s_row0 + wst, dest))
        s_count = token_count - s_row0
        layout.append((key, n))
        if stacked:
            # every row of the group reads the parent's pages up to
            # start; siblings packed back to back share that read
            count = token_count - row0
            reader_rows.append(np.arange(row0, token_count, dtype=np.int64))
            read_key = g["read_key"]
            if read_keys and read_keys[-1] == read_key \
                    and read_used[-1] == start:
                cu_q[-1] += count
                max_q = max(max_q, cu_q[-1] - cu_q[-2])
            else:
                cu_q.append(cu_q[-1] + count)
                read_keys.append(read_key)
                read_used.append(start)
                max_q = max(max_q, count)
        elif s_count and f and paged and tree_path and not whole:
            reader_rows.append(np.arange(s_row0, token_count, dtype=np.int64))
            cu_q.append(cu_q[-1] + s_count)
            read_keys.append(key)
            read_used.append(f)
            max_q = max(max_q, s_count)
        if attention_mode == "unified":
            if paged:
                unified_groups.append(dict(
                    key=key, fresh=fresh, f=f, row0=row0, start=start,
                    prefix_end=prefix_end, row1=token_count,
                    suffix_spans=(s_row0 + begins, s_row0 + ends),
                    canvas=bool(n and width)))
            elif not fresh:
                raise ValueError(
                    "unified attention requires pages for a kept "
                    "group - an unpaged kept group has no KV to read")
    cu_a = np.concatenate(cu_a)

    t = _tick(timing, "pack_py", t)
    reads = None
    if read_keys:
        table, _ = arena.block_table(read_keys)
        t = _tick(timing, "pack_blocktable", t)
        rows = np.concatenate(reader_rows)
        # row -> its index in call B's output, -1 for rows that read
        # nothing resident; the fused merge kernel's map
        source = np.full(token_count, -1, dtype=np.int32)
        source[rows] = np.arange(len(rows), dtype=np.int32)
        reads = dict(
            rows=_staged(torch, rows, torch.int64, pinned),
            cu_q=stage("unified_cu_q", cu_q, torch.int32),
            max_q=max_q, used=_staged(torch, read_used, torch.int32, pinned),
            max_used=max(read_used), table=table,
            source=_staged(torch, source, torch.int32, pinned))
        if node_q:
            q_rows = np.concatenate(node_q)
            reads["nodes"] = dict(
                rows=_staged(torch, q_rows, torch.int64, pinned),
                key_rows=_staged(torch, np.concatenate(node_k), torch.int64,
                                 pinned),
                cu_q=_staged(torch, node_cu_q, torch.int32, pinned),
                cu_k=_staged(torch, node_cu_k, torch.int32, pinned),
                max_q=int(np.diff(node_cu_q).max()),
                max_k=int(np.diff(node_cu_k).max()),
                # each node row's index among call B's rows
                b_index=_staged(torch, source[q_rows], torch.int64, pinned))
    elif node_q:
        raise ValueError("chain rows read their document in call B; "
                         "the document needs arena pages")
    t = _tick(timing, "pack_reads", t)

    # all of the chunk's KV writes as one list of (source, destination)
    # row pairs: the attention pass scatters them with a single kernel
    # launch per layer (kv_row_scatter)
    unified = None
    temporary_keys = []
    if attention_mode == "unified" and unified_groups:
        if len(unified_groups) != len(layout):
            raise ValueError(
                "a unified chunk cannot mix paged and unpaged "
                "groups: the one paged call covers every row or "
                "none (the fast path packs whole chunks unpaged)")
        sliding = arena.has_sliding
        # one builder per pool: the every-token pool, and the
        # sliding pool that holds each key's rows from its window
        # origin on
        pools = [_PoolBuilder()] + ([_PoolBuilder()] if sliding else [])
        cu_q = [0]
        page_tokens = arena.page_tokens

        try:
            for spec in unified_groups:
                key = spec["key"]
                f = spec["f"]
                r0, r1 = spec["row0"], spec["row1"]
                begins, ends = spec["suffix_spans"]
                logical_start = spec["start"] if spec["fresh"] else f
                if spec["fresh"]:
                    fresh_keys.append(key)
                count = r1 - r0
                views = [_PoolView(
                    arena.capacity_rows(key), arena.owned_pages(key), 0)]
                if sliding:
                    views.append(_PoolView(
                        arena.capacity_rows_sliding(key),
                        arena.owned_sliding_pages(key),
                        arena.sliding_start(key)))
                direct = len(begins) == 1 and all(
                    logical_start + count - view.start <= view.rows.numel()
                    for view in views)
                suffix_total = int((ends - begins).sum())
                if direct:
                    for pool, view in zip(pools, views):
                        pool.scatter_direct(view, r0, r1, logical_start)
                        pool.add_sequence(view.pages, f + suffix_total - view.start)
                    cu_q.append(cu_q[-1] + count)
                    if spec["canvas"]:
                        canvas_seq.append(len(cu_q) - 2)
                    continue

                prefix_count = spec["prefix_end"] - r0
                if prefix_count:
                    for pool, view in zip(pools, views):
                        pool.scatter_direct(view, r0, spec["prefix_end"],
                                            spec["start"])
                        pool.add_sequence(
                            view.pages[:arena.pages_needed(f - view.start)],
                            f - view.start)
                    cu_q.append(cu_q[-1] + prefix_count)

                for s0, s1 in zip(begins.tolist(), ends.tolist()):
                    suffix_tokens = s1 - s0
                    remainders = [(f - view.start) % page_tokens for view in views]
                    got = arena.alloc_temporary(
                        remainders[0] + suffix_tokens,
                        sliding_tokens=(remainders[1] + suffix_tokens
                                        if sliding else None))
                    if got is None:
                        sliding_free = (arena.sliding.free_pages
                                        if sliding else None)
                        raise ArenaFullError(
                            "unified suffix pages exceed the free KV arena; "
                            f"split the chunk (asked {remainders[0] + suffix_tokens}"
                            f" rows; free pages {arena.accounting.free_pages}"
                            f" every-token, {sliding_free} sliding; retained"
                            f" {arena.retained_pages}; groups {len(groups)})")
                    temp_key, _ = got
                    temporary_keys.append(temp_key)
                    temp_views = [_PoolView(
                        arena.capacity_rows(temp_key),
                        arena.owned_pages(temp_key), 0)]
                    if sliding:
                        temp_views.append(_PoolView(
                            arena.capacity_rows_sliding(temp_key),
                            arena.owned_sliding_pages(temp_key), 0))
                    for pool, view, temp, remainder in zip(
                            pools, views, temp_views, remainders):
                        pool.scatter_suffix(view, temp, s0, s1, f, remainder,
                                            page_tokens)
                        kept_pages = (f - view.start) // page_tokens
                        pool.add_sequence(
                            view.pages[:kept_pages] + temp.pages,
                            f - view.start + suffix_tokens)
                    cu_q.append(cu_q[-1] + suffix_tokens)
                    if spec["canvas"]:
                        canvas_seq.append(len(cu_q) - 2)

            built = [pool.build(arena, stage, torch, staging, index)
                     for index, pool in enumerate(pools)]
            unified = dict(
                built[0],
                cu_q=stage("unified_cu_q", cu_q, torch.int32),
                max_q=max(b - a for a, b in zip(cu_q, cu_q[1:])))
            if sliding:
                unified["sliding"] = built[1]
        except Exception:
            for key in temporary_keys:
                arena.free_key(key)
            raise

    # All current KV writes use one scatter.
    kv_src = kv_dst = None
    if kv_writes:
        src, dst_parts = [], []
        for key, r0, r1, dest in kv_writes:
            src.append(np.arange(r0, r1, dtype=np.int64))
            rows = arena.capacity_rows(key)[dest:dest + (r1 - r0)]
            if rows.numel() != r1 - r0:
                raise AssertionError("KV write exceeds reserved rows")
            dst_parts.append(rows)
        kv_src = _staged(torch, np.concatenate(src), torch.int64, pinned)
        kv_dst = _staged(torch, torch.cat(dst_parts), torch.int64,
                         pinned)
    t = _tick(timing, "pack_kv", t)

    canvas_meta = None
    if canvas_starts:
        starts = np.concatenate(canvas_starts)
        if (attention_mode == "unified" and unified is None
                and len(starts) != len(cu_a) - 1):
            raise ValueError(
                "the unpaged canvas call pairs one canvas with each "
                "causal segment; a group without a suffix has none")
        widths = np.concatenate(canvas_sizes)
        cum = np.cumsum(widths)
        canvas_meta = dict(
            rows=stage("canvas_rows", np.repeat(starts, widths)
                       + np.arange(int(cum[-1]), dtype=np.int64)
                       - np.repeat(cum - widths, widths), torch.int64),
            cu_q=stage("canvas_cu_q", np.concatenate([[0], cum]),
                       torch.int32),
            # the readouts slice canvases by their row offsets on the host
            cu_q_host=np.concatenate([[0], cum]),
            max_q=int(widths.max()))
        if unified is not None:
            seq = stage("canvas_seq", canvas_seq, torch.int64)
            canvas_meta["table"] = unified["table"].index_select(0, seq)
            canvas_meta["used"] = unified["used"].index_select(0, seq)
            canvas_meta["max_used"] = max(
                pools[0].lengths[i] for i in canvas_seq)
            if "sliding" in unified:
                canvas_meta["sliding"] = dict(
                    table=unified["sliding"]["table"].index_select(0, seq),
                    used=unified["sliding"]["used"].index_select(0, seq),
                    max_used=max(pools[1].lengths[i] for i in canvas_seq))
    meta = dict(
        layer=0, kv_src=kv_src, kv_dst=kv_dst, reads=reads,
        unified=unified, canvas=canvas_meta,
        cu_a=stage("cu_a", cu_a, torch.int32),
        max_a=int(np.diff(cu_a).max()) if len(cu_a) > 1 else 0)
    out = Chunk(
        input_ids=_staged_token_parts(
            torch, id_parts, token_count, pinned, staging),
        positions=stage("positions", concatenate(pos), torch.int64),
        final_indices=stage("finals", concatenate(finals), torch.int64),
        meta=meta, attention_mode=attention_mode, tokens=token_count,
        layout=layout, temporary_keys=tuple(temporary_keys),
        fresh_keys=tuple(fresh_keys),
        rows_per_answer=(tuple(np.concatenate(rows_per_answer).tolist())
                         if multi_row else ()))
    _tick(timing, "pack_h2d", t)
    return out


def attention_path(pipeline, requested, default="unified") -> str:
    """The path a filter stream or join runs: the plan's request where the model has it.

    None asks for the default. A model without tree attention, and a
    canvas model (whose rows the tree path does not pack), run unified.
    """
    if requested is None:
        requested = default
    if requested == "tree" and pipeline.tree_attention and not pipeline.canvas_ids:
        return "tree"
    return "unified"


def _forward(pipeline, arena, chunk):
    """Run the forward pass, then release the chunk's temporary pages.

    The loop owns every page it hands the forward pass, so it frees the
    temporaries whether the pass returns or raises.
    """
    try:
        return pipeline.forward_chunk(chunk)
    finally:
        keys, chunk.temporary_keys = chunk.temporary_keys, ()
        for key in keys:
            arena.free_key(key)


# ------------------------------------------------------------ the join

def run_join(torch, arena, pipeline, async_ans, anchor_prefixes,
             stage_suffixes, budget, stage_frames=None,
             anchor_keys=None, anchor_done=None,
             anchor_partners=None, staging=None,
             attention_mode=None, prefix_tree=None, stats=None,
             read_all_rows=False, advance=None):
    """The join driver: stream partner lists against anchors.

    Survivors are gated between stages.

    Args:
        torch: The torch module, imported by the caller.
        arena: KVArena holding the anchors' KV pages.
        pipeline: ModelPipeline that runs each packed forward chunk.
        async_ans: Readout that turns final hidden states into answer
            rows: AsyncAnswers for TRUE/FALSE bits, AsyncScores for
            numeric scores. Its dtype picks the answer array type.
        anchor_prefixes: Anchor id (list index) -> prefix token list.
        stage_suffixes: Per stage, the partner suffix token lists.
        budget: Chunk token budget.
        stage_frames: Per stage, task framing token list written into
            each anchor's kept KV after the document rows.
        anchor_keys: Stable arena key for each anchor. List positions are
            used when omitted.
        anchor_done: Optional callback(anchor position, final answer row).
            It owns the anchor's final retain or free decision.
        anchor_partners: Optional callable(anchor key) -> per stage,
            the indices into that stage's partner list the anchor
            streams, or None for the whole list. Omitted means every
            anchor streams every partner.
        staging: Optional reusable input transfer buffers.
        attention_mode: "tree" or "unified" as the plan chose; None
            and a model without tree attention run unified.
        prefix_tree: A PrefixTree over anchor_prefixes, or None. A fresh
            anchor whose parent's pages are resident borrows them for
            its shared length and packs only the tokens after it.
        stats: When given, receives borrowed_tokens: the anchor prefix
            tokens read from a parent's KV pages instead of computed.
        read_all_rows: Feed every row of each partner suffix to the
            readout, not only its last. The readout's submit then takes
            rows_per_answer and returns one fixed-size record per
            partner; a frame entry still reads one row.
        advance: Optional callable(anchor, stage, row) -> bool deciding
            from an anchor's complete row whether it goes on to the
            next stage; None advances on any true answer.

    Returns:
        (ans, spans, tokens): ans[j][a] = 0/1 row over the stage-j
        partners anchor a streams, with a in admission order; spans =
        (stage, start_event, end_event) per forward; tokens = fresh
        tokens packed.
    """
    from quail.backends.quail.executor.stages import Stage, run_stages

    k = len(stage_suffixes)
    frames = stage_frames or [[] for _ in range(k)]
    scoring = async_ans.dtype is not None
    stages = [
        Stage(
            suffixes=stage_suffixes[j], readout=async_ans, frame=frames[j],
            requests=(None if anchor_partners is None
                      else (lambda key, j=j: anchor_partners(key)[j])),
            decide=(None if advance is None
                    else (lambda a, row, j=j: advance(a, j, row))),
            read_all_rows=read_all_rows)
        for j in range(k)
    ]
    on_settled = None
    if anchor_done is not None:
        def on_settled(anchor, survived, row):
            anchor_done(anchor, row)
    return run_stages(
        torch, arena, pipeline, stages, anchor_prefixes, budget,
        anchor_keys=anchor_keys, on_settled=on_settled,
        staging=staging, attention_mode=attention_mode,
        prefix_tree=prefix_tree, stats=stats,
        unit="scores" if scoring else "anchors",
        label="AI.SCORE" if scoring else f"join ({k} stages)")


# ------------------------------------------------------------- warmup
#
# Three cost tiers:
#   1. JIT compile (nvcc/Triton) of a kernel configuration: paid once
#      ever per (software stack, GPU, model, budget) by compile_kernels.
#      Cached in the library's configured kernel directories;
#      a marker file records that the pass ran.
#   2. Loading a cached binary into the process: paid once per process
#      by touch_kernels.
#   3. The launch itself: every call, unavoidable.
#
# Kernels key on token counts and model constants, never on token
# values, so both passes run on synthetic ids.

# Bump when either pass covers a different set of shapes. A bumped
# version invalidates every marker, so the next boot re-runs the
# compile pass and re-commits the cache.
WARMUP_VERSION = 5


def _warm_inputs(budget):
    """Synthetic warmup tokens: a 512-id document and a 16-id question suffix.

    Cycled to any length the passes need. Fixed small ids; only the
    counts matter to the kernels.
    """
    doc = [10 + (i % 500) for i in range(512)]
    question = list(range(10, 26))
    q_max = len(question)
    warm_docs, used = [], 0
    while used + len(doc) + q_max <= budget:
        warm_docs.append(doc)
        used += len(doc) + q_max
    return warm_docs or [doc[:max(8, budget - q_max)]], question, doc


def _forward_warm(torch, arena, pipeline, async_ans, budget, *,
                  join_chunk):
    """Run real forward passes over every attention path.

    Both modes with arena writes, plus the unpaged causal fast path.
    join_chunk adds a join chunk and a classification chunk.
    """
    warm_docs, question, doc = _warm_inputs(budget)
    q_max = len(question)
    modes = ("unified", "tree") if pipeline.tree_attention else ("unified",)
    for mode in modes:
        logger.debug("kernels: warming %s attention, full chunk", mode)
        run_filter(torch, arena, pipeline, async_ans, warm_docs,
                   [question], budget, arena_writes=True,
                   attention_mode=mode)
        # small chunks, one document each: the trailing-chunk shapes
        # of gated multi-stage runs (see ModelPipeline.warm_tokens)
        for t in pipeline.warm_tokens:
            if t >= budget:
                continue
            logger.debug("kernels: warming %s attention, %s tokens", mode, t)
            body = (doc * (t // len(doc) + 1))[:max(8, t - q_max)]
            run_filter(torch, arena, pipeline, async_ans, [body],
                       [question], budget, arena_writes=True,
                       attention_mode=mode)
    if join_chunk:
        logger.debug("kernels: warming join forward pass")
        run_join(torch, arena, pipeline, async_ans, warm_docs,
                 [[question] * 8], budget)
        # a classification: the question as the frame after each
        # document, then many short suffixes of mixed lengths
        logger.debug("kernels: warming classification forward pass")
        labels = [question[:1 + i % 6] for i in range(26)]
        run_join(torch, arena, pipeline, async_ans, warm_docs,
                 [labels], budget, stage_frames=[question])
    logger.debug("kernels: warming filter without KV writes")
    run_filter(torch, arena, pipeline, async_ans, warm_docs,
               [question], budget, arena_writes=False)


def compile_kernels(torch, arena, pipeline, async_ans, budget):
    """Build every DeepGEMM kernel configuration, then warm every path.

    Configurations go up to the budget; the warm pass runs forward
    passes over every attention-path shape.

    Runs once per (software stack, GPU, model, budget). Uses vLLM's
    config heuristic generator to enumerate every token count at
    which the chosen GEMM configuration changes.
    """
    if not pipeline.gemm_warmup:
        _forward_warm(torch, arena, pipeline, async_ans, budget,
                      join_chunk=True)
        return
    from vllm.model_executor.warmup.deep_gemm_warmup import (
        _generate_optimal_warmup_m_values,
    )
    linears = pipeline.linears()
    work = [(m, lin) for lin in linears
            for m in _generate_optimal_warmup_m_values(
                budget, lin.weight.shape[0], torch.device("cuda"))]
    progress = Progress("kernels: GEMM warmup", total=len(work),
                        unit="configurations", emit=logger.info)
    logger.info("kernels: starting %s GEMM warmup configurations", len(work))
    with torch.inference_mode():
        # one buffer per linear, row-sliced per call; the sweep only
        # needs each kernel launched once
        cur, buf = None, None
        for done, (m, lin) in enumerate(work, 1):
            if lin is not cur:
                buf = torch.randn(budget, lin.weight.shape[1],
                                  device="cuda",
                                  dtype=torch.bfloat16)
                cur = lin
            q, s = pipeline.engine.quant(buf[:m])
            pipeline.engine.gemm(q, s, lin)
            progress.update(done)
        buf = None
        torch.cuda.synchronize()
    progress.finish("kernels: GEMM warmup done")
    logger.info("kernels: warming filter and join forward passes")
    _forward_warm(torch, arena, pipeline, async_ans, budget,
                  join_chunk=True)


def touch_kernels(torch, arena, pipeline, async_ans, budget):
    """Run each hot kernel once per GPU process.

    Cached binaries then load at boot instead of mid-run.
    """
    _forward_warm(torch, arena, pipeline, async_ans, budget,
                  join_chunk=True)


def _marker_path(model_name, budget):
    import os
    root = os.path.expanduser(os.environ.get(
        "QUAIL_CACHE_DIR", "~/.cache/quail/kernels"))
    safe = model_name.replace("/", "--")
    return os.path.join(root, f"quail-warm-{safe}-{int(budget)}.json")


def _marker_identity(torch, model_name, budget):
    try:
        import vllm
        vllm_version = vllm.__version__
    except ImportError:
        vllm_version = None
    return dict(warmup_version=WARMUP_VERSION, model=model_name,
                budget=int(budget), vllm=vllm_version,
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name())


def warm_kernels(torch, arena, pipeline, async_ans, budget, *,
                 model_name, force_compile=False):
    """Compile once under a file lock, then warm each GPU process.

    Returns:
        A dict with the selected tier and elapsed warmup seconds.
    """
    import fcntl
    import json
    import os

    with quiet():
        path = _marker_path(model_name, budget)
        identity = _marker_identity(torch, model_name, budget)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t0 = time.perf_counter()
        tier = "touch"
        # The marker check, compilation, and publication must be one
        # operation across processes sharing the kernel directory.
        with open(path + ".lock", "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.info("kernels: waiting for another worker to compile")
                fcntl.flock(lock, fcntl.LOCK_EX)
            wait_s = time.perf_counter() - t0
            on_disk = None
            try:
                with open(path) as f:
                    on_disk = json.load(f)
            except (OSError, ValueError):
                pass
            if force_compile or on_disk != identity:
                logger.info("kernels: starting compile pass")
                compile_kernels(torch, arena, pipeline, async_ans, budget)
                torch.cuda.synchronize()
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(identity, f, indent=1)
                os.replace(tmp, path)
                tier = "compile"
        if tier == "touch":
            logger.info("kernels: warming cached filter and join kernels")
            touch_kernels(torch, arena, pipeline, async_ans, budget)
        torch.cuda.synchronize()
        warm_s = round(time.perf_counter() - t0 - wait_s, 2)
        wait_s = round(wait_s, 2)
        logger.info("kernels: %s pass done in %s s; waited %s s for compilation",
                    tier, warm_s, wait_s)
        return dict(tier=tier, warm_s=warm_s, wait_s=wait_s)


# ---------------------------------------------------------- the filter

def borrow_check(arena, keys, borrowing):
    """can_borrow for Borrowing: whether the arena can serve a borrow.

    A fresh parent admitted in the same chunk is not allocated yet;
    its sliding pages will start at the window origin below the tokens
    it borrows itself. A parent already resident, admitted in this
    chunk or earlier, has the pages it has: it may have trimmed its
    window in an earlier query.
    """
    def can_borrow(doc, parent, same_chunk):
        shared = borrowing.tree_shared[doc]
        if not same_chunk or arena.is_resident(keys[parent]):
            return arena.can_borrow(keys[parent], shared)
        return arena.origin(borrowing.shared(parent)) <= arena.origin(shared)
    return can_borrow


def lowest_borrows(borrowing):
    """Per document, the fewest tokens any of its borrowers shares."""
    lowest = {}
    for doc, parent in enumerate(borrowing.tree_parent):
        if parent is not None:
            shared = borrowing.tree_shared[doc]
            lowest[parent] = min(lowest.get(parent, shared), shared)
    return lowest


def run_filter(torch, arena, pipeline, async_ans, doc_ids,
               question_ids, budget, timing=None,
               pinned=True, limit=None, *, arena_writes,
               arena_keys=None, retain_survivors=(), attention_mode=None,
               document_done=None, prefix_tree=None, stats=None,
               staging=None):
    """The filter chain run to the end on the stage scheduler.

    Args:
        torch: The torch module, imported by the caller.
        arena: KVArena holding the documents' KV pages.
        pipeline: ModelPipeline that runs each packed forward chunk.
        async_ans: AsyncAnswers readout for TRUE/FALSE bits.
        doc_ids: Per-document token lists.
        question_ids: Per-stage question token lists.
        budget: Chunk token budget.
        timing: Unused; kept for callers that pass it.
        pinned: Unused; kept for callers that pass it.
        limit: Stop admitting documents after this many survivors.
        arena_writes: Whether document KV is written to the arena.
            Must be True with multiple stages.
        arena_keys: Stable arena key for each document. List positions
            are used when omitted.
        retain_survivors: Passing document positions to keep for
            joins, or True for all of them.
        attention_mode: "tree" or "unified" as the plan chose; None
            and a model without tree attention run unified.
        document_done: Called after each chunk's answers are read with
            the documents that finished in it, as (position, last
            stage asked, passed) tuples.
        prefix_tree: A PrefixTree over doc_ids, or None. Needs
            arena_writes.
        stats: When given, receives borrowed_tokens and pack_s.
        staging: Optional reusable input transfer buffers.

    Returns:
        (answers, spans, tokens): answers[d] = 0/1 list up to the
        first FALSE; spans and tokens as in run_stages.
    """
    from quail.backends.quail.executor.stages import filter_stages, run_stages

    stages = filter_stages(question_ids, async_ans)
    k = len(stages)
    keys = list(range(len(doc_ids))) if arena_keys is None else arena_keys
    if len(keys) != len(doc_ids):
        raise ValueError("arena_keys must match doc_ids")
    retain_all = retain_survivors is True
    retain = set() if retain_all else set(retain_survivors)
    # a model whose attention reads paged KV only writes pages even
    # when the plan skipped them
    arena_writes = arena_writes or pipeline.needs_pages
    if not arena_writes and (k > 1 or retain_all or retain):
        # a later stage re-reads the KV, which needs the pages this
        # switch skips
        raise ValueError("arena_writes=False needs a single stage")
    if prefix_tree is not None and prefix_tree.shared_tokens:
        if not arena_writes:
            raise ValueError("a prefix tree needs arena writes")
    else:
        prefix_tree = None
    if arena.retention_cap_pages is None:
        arena.retention_cap_pages = max(
            0, arena.n_pages - arena.pages_needed(2 * budget))

    def on_settled(anchor, survived, row):
        key = keys[anchor]
        if survived and (retain_all or anchor in retain):
            arena.retain(key, len(doc_ids[anchor]))
        elif arena.is_resident(key):
            arena.free_key(key)

    def on_chunk(transitions):
        finished = [(anchor, stage, passed)
                    for anchor, stage, passed in transitions
                    if not passed or stage == k - 1]
        if finished:
            document_done(finished)

    answers, spans, tokens = run_stages(
        torch, arena, pipeline, stages, doc_ids, budget,
        anchor_keys=keys, on_settled=on_settled,
        attention_mode=attention_mode, prefix_tree=prefix_tree,
        stats=stats, limit=limit, paged=arena_writes, staging=staging,
        label=f"filter ({k} stages)", default_attention="unified",
        on_chunk=on_chunk if document_done is not None else None)
    by_document = {}
    for stage in answers:
        for doc, row in stage.items():
            by_document.setdefault(doc, []).append(int(row[0]))
    return by_document, spans, tokens
