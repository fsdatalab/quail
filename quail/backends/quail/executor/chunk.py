"""Build one chunk's tensors and stage its inputs for the forward pass.

The admission scheduler in pack.py selects the groups. This module lays out
those groups for attention, KV writes, and answer readouts. Torch is supplied
by the caller so importing the executor does not load it.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from quail.backends.quail.executor.attention import ATTENTION_PATHS


@dataclass
class Chunk:
    """One packed forward pass: token rows plus attention bookkeeping.

    Attributes:
        input_ids: Token ids, one per packed row, on the GPU.
        positions: Rotary position of each row.
        final_indices: Rows whose hidden state feeds the answer readout.
        meta: Attention-path bookkeeping built by pack_chunk: the layer
            counter, KV scatter maps, block tables, and sequence bounds.
        attention_mode: "unified" or "tree", the path the packer laid
            the chunk out for.
        tokens: Rows in the chunk.
        layout: (arena key, suffix count) per group in chunk order.
        temporary_keys: Arena keys the loop frees after the forward pass.
        fresh_keys: Keys whose prefix this chunk computes; the loop
            trims their sliding pages after the pass.
        rows_per_answer: Rows of final_indices each answer takes, one
            entry per suffix in chunk order, when some suffix reads
            more than its last row; empty when every answer is one row.
    """

    input_ids: Any
    positions: Any
    final_indices: Any
    meta: dict
    attention_mode: str
    tokens: int
    layout: list
    temporary_keys: tuple = ()
    fresh_keys: tuple = ()
    rows_per_answer: tuple = ()


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
        # a two-dimensional canvas holds one canvas per suffix
        per_suffix = group_canvas.ndim == 2
        if per_suffix and len(group_canvas) != n:
            raise ValueError(
                f"group {key!r}: {len(group_canvas)} canvases for {n} suffixes")
        width = group_canvas.shape[-1]
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
            # suffix s reads canvas row s of a two-dimensional canvas
            first_row = (np.repeat(np.arange(n) * width, widths) if per_suffix
                         else 0)
            source = np.where(
                past >= 0, len(sufs.ids) + first_row + past,
                np.repeat(sufs.offsets[:-1], widths) + within)
            id_parts.append(
                np.concatenate([sufs.ids, group_canvas.ravel()])[source])
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


