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
from quail.backends.quail.executor.gdn import plan_phases


@dataclass
class Chunk:
    """Token tensors and attention metadata for one packed forward pass.

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
        row_spans: (arena key, first row, first suffix row, end row) per
            group in chunk order; a pipeline's forward hook reads them.
        temporary_keys: Arena keys the loop frees after the forward pass.
        fresh_keys: Keys whose prefix this chunk computes; the loop
            trims their sliding pages after the pass.
        rows_per_answer: Rows of final_indices each answer takes, one
            entry per suffix in chunk order, when some suffix reads
            more than its last row; None when every answer is one row.
        decode_rows: The DecodeRows packed after the groups, or None.
            Their rows come last in the chunk and in final_indices.
    """

    input_ids: Any
    positions: Any
    final_indices: Any
    meta: dict
    attention_mode: str
    tokens: int
    layout: list
    row_spans: tuple = ()
    temporary_keys: tuple = ()
    fresh_keys: tuple = ()
    rows_per_answer: np.ndarray | None = None
    decode_rows: Any = None


@dataclass
class DecodeRows:
    """Running answers' token and copy-draft rows after a chunk's groups.

    Each row feeds a chosen or drafted token, writes its KV into the
    answer's own pages, and reads those pages through its position in
    one causal paged attention call. Slots repeat for copy drafts.

    The branches of one document may read its shared pages together:
    then each such row reads only its own pages in the causal call,
    and its group's rows read the shared pages once, stacked, in a
    second call whose result is merged into theirs.

    Attributes:
        positions: Rotary position of each row, which is also where its
            KV is written in the answer's sequence.
        write_rows: Arena row each row's KV is written to.
        page_table: Per row, the answer's page ids in logical order,
            padded on the right.
        slots: Per row, where its input token sits in the token buffer.
        own_table: Per row, the pages its causal call reads: its pages
            past the shared ones, or all of them.
        own_lengths: Per row, the KV tokens its causal call reads.
        shared_rows: The rows that read shared pages, grouped, or an
            empty array.
        shared_cu: Cumulative row counts of the groups, from 0.
        shared_table: Per group, its shared pages.
        shared_used: Per group, its shared tokens.
    """

    positions: np.ndarray
    write_rows: np.ndarray
    page_table: np.ndarray
    slots: np.ndarray
    own_table: np.ndarray | None = None
    own_lengths: np.ndarray | None = None
    shared_rows: np.ndarray | None = None
    shared_cu: np.ndarray | None = None
    shared_table: np.ndarray | None = None
    shared_used: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def sequence_offsets(self) -> np.ndarray:
        """Return bounds of consecutive positions belonging to the same slot."""
        if not len(self):
            return np.zeros(1, dtype=np.int64)
        starts = np.flatnonzero(
            (self.slots[1:] != self.slots[:-1])
            | (self.positions[1:] != self.positions[:-1] + 1)
        ) + 1
        return np.r_[0, starts, len(self)]


class ArenaFullError(RuntimeError):
    """The KV arena has too few free pages to pack the chunk."""


class BorrowStateMissingError(ArenaFullError):
    """A borrower's parent holds no state at the length it borrows.

    The borrower can compute its whole prefix instead.
    """

    def __init__(self, key, message):
        super().__init__(message)
        self.key = key


def _tick(timing, key, t0):
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + time.perf_counter() - t0
    return time.perf_counter()


def _concatenate(parts):
    """Join int64 arrays, or give an empty one when there is nothing to join."""
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _staged(torch, data, dtype, pinned=True):
    """Copy host data to the device.

    With pinned=True the copy goes through pinned memory and does not
    block; with pinned=False it is a pageable, blocking copy.
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
    """Convert token values to an int64 array, avoiding copies when possible."""
    if hasattr(part, "arrow_array"):
        part = part.arrow_array.to_numpy(zero_copy_only=False)
    elif hasattr(part, "numpy") and not isinstance(part, np.ndarray):
        part = part.numpy()
    return np.asarray(part, dtype=np.int64)


class Suffixes:
    """Flattened suffix token sequences with their lengths and offsets.

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
        """Return suffix lengths at the given indices, in index order."""
        if isinstance(indices, range) and indices.step == 1:
            return self.lengths[indices.start:indices.stop]
        return self.lengths[np.asarray(indices, dtype=np.int64)]

    def take(self, indices):
        """Return suffix token sequences at the given indices, in index order."""
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


# the passes a staged input buffer goes unused before the host refills
# it; more than the passes the host builds ahead (map_decoder.IN_FLIGHT)
STAGING_DEPTH = 4


class InputStaging:
    """Reuse CPU and GPU input buffers on the current CUDA stream.

    Each name rotates through STAGING_DEPTH buffers. The CPU must not
    overwrite a pinned buffer while its upload is pending, so it waits on
    the buffer's last upload; with one buffer that upload belongs to the
    pass just queued, and the host could never build more than one pass
    ahead of the GPU. With a ring, the wait falls on an upload several
    passes back, which has landed unless the host is that far ahead.
    """

    def __init__(self, torch):
        self.torch = torch
        self.buffers = {}     # name -> the buffer in use: host, device, event
        self.rings = {}       # name -> (next slot, list of buffers or None)
        self.fixed_tokens = {}

    def host(self, name, count, dtype):
        torch = self.torch
        slot, ring = self.rings.get(name, (0, [None] * STAGING_DEPTH))
        previous = ring[slot]
        if previous is not None:
            host, device, event = previous
            # The CPU must not overwrite a buffer while its transfer is pending.
            event.synchronize()
        if previous is None or host.numel() < count or host.dtype != dtype:
            host = torch.empty(count, dtype=dtype, pin_memory=True)
            device = torch.empty(count, dtype=dtype, device="cuda")
            event = torch.cuda.Event()
        ring[slot] = self.buffers[name] = host, device, event
        self.rings[name] = (slot + 1) % STAGING_DEPTH, ring
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
    """Rows and pages owned by one KV key in one arena pool.

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
    """KV scatter indices, block tables, and sequence lengths for one pool."""

    def __init__(self):
        self.page_rows = []
        self.lengths = []
        self.src = []
        self.dst = []
        self.tail_src = []
        self.tail_dst = []
        # (page table, lengths) blocks of decode sequences, after the rows
        self.blocks = []

    def add_sequence(self, pages, kv_tokens):
        self.page_rows.append(pages)
        self.lengths.append(kv_tokens)

    def add_block(self, rows: DecodeRows, r0: int):
        """Add consecutive decode sequences, starting at input row r0."""
        offsets = rows.sequence_offsets
        first, last = offsets[:-1], offsets[1:] - 1
        if rows.own_table is None:
            self.blocks.append((rows.page_table[first], rows.positions[last] + 1))
        else:
            self.blocks.append((rows.own_table[first], rows.own_lengths[last]))
        self.src.append(np.arange(r0, r0 + len(rows), dtype=np.int64))
        self.dst.append(rows.write_rows)

    def scatter_direct(self, view, r0, r1, logical_start):
        """Map input rows to the key's KV pages at the given logical position.

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
        """Map suffix rows to temporary pages and copy any retained partial page."""
        suffix_tokens = s1 - s0
        self.src.append(np.arange(s0, s1, dtype=np.int64))
        self.dst.append(temp.rows[remainder:remainder + suffix_tokens].numpy())
        if remainder:
            anchor_page = view.pages[(f - view.start) // page_tokens]
            base = anchor_page * page_tokens
            self.tail_src.append(np.arange(base, base + remainder, dtype=np.int64))
            self.tail_dst.append(temp.rows[:remainder].numpy())

    def build(self, arena, stage, torch, staging, index):
        tag = "" if index == 0 else "_sliding"
        lengths = np.concatenate(
            [np.asarray(self.lengths, dtype=np.int64)]
            + [block_lengths for _, block_lengths in self.blocks])
        page_rows = self.page_rows + [
            row for table, _ in self.blocks for row in table]
        if staging is None and not self.blocks:
            table = arena.block_table_rows(self.page_rows)
        else:
            width = max([len(pages) for pages in self.page_rows]
                        + [table.shape[1] for table, _ in self.blocks])
            block_table = np.zeros((len(lengths), width), dtype=np.int32)
            for row, pages in enumerate(self.page_rows):
                block_table[row, :len(pages)] = pages
            row = len(self.page_rows)
            for table, _ in self.blocks:
                block_table[row:row + len(table), :table.shape[1]] = table
                row += len(table)
            table = stage(f"block_table{tag}", block_table, torch.int32).view(
                len(lengths), width)
        return dict(
            src=stage(f"unified_src{tag}", _concatenate(self.src), torch.int64),
            dst=stage(f"unified_dst{tag}", _concatenate(self.dst), torch.int64),
            tail_src=(stage(f"tail_src{tag}", _concatenate(self.tail_src),
                            torch.int64) if self.tail_src else None),
            tail_dst=(stage(f"tail_dst{tag}", _concatenate(self.tail_dst),
                            torch.int64) if self.tail_dst else None),
            used=stage(f"unified_lengths{tag}", lengths.astype(np.int32),
                       torch.int32),
            table=table, page_rows=page_rows, used_host=lengths.tolist(),
            max_used=int(lengths.max()))


def pack_chunk(torch, arena, groups, timing=None, pinned=True, *,
               attention_mode, staging=None, canvas=(), answer_row=0,
               decode=None):
    """Pack scheduled document groups into tensors for one forward pass.

    Groups remain in scheduler order. Each group contains key, prefix, f
    (retained context length), and suffixes. Optional group fields are:

    * start: First fresh token position, after any borrowed prefix KV.
    * read_key: Parent key read by a single-suffix tree-attention group.
    * write_suffix_tokens: Leading suffix tokens to retain in the key's KV.
    * read_all_rows: Whether every suffix row feeds the answer readout.
    * read_rows: Number of trailing rows to read from each suffix.
    * single: Whether the prefix and sole suffix form one request.
    * paths: Trie paths and ancestor rows from trie_paths(). The document
      KV must already be resident.
    * canvas: Per-group answer canvas tokens, overriding the default canvas.

    Fresh groups without arena pages must have at most one suffix. Unified
    attention cannot mix paged and unpaged groups. Canvas rows use unified
    attention and their KV is not retained.

    Args:
        torch: Torch module.
        arena: KV arena holding document pages.
        groups: Scheduled group mappings with the fields described above.
        timing: Optional mapping receiving elapsed packing times.
        pinned: Whether host-to-device copies use pinned memory.
        attention_mode: Attention path, either tree or unified.
        staging: Optional reusable host and device transfer buffers.
        canvas: Default diffusion answer canvas token IDs.
        answer_row: Canvas row used for the answer when not reading all rows.
        decode: DecodeRows to pack after the groups, or None. They run
            the unified path; their input ids are placeholders the
            caller fills on the GPU.

    Returns:
        A Chunk containing token tensors, attention metadata, and readout rows.

    Raises:
        ValueError: The attention path, canvas row, or group layout is invalid.
        ArenaFullError: The arena has too few free pages for suffixes.
    """
    def stage(name, values, dtype):
        if staging is not None:
            return staging.copy(name, values, dtype)
        return _staged(torch, values, dtype, pinned)

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
    id_parts, num_tokens = [], 0
    position, finals = [], []
    cu_a = [np.zeros(1, dtype=np.int64)]
    rows_per_answer = []
    multi_row = False
    # rows that read resident KV in call B: a borrowing document's
    # rows reading its parent, and a kept document's tail reading its
    # own prefix (wider than the planner's readers, which are children)
    reader_rows = []
    kv_writes, layout = [], []
    row_spans = []
    fresh_keys = []
    read_keys, read_used, cu_q = [], [], [0]
    max_q = 0
    unified_groups = []
    # call C: path rows reading their off-path ancestor rows
    node_q, node_k, node_cu_q, node_cu_k = [], [], [0], [0]
    for g in groups:
        fresh = g.get("prefix") is not None
        f = g["f"]
        key = g["key"]
        start = g.get("start", 0)
        paged = arena.is_resident(key)
        row0 = num_tokens
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
            num_tokens += prefix_len
            position.append(np.arange(start, start + prefix_len, dtype=np.int64))
            if (paged or not n) and not stacked and not whole:
                cu_a.append(np.array([num_tokens], dtype=np.int64))
            if borrowing and not stacked:
                reader_rows.append(np.arange(row0, num_tokens, dtype=np.int64))
                cu_q.append(cu_q[-1] + prefix_len)
                read_keys.append(key)
                read_used.append(start)
                max_q = max(max_q, prefix_len)
            if paged and tree_path:
                kv_writes.append((key, row0, num_tokens, start))
                fresh_keys.append(key)
        prefix_end = num_tokens
        s_row0 = num_tokens
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
        elif n and g.get("read_mask") is not None:
            # the marked rows of the one suffix carry the answers
            id_parts.append(sufs.ids)
            mask = np.asarray(g["read_mask"], dtype=bool)
            if n != 1 or len(mask) != total:
                raise ValueError(
                    f"group {key!r}: read_mask marks one suffix's rows")
            finals.append(s_row0 + np.flatnonzero(mask))
            rows_per_answer.append(np.array([int(mask.sum())], dtype=np.int64))
            multi_row = True
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
        paths = g.get("paths")
        if paths is not None:
            if n != 1 or not read_all or not tree_path:
                raise ValueError("paths pack one suffix, every row read, "
                                 "under tree attention")
            lengths = np.array([len(nodes) for nodes, _, _ in paths])
            starts = np.array([start for _, start, _ in paths])
            path_ends = s_row0 + np.cumsum(lengths)
            path_row0 = path_ends - lengths
            position.append(f + np.repeat(starts, lengths)
                       + (np.arange(total, dtype=np.int64)
                          - np.repeat(path_row0 - s_row0, lengths)))
            cu_a.append(path_ends)
            # a path's rows read the rows above its first node, which
            # earlier paths hold, in call C
            for index, (_, _, gathers) in enumerate(paths):
                if not gathers:
                    continue
                keys = np.concatenate([
                    np.arange(path_row0[c], path_row0[c] + count,
                              dtype=np.int64) for c, count in gathers])
                node_q.append(np.arange(path_row0[index], path_ends[index],
                                        dtype=np.int64))
                node_k.append(keys)
                node_cu_q.append(node_cu_q[-1] + int(lengths[index]))
                node_cu_k.append(node_cu_k[-1] + len(keys))
            num_tokens += total
        elif n:
            position.append(f + within)
            cu_a.append(s_row0 + ends)
            num_tokens += total
            wst = g.get("write_suffix_tokens", 0)
            if wst and paged and tree_path:
                # the shared question preamble joins the kept KV right
                # after the document rows: after the fresh prefix, or
                # after a kept document's f rows
                dest = start + prefix_len if fresh else f
                kv_writes.append((key, s_row0, s_row0 + wst, dest))
        num_s = num_tokens - s_row0
        layout.append((key, n))
        row_spans.append((key, row0, prefix_end, num_tokens))
        if stacked:
            # every row of the group reads the parent's pages up to
            # start; siblings packed back to back share that read
            count = num_tokens - row0
            reader_rows.append(np.arange(row0, num_tokens, dtype=np.int64))
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
        elif num_s and f and paged and tree_path and not whole:
            reader_rows.append(np.arange(s_row0, num_tokens, dtype=np.int64))
            cu_q.append(cu_q[-1] + num_s)
            read_keys.append(key)
            read_used.append(f)
            max_q = max(max_q, num_s)
        if attention_mode == "unified":
            if paged:
                unified_groups.append(dict(
                    key=key, fresh=fresh, f=f, row0=row0, start=start,
                    prefix_end=prefix_end, row1=num_tokens,
                    suffix_spans=(s_row0 + begins, s_row0 + ends),
                    canvas=bool(n and width),
                    keeps=bool(g.get("write_suffix_tokens")),
                    kept=int(g.get("write_suffix_tokens") or 0),
                    borrow_key=g.get("borrow_key"),
                    share_points=g.get("share_points", ())))
            elif not fresh:
                raise ValueError(
                    "unified attention requires pages for a kept "
                    "group - an unpaged kept group has no KV to read")
    decode_row0 = num_tokens
    if decode is not None and not len(decode):
        decode = None
    if decode is not None:
        if arena.has_sliding or getattr(arena, "num_state_slots", 0):
            raise ValueError("decode rows read a model with every-token "
                             "attention only")
        count = len(decode)
        decode_offsets = decode.sequence_offsets
        if tree_path:
            # A draft reads its resident prefix in B and its causal rows in A.
            reader_rows.append(np.arange(num_tokens, num_tokens + count,
                                         dtype=np.int64))
            cu_q.extend((cu_q[-1] + decode_offsets[1:]).tolist())
            read_used.extend(decode.positions[decode_offsets[:-1]].tolist())
            max_q = max(max_q, int(np.diff(decode_offsets).max()))
        id_parts.append(np.zeros(count, dtype=np.int64))
        position.append(np.asarray(decode.positions, dtype=np.int64))
        finals.append(np.arange(num_tokens, num_tokens + count, dtype=np.int64))
        rows_per_answer.append(np.ones(count, dtype=np.int64))
        cu_a.append(num_tokens + decode_offsets[1:])
        num_tokens += count
    cu_a = np.concatenate(cu_a)

    t = _tick(timing, "pack_py", t)
    reads = None
    tree_decode = decode is not None and tree_path
    if read_keys or tree_decode:
        if not tree_decode:
            table, _ = arena.block_table(read_keys)
        else:
            pages = [arena.owned_pages(key) for key in read_keys]
            width = max([len(row) for row in pages] + [decode.page_table.shape[1]])
            flat = np.zeros(
                (len(pages) + len(decode_offsets) - 1, width), dtype=np.int32)
            for row, own in enumerate(pages):
                flat[row, :len(own)] = own
            flat[len(pages):, :decode.page_table.shape[1]] = (
                decode.page_table[decode_offsets[:-1]])
            table = stage("read_table", flat, torch.int32).view(*flat.shape)
        t = _tick(timing, "pack_blocktable", t)
        rows = np.concatenate(reader_rows)
        # row -> its index in call B's output, -1 for rows that read
        # nothing resident; the fused merge kernel's map
        source = np.full(num_tokens, -1, dtype=np.int32)
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
        raise ValueError("path rows read their document in call B; "
                         "the document needs arena pages")
    t = _tick(timing, "pack_reads", t)

    unified = None
    temporary_keys = []
    if attention_mode == "unified" and (unified_groups or decode is not None):
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
                if spec["keeps"] and not direct:
                    raise ValueError(
                        f"group {key!r} keeps suffix KV but its pages hold "
                        f"{views[0].rows.numel()} rows, not "
                        f"{logical_start + count}")
                if direct:
                    for pool, view in zip(pools, views):
                        pool.scatter_direct(view, r0, r1, logical_start)
                        pool.add_sequence(view.pages, f + suffix_total - view.start)
                    cu_q.append(cu_q[-1] + count)
                    if spec["canvas"]:
                        canvas_seq.append(len(cu_q) - 2)
                    continue

                num_prefix_tokens = spec["prefix_end"] - r0
                if num_prefix_tokens:
                    for pool, view in zip(pools, views):
                        pool.scatter_direct(view, r0, spec["prefix_end"],
                                            spec["start"])
                        pool.add_sequence(
                            view.pages[:arena.pages_needed(f - view.start)],
                            f - view.start)
                    cu_q.append(cu_q[-1] + num_prefix_tokens)

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

            if decode is not None:
                pools[0].add_block(decode, decode_row0)
                cu_q.extend((cu_q[-1] + decode_offsets[1:]).tolist())
            built = [pool.build(arena, stage, torch, staging, index)
                     for index, pool in enumerate(pools)]
            unified = dict(
                built[0],
                cu_q=stage("unified_cu_q", cu_q, torch.int32),
                cu_q_host=cu_q,
                max_q=max(b - a for a, b in zip(cu_q, cu_q[1:])))
            if sliding:
                unified["sliding"] = built[1]
        except Exception:
            for key in temporary_keys:
                arena.free_key(key)
            raise

    # the tree path's KV writes as one list of (source, destination)
    # row pairs: the attention pass scatters them with a single kernel
    # launch per layer (kv_row_scatter)
    kv_src = kv_dst = None
    if kv_writes or tree_decode:
        src, dst_parts = [], []
        for key, r0, r1, dest in kv_writes:
            src.append(np.arange(r0, r1, dtype=np.int64))
            rows = arena.capacity_rows(key)[dest:dest + (r1 - r0)]
            if rows.numel() != r1 - r0:
                raise AssertionError("KV write exceeds reserved rows")
            dst_parts.append(rows.numpy())
        if tree_decode:
            src.append(np.arange(decode_row0, decode_row0 + len(decode),
                                 dtype=np.int64))
            dst_parts.append(np.asarray(decode.write_rows, dtype=np.int64))
        kv_src = _staged(torch, np.concatenate(src), torch.int64, pinned)
        kv_dst = _staged(torch, np.concatenate(dst_parts), torch.int64,
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
            canvas_meta["used_host"] = [pools[0].lengths[i] for i in canvas_seq]
            canvas_meta["page_rows"] = [pools[0].page_rows[i] for i in canvas_seq]
            canvas_meta["max_used"] = max(canvas_meta["used_host"])
            if "sliding" in unified:
                canvas_meta["sliding"] = dict(
                    table=unified["sliding"]["table"].index_select(0, seq),
                    used=unified["sliding"]["used"].index_select(0, seq),
                    max_used=max(pools[1].lengths[i] for i in canvas_seq))
    gdn = None
    if getattr(arena, "num_state_slots", 0):
        if attention_mode != "unified":
            raise ValueError("a model with Gated DeltaNet layers runs "
                             "the unified attention path")
        try:
            gdn = plan_phases(arena, unified_groups, stage)
        except Exception:
            for key in temporary_keys:
                arena.free_key(key)
            raise
    shared = None
    if (unified is not None and decode is not None
            and decode.shared_rows is not None and len(decode.shared_rows)):
        rows = decode_row0 + decode.shared_rows
        source = np.full(num_tokens, -1, dtype=np.int32)
        source[rows] = np.arange(len(rows), dtype=np.int32)
        shared = dict(
            rows=stage("shared_rows", rows, torch.int64),
            cu_q=stage("shared_cu_q", decode.shared_cu, torch.int32),
            max_q=int(np.diff(decode.shared_cu).max()),
            table=stage("shared_table", decode.shared_table, torch.int32).view(
                *decode.shared_table.shape),
            used=stage("shared_used", decode.shared_used, torch.int32),
            max_used=int(decode.shared_used.max()),
            source=stage("shared_source", source, torch.int32))
    meta = dict(
        layer=0, kv_src=kv_src, kv_dst=kv_dst, reads=reads,
        unified=unified, canvas=canvas_meta, gdn=gdn, shared=shared,
        cu_a=stage("cu_a", cu_a, torch.int32),
        max_a=int(np.diff(cu_a).max()) if len(cu_a) > 1 else 0)
    out = Chunk(
        input_ids=_staged_token_parts(
            torch, id_parts, num_tokens, pinned, staging),
        positions=stage("positions", _concatenate(position), torch.int64),
        final_indices=stage("finals", _concatenate(finals), torch.int64),
        meta=meta, attention_mode=attention_mode, tokens=num_tokens,
        layout=layout, row_spans=tuple(row_spans),
        temporary_keys=tuple(temporary_keys),
        fresh_keys=tuple(fresh_keys),
        rows_per_answer=(np.concatenate(rows_per_answer) if multi_row else None),
        decode_rows=decode)
    _tick(timing, "pack_h2d", t)
    return out


