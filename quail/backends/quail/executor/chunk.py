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
        temporary_keys: Arena keys the loop frees after the forward pass.
        fresh_keys: Keys whose prefix this chunk computes; the loop
            trims their sliding pages after the pass.
        rows_per_answer: Rows of final_indices each answer takes, one
            entry per suffix in chunk order, when some suffix reads
            more than its last row; None when every answer is one row.
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
    rows_per_answer: np.ndarray | None = None

    def apply_feed(self, torch, feed, source, page_tokens):
        """Fill the device-fed rounds from the answers they follow, on the GPU.

        Each device-fed round's first token becomes the token its
        source answer chose, its rows move past the draft tokens that
        answer accepted, and its KV reads and writes follow: the rows
        read one more resident token per accepted token, and the kept
        rows are written at their moved positions.

        Args:
            torch: The torch module.
            feed: The RoundFeed of the pass holding the source answers.
            source: Per device-fed round, in chunk order, the index of
                its source answer in that pass.
            page_tokens: Rows per KV page.
        """
        parts = self.meta["feed"]
        names = ("order", "first", "entry", "sequence", "rows", "kv_rows", "offset")
        arrays = device_arrays(torch, [source] + [plan[name] for _, plan in parts
                                                  for name in names],
                               self.input_ids.device)
        source = arrays[0]
        for index, (target, _) in enumerate(parts):
            order, first, entry, sequence, rows, kv_rows, offset = arrays[
                1 + index * len(names):1 + (index + 1) * len(names)]
            if target == "unified":
                unified = self.meta["unified"]
                used, table, dst = unified["used"], unified["table"], unified["dst"]
            else:
                reads = self.meta["reads"]
                used, table, dst = reads["used"], reads["table"], self.meta["kv_dst"]
            chosen = source.index_select(0, order)
            shift = feed.accepted.index_select(0, chosen) + 1
            self.input_ids.index_copy_(0, first, feed.chosen.index_select(0, chosen))
            self.positions.index_add_(0, rows, shift.index_select(0, entry))
            used.index_add_(0, sequence, shift.to(used.dtype))
            positions = self.positions.index_select(0, rows)
            # a table read from a page past the first starts at offset
            column = (positions - offset.index_select(0, entry)) // page_tokens
            pages = table[sequence.index_select(0, entry), column]
            dst.index_copy_(0, kv_rows, pages.to(torch.int64) * page_tokens
                            + positions % page_tokens)


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


_ZERO_LENGTHS = np.zeros(1, dtype=np.int64)


def _span_rows(spans) -> np.ndarray:
    """Join half-open integer spans into one array of their rows."""
    if not spans:
        return np.empty(0, dtype=np.int64)
    starts = np.fromiter((span[0] for span in spans), dtype=np.int64,
                         count=len(spans))
    stops = np.fromiter((span[1] for span in spans), dtype=np.int64,
                        count=len(spans))
    lengths = stops - starts
    ends = np.cumsum(lengths)
    total = int(ends[-1])
    return (np.repeat(starts - (ends - lengths), lengths)
            + np.arange(total, dtype=np.int64))


def _concatenate(parts):
    """Join int64 arrays, or give an empty one when there is nothing to join."""
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _join(parts):
    """Join int64 arrays, (start, length) ranges, and single values, in order.

    A decode round's rows are a range, so a chunk of rounds is joined
    without an array per round.
    """
    if not parts:
        return np.empty(0, dtype=np.int64)
    kinds = np.fromiter((2 if type(part) is tuple else 1 if type(part) is int
                         else 0 for part in parts), dtype=np.int64,
                        count=len(parts))
    if not kinds.any():
        return np.concatenate(parts)
    lengths = np.fromiter(
        (part[1] if type(part) is tuple else 1 if type(part) is int
         else len(part) for part in parts), dtype=np.int64, count=len(parts))
    ends = np.cumsum(lengths)
    offsets = ends - lengths
    out = np.empty(int(ends[-1]), dtype=np.int64)
    ranges = np.flatnonzero(kinds == 2)
    if len(ranges):
        starts = np.fromiter((parts[i][0] for i in ranges), dtype=np.int64,
                             count=len(ranges))
        span = lengths[ranges]
        total = int(span.sum())
        within = (np.arange(total, dtype=np.int64)
                  - np.repeat(np.cumsum(span) - span, span))
        out[np.repeat(offsets[ranges], span) + within] = (
            np.repeat(starts, span) + within)
    singles = np.flatnonzero(kinds == 1)
    if len(singles):
        out[offsets[singles]] = np.fromiter((parts[i] for i in singles),
                                            dtype=np.int64, count=len(singles))
    for i in np.flatnonzero(kinds == 0):
        out[offsets[i]:ends[i]] = parts[i]
    return out


def device_arrays(torch, arrays, device):
    """Copy int64 host arrays to the device in one transfer; return device views.

    The copy goes through pinned memory, so the host does not wait for
    the stream to reach it: a pageable copy would wait for every
    forward pass queued ahead of it.
    """
    lengths = [len(array) for array in arrays]
    joined = torch.from_numpy(np.concatenate(
        [np.asarray(array, dtype=np.int64).reshape(-1) for array in arrays]))
    if device.type == "cuda":
        joined = joined.pin_memory()
    joined = joined.to(device, non_blocking=True)
    bounds = np.cumsum([0, *lengths])
    return [joined[start:end] for start, end in zip(bounds, bounds[1:])]


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
        self.ids = (ids if isinstance(ids, np.ndarray)
                    and ids.dtype == np.int64 else
                    np.asarray(ids, dtype=np.int64))
        if len(lengths) == 1:
            # a decode round's one suffix skips the cumulative sums
            length = int(lengths[0])
            self.lengths = np.array([length], dtype=np.int64)
            self.offsets = np.array([0, length], dtype=np.int64)
        else:
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
        if isinstance(sequences, Suffixes):
            return sequences
        if (len(sequences) == 1 and isinstance(sequences[0], list)
                and len(sequences[0]) < 64):
            # a decode round's one short suffix skips the parts walk
            return cls(np.array(sequences[0], dtype=np.int64),
                       [len(sequences[0])])
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


def _feed_plan(fed_rounds, span_lengths) -> list:
    """Lay out the device-fed rounds' rows for Chunk.apply_feed(), on the host.

    Args:
        fed_rounds: Per round, in chunk order, (first row, rows, the
            reads it is a sequence of, its sequence there, its index
            among the KV write spans, the logical row its table's first
            page holds). The reads are "reads" (call B) or "unified".
        span_lengths: The rows of each KV write span, in order.

    Returns:
        Per reads with fed rounds, (reads, plan). A plan's order is its
        rounds' places in chunk order; first is each round's first row,
        rows every row of the rounds, entry each such row's round,
        sequence each round's sequence, kv_rows each row's index among
        the KV writes, and offset each round's table offset.
    """
    offsets = np.cumsum(span_lengths)
    offsets = np.concatenate(([0], offsets[:-1]))
    parts = []
    for target in ("reads", "unified"):
        order = [index for index, entry in enumerate(fed_rounds)
                 if entry[2] == target]
        if not order:
            continue
        first, counts, _, sequence, write, offset = zip(
            *(fed_rounds[index] for index in order))
        first, counts, sequence, write, offset = (
            np.array(column, dtype=np.int64)
            for column in (first, counts, sequence, write, offset))
        entry = np.repeat(np.arange(len(first)), counts)
        within = np.arange(int(counts.sum())) - np.repeat(
            np.cumsum(counts) - counts, counts)
        parts.append((target, dict(
            order=np.array(order, dtype=np.int64), first=first,
            rows=first[entry] + within, entry=entry, sequence=sequence,
            kv_rows=offsets[write][entry] + within, offset=offset)))
    return parts


def _shared_reads(keys, starts, counts, shared, share_groups, out) -> None:
    """Lay out the stacked reads of rounds whose documents share leading pages.

    Consecutive rounds of one share group read the shared pages as one
    sequence, so the pages are read once for all of them; each round's
    unified sequence then reads its table from its first own page, and
    the two are merged. A round that shares nothing reads its whole
    table in the unified call alone.

    Args:
        keys: Each round's key.
        starts: Each round's first row.
        counts: Each round's rows.
        shared: Each round's shared tokens, a whole number of pages.
        share_groups: Each round's share group.
        out: The shared reads: rows, cu_q, keys, used, and max_q.
    """
    index, count = 0, len(keys)
    while index < count:
        share = shared[index]
        if not share:
            index += 1
            continue
        end = index + 1
        while (end < count and shared[end] == share
               and share_groups[end] == share_groups[index]):
            end += 1
        rows = sum(counts[index:end])
        # the group's rows are contiguous; any member's table starts
        # with the shared pages
        out["rows"].append((starts[index], rows))
        out["cu_q"].append(out["cu_q"][-1] + rows)
        out["keys"].append(keys[index])
        out["used"].append(share)
        out["max_q"] = max(out["max_q"], rows)
        index = end


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
        self.src_spans = []     # (first row, last row + 1) per scatter
        self.dst = []           # the pool rows each span writes
        self.tail_src = []
        self.tail_dst = []

    def add_sequence(self, pages, kv_tokens):
        self.page_rows.append(pages)
        self.lengths.append(kv_tokens)

    def scatter_direct(self, view, r0, r1, logical_start):
        """Map input rows to the key's KV pages at the given logical position.

        Rows before the pool's start are not stored there.
        """
        first = max(logical_start, view.start)
        skip = first - logical_start
        if r0 + skip >= r1:
            return
        self.src_spans.append((r0 + skip, r1))
        offset = first - view.start
        self.dst.append(view.rows[offset:offset + (r1 - r0 - skip)])

    def scatter_suffix(self, view, temp, s0, s1, f, remainder, page_tokens):
        """Map suffix rows to temporary pages and copy any retained partial page."""
        suffix_tokens = s1 - s0
        self.src_spans.append((s0, s1))
        self.dst.append(temp.rows[remainder:remainder + suffix_tokens].numpy())
        if remainder:
            anchor_page = view.pages[(f - view.start) // page_tokens]
            base = int(anchor_page) * page_tokens
            self.tail_src.append((base, base + remainder))
            self.tail_dst.append(temp.rows[:remainder].numpy())

    def build(self, arena, stage, torch, staging, index):
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
            src=stage(f"unified_src{tag}", _span_rows(self.src_spans),
                      torch.int64),
            dst=stage(f"unified_dst{tag}", _concatenate(self.dst),
                      torch.int64),
            tail_src=(stage(f"tail_src{tag}", _span_rows(self.tail_src),
                            torch.int64) if self.tail_src else None),
            tail_dst=(stage(f"tail_dst{tag}", _concatenate(self.tail_dst),
                            torch.int64) if self.tail_dst else None),
            used=stage(f"unified_lengths{tag}", self.lengths, torch.int32),
            table=table,
            max_used=max(self.lengths))


def pack_chunk(torch, arena, groups, timing=None, pinned=True, *,
               attention_mode, staging=None, canvas=(), answer_row=0):
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
    * rounds: A RoundRequests of many documents' decode rounds packed as
      one group, with keys naming each one's document and feed_bound the
      most rows a device-fed round's source could have accepted plus
      one, bounding the KV read; every row is read and kept. Not with a canvas or a
      sliding pool. On the unified path, shared and share_groups give
      each round the leading pages its document shares with others and
      the group it shares them with: consecutive rounds of one group
      read those pages once (see _shared_reads).

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
    fresh_keys = []
    read_keys, read_used, cu_q = [], [], [0]
    max_q = 0
    unified_groups = []
    # call C: path rows reading their off-path ancestor rows
    node_q, node_k, node_cu_q, node_cu_k = [], [], [0], [0]
    plain_rounds = (not tree_path and not canvas
                    and not getattr(arena, "has_sliding", False))
    tree_rounds = tree_path and not canvas
    # the device-fed rounds: (first row, rows, read sequence, index
    # among the KV writes), and the most resident tokens one reads;
    # on the unified path the sequence and write indices are known
    # once the pool is laid out, so its rounds wait by group index
    fed_rounds = []
    fed_used = 0
    unified_fed = {}
    # on the unified path, rounds of one share group read the pages
    # their documents share once more, stacked (see _shared_reads)
    shared_reads = dict(rows=[], cu_q=[0], keys=[], used=[], max_q=0)
    for g in groups:
        rounds = g.get("rounds")
        if rounds is not None:
            if not (tree_rounds or plain_rounds):
                raise ValueError("rounds packed together run without a canvas "
                                 "or a sliding pool")
            keys = g["keys"]
            lengths = rounds.lengths
            ends = np.cumsum(lengths)
            begins = ends - lengths
            total = int(ends[-1]) if len(ends) else 0
            row0 = num_tokens
            id_parts.append(rounds.ids)
            position.append(np.repeat(rounds.positions, lengths)
                            + (np.arange(total, dtype=np.int64)
                               - np.repeat(begins, lengths)))
            finals.append((row0, total))
            cu_a.append(row0 + ends)
            rows_per_answer.append(lengths)
            multi_row = True
            layout.extend((key, 1) for key in keys)
            fed = np.flatnonzero(rounds.device_fed).tolist()
            bound = int(g.get("feed_bound", 0))
            starts = (row0 + begins).tolist()
            positions = rounds.positions.tolist()
            counts = lengths.tolist()
            if tree_path:
                first_write = len(kv_writes)
                kv_writes.extend(zip(keys, starts, (row0 + ends).tolist(), positions))
                reader_rows.append((row0, total))
                cu_q.extend((cu_q[-1] + ends).tolist())
                read_keys.extend(keys)
                read_used.extend(positions)
                if len(lengths) and int(lengths.max()) > max_q:
                    max_q = int(lengths.max())
                first_sequence = len(read_keys) - len(keys)
                for index in fed:
                    fed_rounds.append((starts[index], counts[index], "reads",
                                       first_sequence + index,
                                       first_write + index, 0))
                    fed_used = max(fed_used, positions[index] + bound)
            else:
                shared = g.get("shared")
                offsets = [0] * len(keys) if shared is None else shared.tolist()
                if shared is not None and np.any(shared):
                    _shared_reads(keys, starts, counts, offsets,
                                  g["share_groups"].tolist(), shared_reads)
                for index, (key, start, f, count) in enumerate(
                        zip(keys, starts, positions, counts)):
                    offset = offsets[index]
                    if rounds.device_fed[index]:
                        # a shifted round reads up to its draft's bound
                        # past its rows
                        unified_fed[len(unified_groups)] = (
                            start, count, f + count + bound - offset)
                    unified_groups.append((key, f, start, count, offset))
            num_tokens = row0 + total
            continue
        sufs0 = g["suffixes"]
        if (tree_rounds and g.get("prefix") is None and g.get("single")
                and isinstance(sufs0, Suffixes) and len(sufs0) == 1
                and g.get("read_all_rows") and g.get("paths") is None
                and g.get("canvas") is None and g.get("read_rows") is None
                and g.get("read_mask") is None and g["f"]
                and arena.is_resident(g["key"])):
            # a decode round under tree attention: one suffix over the
            # key's resident rows, every row read in call B and the
            # kept rows written to the key's pages; the general
            # bookkeeping below reduces to this
            f = g["f"]
            key = g["key"]
            total = int(sufs0.lengths[0])
            row0 = num_tokens
            id_parts.append(sufs0.ids)
            position.append((f, total))
            finals.append((row0, total))
            cu_a.append(row0 + total)
            rows_per_answer.append(total)
            multi_row = True
            kept = g.get("write_suffix_tokens") or 0
            if kept:
                kv_writes.append((key, row0, row0 + kept, f))
            layout.append((key, 1))
            reader_rows.append((row0, total))
            cu_q.append(cu_q[-1] + total)
            read_keys.append(key)
            read_used.append(f)
            if total > max_q:
                max_q = total
            num_tokens = row0 + total
            continue
        if (plain_rounds and g.get("prefix") is None
                and isinstance(sufs0, Suffixes) and len(sufs0) == 1
                and g.get("read_all_rows") and g.get("paths") is None
                and g.get("canvas") is None and g.get("read_rows") is None
                and g.get("read_mask") is None
                and arena.is_resident(g["key"])):
            # a decode round: one kept suffix over resident KV, every
            # row read; the general bookkeeping below reduces to this
            f = g["f"]
            total = int(sufs0.lengths[0])
            row0 = num_tokens
            id_parts.append(sufs0.ids)
            position.append((f, total))
            finals.append((row0, total))
            cu_a.append(row0 + total)
            rows_per_answer.append(total)
            multi_row = True
            num_tokens = row0 + total
            layout.append((g["key"], 1))
            unified_groups.append((g["key"], f, row0, total, 0))
            continue
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
        if n == 1 and not width:
            # a decode round's one suffix needs no cumulative arithmetic
            total = int(sufs.lengths[0])
            within = np.arange(total, dtype=np.int64)
            ends = sufs.lengths
            begins = _ZERO_LENGTHS
        else:
            widths = sufs.lengths + width
            total = int(widths.sum())
            ends = np.cumsum(widths)
            begins = ends - widths
            within = (np.arange(total, dtype=np.int64)
                      - np.repeat(begins, widths))
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
    cu_a = _join(cu_a)

    t = _tick(timing, "pack_py", t)
    reads = None
    if read_keys:
        table = arena.block_table_rows(
            [arena.owned_pages(key) for key in read_keys])
        t = _tick(timing, "pack_blocktable", t)
        rows = _join(reader_rows)
        # row -> its index in call B's output, -1 for rows that read
        # nothing resident; the fused merge kernel's map
        source = np.full(num_tokens, -1, dtype=np.int32)
        source[rows] = np.arange(len(rows), dtype=np.int32)
        reads = dict(
            rows=_staged(torch, rows, torch.int64, pinned),
            cu_q=stage("unified_cu_q", cu_q, torch.int32),
            max_q=max_q, used=_staged(torch, read_used, torch.int32, pinned),
            max_used=max(max(read_used), fed_used), table=table,
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
    feed = None
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
            for group_index, spec in enumerate(unified_groups):
                if type(spec) is tuple:
                    # a decode round packed by the fast path above:
                    # contiguous rows scattered straight into its key's
                    # capacity rows
                    key, f, r0, count, offset = spec
                    rows_np = arena.capacity_rows_numpy(key)
                    end = f + count
                    if end > len(rows_np):
                        raise ValueError(
                            f"group {key!r} keeps suffix KV but its pages "
                            f"hold {len(rows_np)} rows, not {end}")
                    pool = pools[0]
                    waiting = unified_fed.get(group_index)
                    if waiting is not None:
                        fed_rounds.append((waiting[0], waiting[1], "unified",
                                           len(cu_q) - 1, len(pool.src_spans),
                                           offset))
                        fed_used = max(fed_used, waiting[2])
                    pool.src_spans.append((r0, r0 + count))
                    pool.dst.append(rows_np[f:end])
                    # a round whose shared pages another read covers
                    # reads its table from its first own page
                    pool.add_sequence(
                        arena.owned_pages(key)[offset // page_tokens:], end - offset)
                    cu_q.append(cu_q[-1] + count)
                    continue
                key = spec["key"]
                f = spec["f"]
                r0, r1 = spec["row0"], spec["row1"]
                begins, ends = spec["suffix_spans"]
                logical_start = spec["start"] if spec["fresh"] else f
                if spec["fresh"]:
                    fresh_keys.append(key)
                count = r1 - r0
                # numpy views of the cached row tables slice without
                # a copy; 20,000 torch slices a pass do not
                views = [_PoolView(
                    arena.capacity_rows(key).numpy(),
                    arena.owned_pages(key), 0)]
                if sliding:
                    views.append(_PoolView(
                        arena.capacity_rows_sliding(key).numpy(),
                        arena.owned_sliding_pages(key),
                        arena.sliding_start(key)))
                if len(views) == 1:
                    direct = (len(begins) == 1 and logical_start + count
                              <= len(views[0].rows))
                else:
                    direct = len(begins) == 1 and all(
                        logical_start + count - view.start
                        <= len(view.rows)
                        for view in views)
                suffix_total = int((ends - begins).sum())
                if spec["keeps"] and not direct:
                    raise ValueError(
                        f"group {key!r} keeps suffix KV but its pages hold "
                        f"{len(views[0].rows)} rows, not "
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

            built = [pool.build(arena, stage, torch, staging, index)
                     for index, pool in enumerate(pools)]
            unified = dict(
                built[0],
                cu_q=stage("unified_cu_q", cu_q, torch.int32),
                max_q=max(b - a for a, b in zip(cu_q, cu_q[1:])))
            unified["max_used"] = max(unified["max_used"], fed_used)
            if shared_reads["keys"]:
                if sliding:
                    raise ValueError("shared reads run without a sliding pool")
                rows = _join(shared_reads["rows"])
                # row -> its index in the shared read, -1 for rows that
                # read no shared pages; the fused merge kernel's map
                source = np.full(num_tokens, -1, dtype=np.int32)
                source[rows] = np.arange(len(rows), dtype=np.int32)
                unified["shared"] = dict(
                    rows=_staged(torch, rows, torch.int64, pinned),
                    cu_q=_staged(torch, shared_reads["cu_q"], torch.int32, pinned),
                    max_q=shared_reads["max_q"],
                    used=_staged(torch, shared_reads["used"], torch.int32, pinned),
                    max_used=max(shared_reads["used"]),
                    table=arena.block_table_rows(
                        [arena.owned_pages(key) for key in shared_reads["keys"]]),
                    source=_staged(torch, source, torch.int32, pinned))
            if sliding:
                unified["sliding"] = built[1]
            if fed_rounds:
                feed = _feed_plan(fed_rounds,
                                  [r1 - r0 for r0, r1 in pools[0].src_spans])
        except Exception:
            for key in temporary_keys:
                arena.free_key(key)
            raise

    # the tree path's KV writes as one list of (source, destination)
    # row pairs: the attention pass scatters them with a single kernel
    # launch per layer (kv_row_scatter)
    kv_src = kv_dst = None
    if kv_writes:
        dst_parts = []
        for key, r0, r1, dest in kv_writes:
            rows = arena.capacity_rows_numpy(key)[dest:dest + (r1 - r0)]
            if len(rows) != r1 - r0:
                raise AssertionError("KV write exceeds reserved rows")
            dst_parts.append(rows)
        kv_src = _staged(torch, _span_rows([(r0, r1) for _, r0, r1, _ in kv_writes]),
                         torch.int64, pinned)
        kv_dst = _staged(torch, np.concatenate(dst_parts), torch.int64, pinned)
    if fed_rounds and tree_path:
        feed = _feed_plan(fed_rounds, [r1 - r0 for _, r0, r1, _ in kv_writes])
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
    meta = dict(
        layer=0, kv_src=kv_src, kv_dst=kv_dst, reads=reads, feed=feed,
        unified=unified, canvas=canvas_meta, gdn=gdn,
        cu_a=stage("cu_a", cu_a, torch.int32),
        max_a=int(np.diff(cu_a).max()) if len(cu_a) > 1 else 0)
    out = Chunk(
        input_ids=_staged_token_parts(
            torch, id_parts, num_tokens, pinned, staging),
        positions=stage("positions", _join(position), torch.int64),
        final_indices=stage("finals", _join(finals), torch.int64),
        meta=meta, attention_mode=attention_mode, tokens=num_tokens,
        layout=layout, temporary_keys=tuple(temporary_keys),
        fresh_keys=tuple(fresh_keys),
        rows_per_answer=_join(rows_per_answer) if multi_row else None)
    _tick(timing, "pack_h2d", t)
    return out


