"""The overlapped chunk loop.

Pack on CPU while the GPU runs, gate answers, free pages immediately.

- run_join: continuous anchor admission with JoinAdmission, stages
  mixed in one chunk, pages freed when an anchor fails a gate or
  answers its last stage.
- run_filter: continuous admission with FilterAdmission, pages freed
  on FALSE or after the last stage.

Torch is imported lazily when model execution starts.
"""

import time

import numpy as np

from quail.backends.quail.executor.attention import (
    FILTER_ATTENTION,
    Chunk,
    ChunkImage,
)
from quail.backends.quail.executor.pack import FilterAdmission, JoinAdmission
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
    """Copy Arrow token views into one GPU input tensor."""
    host = (torch.empty(total, dtype=torch.int64, pin_memory=pinned)
            if staging is None else staging.host("tokens", total, torch.int64))
    offset = 0
    for sequence in sequences:
        for part in _token_parts(sequence):
            count = len(part)
            if not count:
                continue
            if hasattr(part, "arrow_array"):
                source = torch.from_dlpack(part.arrow_array)
            elif torch.is_tensor(part):
                source = part
            elif staging is not None and isinstance(part, tuple):
                source = staging.fixed(part)
            else:
                source = torch.as_tensor(part)
            host[offset:offset + count].copy_(source)
            offset += count
    if offset != total:
        raise AssertionError(
            f"packed {offset} token ids into a {total}-token chunk")
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
      f         Kept-context length (suffix positions start here).
      suffixes  List of suffix token lists.
      write_suffix_tokens  Leading rows of the first suffix to scatter
                into the key's pages at offset len(prefix).

    A fresh group without arena pages packs [prefix | suffix] as one
    causal segment (no scatter, no paged read). This only works with
    at most one suffix. Under unified attention a chunk is either all
    paged or all unpaged.

    canvas is the token ids a diffusion model denoises: they follow
    every suffix as extra rows, and the answer row is canvas row
    answer_row instead of the suffix's last row. Canvas KV goes
    wherever the suffix's KV goes and is never kept. Canvas rows run
    the unified path only.

    A fresh group may carry images: (block, page) pairs where block
    names the prefix span its soft tokens occupy (offset, soft_tokens)
    and page is the rendered pixels. The chunk lists them as
    ChunkImage rows for the model to embed, and meta["blocks"] gives
    the attention path each block's rows and the KV bound it may read
    up to. Images run the paged unified path only.
    """
    def stage(name, values, dtype):
        if staging is not None:
            return staging.copy(name, values, dtype)
        return _staged(torch, values, dtype, pinned)

    def concatenate(parts):
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    if attention_mode not in ("merge_quant", "unified"):
        raise ValueError(
            "attention_mode must be 'merge_quant' or 'unified', "
            f"got {attention_mode!r}")
    canvas = tuple(canvas)
    if canvas and attention_mode != "unified":
        raise ValueError("canvas rows run the unified attention path")
    if canvas and not 0 <= answer_row < len(canvas):
        raise ValueError(
            f"answer_row {answer_row} is outside the {len(canvas)}-row canvas")
    t = time.perf_counter() if timing is not None else 0.0
    # unified scatters every fresh row through its own src/dst map, so
    # the cross and kv_writes bookkeeping below is two-call only
    two_call = attention_mode != "unified"
    canvas_rows = []      # (first row, end row) per canvas
    canvas_seq = []       # its sequence in the unified paged call
    images = []           # ChunkImage per image, in row order
    block_ends = []       # (unified group index, logical end) per image
    blocks_meta = None
    id_parts, token_count = [], 0
    pos, cu_a, finals = [], [0], []
    suffix_rows = []
    kv_writes, layout = [], []
    fresh_keys = []
    cross_keys, cross_used, cu_q = [], [], [0]
    max_q = 0
    unified_groups = []
    for g in groups:
        fresh = g.get("prefix") is not None
        f = g["f"]
        key = g["key"]
        paged = arena.is_resident(key)
        row0 = token_count
        if g.get("images"):
            if not (fresh and paged and attention_mode == "unified"):
                raise ValueError(
                    f"group {key!r}: images ride a fresh prefix on the "
                    f"paged unified path")
            for block, page in g["images"]:
                if block.end > len(g["prefix"]):
                    raise ValueError(
                        f"group {key!r}: image block {block} runs past "
                        f"its {len(g['prefix'])}-token prefix")
                images.append(ChunkImage(row0 + block.offset,
                                         block.soft_tokens, page))
                block_ends.append((len(unified_groups), block.end))
        if fresh:
            if not paged and len(g["suffixes"]) > 1:
                raise ValueError(
                    f"group {key!r}: an unpaged fresh group packs "
                    f"[prefix | suffix] as one causal segment, which "
                    f"only one suffix may join - allocate pages or "
                    f"split the group")
            id_parts.append(g["prefix"])
            token_count += len(g["prefix"])
            pos.append(np.arange(len(g["prefix"]), dtype=np.int64))
            if paged or not g["suffixes"]:
                cu_a.append(token_count)
            if paged and two_call:
                kv_writes.append((key, row0, row0 + len(g["prefix"]),
                                  0))
                fresh_keys.append(key)
        prefix_end = token_count
        s_row0 = token_count
        suffix_spans = []
        for si, suf in enumerate(g["suffixes"]):
            srow = token_count
            id_parts.append(suf)
            token_count += len(suf)
            pos.append(np.arange(f, f + len(suf), dtype=np.int64))
            if canvas:
                # the canvas continues the suffix's positions; its
                # first row carries the answer
                first = f + len(suf)
                id_parts.append(canvas)
                pos.append(np.arange(first, first + len(canvas),
                                     dtype=np.int64))
                canvas_rows.append((token_count, token_count + len(canvas)))
                finals.append(token_count + answer_row)
                token_count += len(canvas)
            else:
                finals.append(token_count - 1)
            suffix_spans.append((srow, token_count))
            cu_a.append(token_count)
            wst = g.get("write_suffix_tokens", 0)
            if si == 0 and wst and paged and two_call:
                # the shared question preamble joins the kept KV right
                # after the document rows: after the fresh prefix, or
                # after a kept document's f rows
                dest = len(g["prefix"]) if fresh else f
                kv_writes.append((key, srow, srow + wst, dest))
        s_count = token_count - s_row0
        layout.append((key, len(g["suffixes"])))
        if s_count and f and paged and two_call:
            suffix_rows.extend(range(s_row0, token_count))
            cu_q.append(cu_q[-1] + s_count)
            cross_keys.append(key)
            cross_used.append(f)
            max_q = max(max_q, s_count)
        if attention_mode == "unified":
            if paged:
                unified_groups.append(dict(
                    key=key, fresh=fresh, f=f, row0=row0,
                    prefix_end=prefix_end, row1=token_count,
                    suffix_spans=suffix_spans))
            elif not fresh:
                raise ValueError(
                    "unified attention requires pages for a kept "
                    "group - an unpaged kept group has no KV to read")

    t = _tick(timing, "pack_py", t)
    cross = None
    if cross_keys:
        table, _ = arena.block_table(cross_keys)
        t = _tick(timing, "pack_blocktable", t)
        used = _staged(torch, cross_used, torch.int32, pinned)
        cross = dict(
            rows=_staged(torch, suffix_rows, torch.int64, pinned),
            cu_q=stage("unified_cu_q", cu_q, torch.int32),
            max_q=max_q, used=used,
            max_used=max(cross_used), table=table)
        # row -> its index in call B's output, -1 for prefix rows;
        # the fused merge kernel's map
        source = [-1] * token_count
        for i, row in enumerate(suffix_rows):
            source[row] = i
        cross["source"] = _staged(
            torch, source, torch.int32, pinned)
    t = _tick(timing, "pack_cross", t)

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
        prefix_seq = []       # per unified group, the sequence its prefix rows join
        pool_starts = []      # per unified group, each pool view's start row

        try:
            for spec in unified_groups:
                key = spec["key"]
                f = spec["f"]
                r0, r1 = spec["row0"], spec["row1"]
                spans = spec["suffix_spans"]
                logical_start = 0 if spec["fresh"] else f
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
                pool_starts.append([view.start for view in views])
                direct = len(spans) == 1 and all(
                    logical_start + count - view.start <= view.rows.numel()
                    for view in views)
                suffix_total = sum(e - s for s, e in spans)
                if direct:
                    for pool, view in zip(pools, views):
                        pool.scatter_direct(view, r0, r1, logical_start)
                        pool.add_sequence(view.pages, f + suffix_total - view.start)
                    cu_q.append(cu_q[-1] + count)
                    prefix_seq.append(len(cu_q) - 2)
                    if canvas:
                        canvas_seq.append(len(cu_q) - 2)
                    continue

                prefix_count = spec["prefix_end"] - r0
                prefix_seq.append(len(cu_q) - 1 if prefix_count else None)
                if prefix_count:
                    for pool, view in zip(pools, views):
                        pool.scatter_direct(view, r0, spec["prefix_end"], 0)
                        pool.add_sequence(
                            view.pages[:arena.pages_needed(f - view.start)],
                            f - view.start)
                    cu_q.append(cu_q[-1] + prefix_count)

                for s0, s1 in spans:
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
                    if canvas:
                        canvas_seq.append(len(cu_q) - 2)

            built = [pool.build(arena, stage, torch, staging, index)
                     for index, pool in enumerate(pools)]
            unified = dict(
                built[0],
                cu_q=stage("unified_cu_q", cu_q, torch.int32),
                max_q=max(b - a for a, b in zip(cu_q, cu_q[1:])))
            if sliding:
                unified["sliding"] = built[1]
            if images:
                blocks_meta = _blocks_meta(
                    torch, stage, images, block_ends, prefix_seq,
                    pool_starts, unified)
        except Exception:
            for key in temporary_keys:
                arena.free_key(key)
            raise

    # All current KV writes use one scatter.
    kv_src = kv_dst = None
    if kv_writes:
        src = []
        dst_parts = []
        for key, r0, r1, dest in kv_writes:
            src.extend(range(r0, r1))
            rows = arena.capacity_rows(key)[dest:dest + (r1 - r0)]
            if rows.numel() != r1 - r0:
                raise AssertionError("KV write exceeds reserved rows")
            dst_parts.append(rows)
        kv_src = _staged(torch, src, torch.int64, pinned)
        kv_dst = _staged(torch, torch.cat(dst_parts), torch.int64,
                         pinned)
    t = _tick(timing, "pack_kv", t)

    canvas_meta = None
    if canvas_rows:
        if (attention_mode == "unified" and unified is None
                and len(canvas_rows) != len(cu_a) - 1):
            raise ValueError(
                "the unpaged canvas call pairs one canvas with each "
                "causal segment; a group without a suffix has none")
        cu_c = [0]
        for a, b in canvas_rows:
            cu_c.append(cu_c[-1] + b - a)
        canvas_meta = dict(
            rows=stage("canvas_rows", concatenate(
                [np.arange(a, b, dtype=np.int64) for a, b in canvas_rows]),
                torch.int64),
            cu_q=stage("canvas_cu_q", cu_c, torch.int32),
            max_q=len(canvas))
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
        layer=0, kv_src=kv_src, kv_dst=kv_dst, cross=cross,
        unified=unified, canvas=canvas_meta, blocks=blocks_meta,
        cu_a=stage("cu_a", cu_a, torch.int32),
        max_a=max(cu_a[i + 1] - cu_a[i] for i in range(len(cu_a) - 1)))
    out = Chunk(
        input_ids=_staged_token_parts(
            torch, id_parts, token_count, pinned, staging),
        positions=stage("positions", concatenate(pos), torch.int64),
        final_indices=stage("finals", finals, torch.int64),
        meta=meta, attention_mode=attention_mode, tokens=token_count,
        layout=layout, temporary_keys=tuple(temporary_keys),
        fresh_keys=tuple(fresh_keys), images=tuple(images))
    _tick(timing, "pack_h2d", t)
    return out


def _blocks_meta(torch, stage, images, block_ends, prefix_seq, pool_starts,
                 unified):
    """The block attention call's rows, sequences, and KV bounds per pool.

    Each image block is one query sequence whose KV ends at the block's
    last soft token, in the every-token pool and, when present, the
    sliding pool.
    """
    rows = np.concatenate([
        np.arange(image.row0, image.row0 + image.rows, dtype=np.int64)
        for image in images])
    cu_b = [0]
    for image in images:
        cu_b.append(cu_b[-1] + image.rows)
    seq = []
    for group, _end in block_ends:
        if prefix_seq[group] is None:
            raise AssertionError("an image block belongs to a prefix sequence")
        seq.append(prefix_seq[group])
    seq_index = stage("blocks_seq", seq, torch.int64)
    meta = dict(
        rows=stage("blocks_rows", rows, torch.int64),
        cu_q=stage("blocks_cu_q", cu_b, torch.int32),
        max_q=max(image.rows for image in images))
    for index, pool in enumerate([unified] + [unified.get("sliding")]):
        if pool is None:
            continue
        used = [end - pool_starts[group][index] for group, end in block_ends]
        if min(used) <= 0:
            raise ValueError(
                "an image block lies before its key's sliding window origin")
        entry = dict(
            table=pool["table"].index_select(0, seq_index),
            used=stage(f"blocks_used{'_sliding' if index else ''}", used,
                       torch.int32),
            max_used=max(used))
        if index == 0:
            meta.update(entry)
        else:
            meta["sliding"] = entry
    return meta


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
             anchor_keys=None, anchor_done=None, anchor_source=None,
             anchor_partners=None, anchor_batch=None, staging=None,
             images=None):
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
            With anchor_source it must be an empty list; the join
            appends each streamed anchor's prefix to it.
        stage_suffixes: Per stage, the partner suffix token lists.
        budget: Chunk token budget.
        stage_frames: Per stage, task framing token list written into
            each anchor's kept KV after the document rows.
        anchor_keys: Stable arena key for each anchor. List positions are
            used when omitted. With anchor_source it must be an empty
            list; the join appends each streamed anchor's key to it.
        anchor_done: Optional callback(anchor position, final answer row).
            It owns the anchor's final retain or free decision.
        anchor_source: Optional stream that admits anchors while the
            join runs: next() runs one chunk of its own and returns
            ((key, prefix) pairs, blocked); done is True once nothing
            is left; chunks counts its launched chunks. Every key it
            hands over is resident in the arena, so the join packs no
            prefix tokens for it. The join pulls until it can fill a
            chunk and then runs one. blocked means the source is out
            of arena pages until the join frees some.
        anchor_partners: Optional callable(anchor key) -> per stage,
            the indices into that stage's partner list the anchor
            streams, or None for the whole list. Omitted means every
            anchor streams every partner.
        anchor_batch: Optional callable(keys) -> the keys to admit, run
            on each batch the source hands over before admission. A
            key it leaves out is freed, never admitted.
        staging: Optional reusable input transfer buffers.
        images: Optional image source for anchors whose prefixes hold
            image placeholders (PageImages): take(a) gives anchor a's
            rendered pages, which ride the chunk that packs its prefix.
            Only for anchors listed in anchor_prefixes; a streamed
            anchor's pages were embedded by its own chain. The caller
            closes it.

    Returns:
        (ans, spans, tokens): ans[j][a] = 0/1 row over the stage-j
        partners anchor a streams, with a in admission order; spans =
        (stage, start_event, end_event) per forward; tokens = fresh
        tokens packed.
    """
    k = len(stage_suffixes)
    if anchor_source is not None:
        if anchor_prefixes or anchor_keys:
            raise ValueError(
                "anchor_source fills anchor_prefixes and anchor_keys")
        prefixes, keys = anchor_prefixes, anchor_keys
    else:
        prefixes = list(anchor_prefixes)
        keys = (list(range(len(prefixes))) if anchor_keys is None
                else list(anchor_keys))
    if len(keys) != len(prefixes):
        raise ValueError("anchor_keys must match anchor_prefixes")
    frames = stage_frames or [[] for _ in range(k)]
    mode = pipeline.join_attention
    if images is not None:
        if anchor_source is not None:
            raise ValueError(
                "images belong to anchor_prefixes; a streamed anchor's "
                "pages were embedded by its chain")
        if mode != FILTER_ATTENTION:
            raise ValueError("images run the paged unified path")
        if not pipeline.takes_images:
            raise ValueError(
                f"{type(pipeline).__name__} does not embed images")
    # pages taken for an anchor whose chunk was split or retried, so
    # the retry packs the same pages instead of taking the row again
    taken_pages = {}
    canvas = tuple(pipeline.canvas_ids)
    answer_row = pipeline.canvas_answer_row

    def entry_rows(a, j, start, end, carried):
        f = len(prefixes[a])
        frame = frames[j]
        sufs = [stage_suffixes[j][i]
                for i in sched.partner_indices(a, j, start, end)]
        rows = (f if carried else 0) + sum(len(s) + len(canvas) for s in sufs)
        if frame and start == 0:
            rows += len(frame) + len(canvas)
        return rows

    if k == 0:
        return [], [], 0
    # a frame entry's canvas rows land in the anchor's pages after the
    # frame, so the pages cover them
    frame_max = max(len(f) + (len(canvas) if f else 0) for f in frames)
    resident = {a: arena.held_cost(keys[a]) for a in range(len(keys))
                if arena.is_resident(keys[a])}
    # Every resident anchor stays available for the whole join while
    # fresh admissions evict unrelated retained KV.
    for a in resident:
        arena.pin(keys[a])
    def partners_of(key):
        return None if anchor_partners is None else anchor_partners(key)

    sched = JoinAdmission(
        [len(p) for p in prefixes],
        [[len(s) for s in sufs] for sufs in stage_suffixes],
        budget, arena.n_pages, arena.page_tokens,
        frame_tokens=[len(f) for f in frames], resident=resident,
        anchor_partners={a: partners_of(keys[a]) for a in range(len(keys))},
        # a windowed model may pack any chunk unified, so it reserves
        # the unified path's temporary pages throughout
        temporary_suffix_pages=mode == FILTER_ATTENTION,
        answer_dtype=async_ans.dtype,
        canvas_tokens=len(canvas),
        page_cost=arena.page_cost,
    )
    if images is not None:
        # an anchor without a partner never packs, so its pages are
        # not rendered
        images.open(budget, docs=[a for a in range(len(keys))
                                  if sched.runs(a)])
    spans = []
    tokens = 0
    outstanding = []     # (groups, handle) in launch order
    scoring = async_ans.dtype is not None
    label = "AI.SCORE" if scoring else f"join ({k} stages)"
    total = (sum(sched._count(a, j) for a in range(len(prefixes)) for j in range(k))
             if scoring else len(prefixes))
    progress = Progress(
        label, total=None if anchor_source is not None else total,
        unit="scores" if scoring else "anchors")
    finished = [0]

    def admit(items):
        if anchor_batch is not None and items:
            kept = set(anchor_batch([key for key, _ in items]))
            for key, _ in items:
                if key not in kept and arena.is_resident(key):
                    arena.free_key(key)
            items = [(key, prefix) for key, prefix in items if key in kept]
        for key, prefix in items:
            if not arena.is_resident(key):
                raise ValueError(
                    f"streamed anchor {key!r} has no KV in the arena")
            arena.pin(key)
            keys.append(key)
            prefixes.append(prefix)
            sched.admit(len(prefix), arena.held_cost(key),
                        partners=partners_of(key))

    def pull(evict_retained=False, force=False):
        """Run source chunks until a join chunk can fill or the source blocks.

        Returns whether the source launched a chunk or handed over an
        anchor.
        """
        moved = False
        while not anchor_source.done and (
                force or sched.buildable_tokens() < budget):
            force = False
            before = anchor_source.chunks
            items, blocked = anchor_source.next(
                evict_retained=evict_retained)
            admit(items)
            moved = moved or bool(items) or anchor_source.chunks > before
            if blocked:
                break
        return moved

    def build(chunk_groups):
        specs = []
        for a, j, start, end, carried in chunk_groups:
            key = keys[a]
            f = len(prefixes[a])
            frame = frames[j]
            got = arena.activate(key, f, capacity_tokens=f + frame_max,
                                 base_tokens=f)
            assert got is not None, \
                "scheduler admitted an anchor the arena cannot hold"
            sufs = [stage_suffixes[j][i]
                    for i in sched.partner_indices(a, j, start, end)]
            if frame and start == 0:
                # frame entry: scatter the frame into KV after the
                # document rows. The pair entry reads doc + frame.
                # The frame entry's answer bit is skipped by report().
                specs.append(dict(
                    key=key,
                    prefix=prefixes[a] if carried else None,
                    f=f, suffixes=[frame],
                    write_suffix_tokens=len(frame)))
                specs.append(dict(
                    key=key, prefix=None, f=f + len(frame),
                    suffixes=sufs))
            else:
                specs.append(dict(
                    key=key,
                    prefix=prefixes[a] if carried else None,
                    f=f + len(frame),
                    suffixes=sufs))
            if carried and images is not None:
                if a not in taken_pages:
                    taken_pages[a] = images.take(a)
                specs[-2 if frame and start == 0 else -1]["images"] = \
                    taken_pages[a]
        return pack_chunk(torch, arena, specs, attention_mode=mode,
                          staging=staging, canvas=canvas,
                          answer_row=answer_row)

    def settle(anchor):
        if anchor_done is None:
            if arena.is_resident(keys[anchor]):
                arena.free_key(keys[anchor])
        else:
            anchor_done(anchor, sched.answers[k - 1].get(anchor, []))

    def event(kind, anchor):
        if kind == "finished":
            settle(anchor)
            finished[0] += 1
        elif arena.is_resident(keys[anchor]):
            arena.free_key(keys[anchor])

    def report(entry):
        groups, handle = entry
        bits = async_ans.result(handle)
        pos = 0
        for a, j, start, end, _ in groups:
            if frames[j] and start == 0:
                pos += 1        # the frame entry's bit means nothing
            cnt = end - start
            for kind, anchor in sched.report(
                    a, j, start, end, bits[pos:pos + cnt]):
                event(kind, anchor)
            pos += cnt
        progress.update(
            progress.done + sum(end - start for _, _, start, end, _ in groups)
            if scoring else finished[0])

    def run_part(part):
        """Run one chunk of groups, halving it when its pages do not fit.

        The admission prices a chunk's temporary suffix pages, but the
        packer takes them in both pools a page at a time, and on long
        documents with many short suffixes the tighter pool can run
        out. Halving costs only chunk efficiency.
        """
        try:
            run_one(part)
        except ArenaFullError as error:
            # retained KV nothing here reads makes room first; it is
            # a cache
            need = arena.page_cost(sum(entry_rows(*e) for e in part))
            if arena.evict_retained(need):
                logger.info("join chunk of %d groups retried after "
                            "evicting retained KV", len(part))
                run_part(part)
                return
            if len(part) < 2:
                raise
            logger.info("join chunk of %d groups split: %s", len(part), error)
            half = len(part) // 2
            run_part(part[:half])
            run_part(part[half:])

    def run_one(part):
        nonlocal tokens
        chunk = build(part)
        tokens += chunk.tokens
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normed = _forward(pipeline, arena, chunk)
        e1.record()
        for key in chunk.fresh_keys:
            arena.trim_window(key)
        for a, _, _, _, carried in part:
            if carried:
                taken_pages.pop(a, None)
        spans.append((part[0][1], e0, e1))
        outstanding.append((part, async_ans.submit(normed)))
        # read the previous chunk's answers while this one runs
        while len(outstanding) > 1:
            report(outstanding.pop(0))

    while True:
        if anchor_source is not None and not anchor_source.done:
            pull()
        # anchors with no partner at their first stage never run
        for kind, anchor in sched.take_settled():
            event(kind, anchor)
        if sched.done() and (anchor_source is None
                             or anchor_source.done):
            break
        if sched.blocked_pages:
            # the free list is short for the next fresh anchor:
            # retained KV nothing here reads makes room
            arena.evict_retained(sched.blocked_pages)
        groups = sched.next_chunk(arena.free_pages)
        if not groups:
            if outstanding:
                report(outstanding.pop(0))
                continue
            if anchor_source is not None and not anchor_source.done:
                # the source has to move; it may evict retained KV to admit
                if pull(evict_retained=True, force=True):
                    continue
            raise AssertionError("nothing buildable and nothing in flight")
        run_part(groups)
    while outstanding:
        report(outstanding.pop(0))
    progress.finish(f"{label} done", f"{tokens:,} fresh tokens")
    return sched.answers, spans, tokens


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
WARMUP_VERSION = 4


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
    join_chunk adds one run_join call.
    """
    warm_docs, question, doc = _warm_inputs(budget)
    q_max = len(question)
    modes = dict.fromkeys((FILTER_ATTENTION, pipeline.join_attention))
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

def _shared_preamble_tokens(question_ids):
    """Longest common token prefix across the stage questions."""
    if len(question_ids) < 2:
        return 0
    p = 0
    while all(len(q) > p and q[p] == question_ids[0][p]
              for q in question_ids):
        p += 1
    return p


class FilterStream:
    """The filter chain, one chunk per next() call.

    Continuous admission with survivor priority. Pages are freed on
    FALSE or after the last stage. With hold_survivors, a document
    that passes its last stage keeps its pages pinned instead, and
    next() hands it over as a (key, prefix) pair for a join that reads
    its KV; that join frees or retains the key.

    Args:
        torch: The torch module, imported by the caller.
        arena: KVArena holding the documents' KV pages.
        pipeline: ModelPipeline that runs each packed forward chunk.
        async_ans: AsyncAnswers readout for TRUE/FALSE bits.
        doc_ids: Per-document token lists.
        question_ids: Per-stage question token lists.
        budget: Chunk token budget.
        timing: CPU seconds per loop phase accumulate into it.
        pinned: False for pageable blocking copies.
        limit: Stop admitting documents after this many survivors.
            None runs every document.
        arena_writes: Whether document KV is written to the arena.
            Must be True with multiple stages.
        arena_keys: Stable arena key for each document. List positions
            are used when omitted.
        retain_survivors: Passing document positions to keep for
            joins, through the shared arena retention pool capped at
            the arena minus the scan ring.
        hold_survivors: Hand passing documents over with their KV
            pinned instead of freeing or retaining them.
        hold_extra_tokens: Rows past the prefix a held document's
            pages must cover (the consumer's largest frame), so the
            consumer never needs a page this chain did not claim.
        attention_mode: Attention path of this chain's chunks; unified
            when omitted. The warm-up runs the merge_quant path too.
        images: The documents' images, when their prefixes hold image
            soft tokens: open() starts the source, take(doc) gives
            (block, page) pairs for a fresh document in admission
            order, close() releases the source, metrics() reports its
            counters. None for text.
    """

    def __init__(self, torch, arena, pipeline, async_ans, doc_ids,
                 question_ids, budget, timing=None, pinned=True,
                 limit=None, *, arena_writes, arena_keys=None,
                 retain_survivors=(), hold_survivors=False,
                 hold_extra_tokens=0, attention_mode=None, images=None):
        p = _shared_preamble_tokens(question_ids)
        keys = range(len(doc_ids)) if arena_keys is None else arena_keys
        if len(keys) != len(doc_ids):
            raise ValueError("arena_keys must match doc_ids")
        retain_all = retain_survivors is True
        retain = set() if retain_all else set(retain_survivors)
        if hold_survivors and (retain_all or retain):
            raise ValueError("held survivors are pinned, not retained")
        canvas = tuple(pipeline.canvas_ids)
        c = len(canvas)
        # a model whose attention reads paged KV only writes pages even
        # when the plan skipped them
        arena_writes = arena_writes or pipeline.needs_pages
        stage_tokens = [len(question_ids[0]) + c] \
            + [len(q) - p + c for q in question_ids[1:]]
        tails = [question_ids[0]] + [q[p:] for q in question_ids[1:]]
        for i, t in enumerate(tails):
            if not t:
                raise ValueError(
                    f"stage {i} question has no tokens beyond the shared "
                    f"preamble ({p} tokens)")
        if not arena_writes and (
            len(question_ids) > 1 or retain_all or retain or hold_survivors
        ):
            # a later stage re-reads the KV, which needs the pages this
            # switch skips
            raise ValueError("arena_writes=False needs a single stage")
        self.attention_mode = attention_mode or FILTER_ATTENTION
        unified = self.attention_mode == "unified"
        if images is not None:
            if not (unified and arena_writes):
                raise ValueError("images run the paged unified path")
            if not pipeline.takes_images:
                raise ValueError(
                    f"{type(pipeline).__name__} does not embed images")
        self.images = images
        # capacity must cover the longest tail past the kept preamble,
        # and the canvas rows that follow it
        temp_tail = max(0, *(len(q) - p + c for q in question_ids)) \
            if unified and arena_writes else 0
        capacity_extra = p + temp_tail
        if hold_survivors:
            capacity_extra = max(capacity_extra, hold_extra_tokens)
        if arena.retention_cap_pages is None:
            arena.retention_cap_pages = max(
                0, arena.n_pages
                - arena.pages_needed(2 * budget))
        self.sched = FilterAdmission(
            [len(d) for d in doc_ids], stage_tokens, budget,
            arena_pages=(arena.n_pages if arena_writes
                         else None),
            page_tokens=arena.page_tokens,
            kept_extra_tokens=capacity_extra, limit=limit,
            available_pages=(arena.free_pages
                             if arena_writes else None),
            page_cost=arena.page_cost)
        self.torch = torch
        self.arena = arena
        self.pipeline = pipeline
        self.async_ans = async_ans
        self.doc_ids = doc_ids
        self.keys = keys
        self.timing = timing
        self.pinned = pinned
        self.arena_writes = arena_writes
        self.retain_all = retain_all
        self.retain = retain
        self.hold = hold_survivors
        self.preamble = p
        self.capacity_extra = capacity_extra
        self.tails = tails
        self.canvas = canvas
        self.answer_row = pipeline.canvas_answer_row
        self.spans = []
        self.tokens = 0
        self.chunks = 0
        self.done = False
        self.outstanding = []   # (groups, handle) in launch order
        self.progress = Progress(
            f"filter ({len(question_ids)} stages)", total=len(doc_ids))
        self._finished = 0
        self.image_metrics = {}
        if images is not None:
            images.open(budget)

    @property
    def answers(self):
        """Per document, its 0/1 answers up to the first FALSE."""
        return self.sched.answers

    def close(self):
        """Release the image source; its counters stay in image_metrics.

        _finish calls this when the chain completes; a caller that
        abandons the chain early calls it to stop the render processes.
        """
        if self.images is not None:
            self.image_metrics = self.images.metrics()
            self.images.close()
            self.images = None

    def _spec(self, doc, stage, fresh):
        if fresh:
            spec = dict(key=self.keys[doc], prefix=self.doc_ids[doc],
                        f=len(self.doc_ids[doc]), suffixes=[self.tails[0]],
                        write_suffix_tokens=self.preamble)
            if self.images is not None:
                spec["images"] = self.images.take(doc)
            return spec
        return dict(key=self.keys[doc], prefix=None,
                    f=len(self.doc_ids[doc]) + self.preamble,
                    suffixes=[self.tails[stage]])

    def _report(self, entry, items):
        timing = self.timing
        t = time.perf_counter() if timing is not None else 0.0
        groups, handle = entry
        bits = self.async_ans.result(handle)
        t = _tick(timing, "report_wait", t)
        last_stage = len(self.tails) - 1
        for (doc, stage, _fresh), bit in zip(groups, bits):
            passed = bool(bit)
            last = stage == last_stage
            keep = passed and last and (self.retain_all
                                        or doc in self.retain)
            hold = passed and last and self.hold
            for d in self.sched.report(doc, stage, passed,
                                       release=not (keep or hold)):
                self.arena.free_key(self.keys[d])
            if keep:
                freed = self.arena.retain(self.keys[doc],
                                          len(self.doc_ids[doc]))
                self.sched.add_free_pages(freed)
            if hold:
                items.append((self.keys[doc], self.doc_ids[doc]))
            if not passed or last:
                self._finished += 1
        self.progress.update(self._finished)
        _tick(timing, "report_rest", t)

    def _finish(self, items):
        while self.outstanding:
            self._report(self.outstanding.pop(0), items)
        for doc in self.sched.drain_ready():
            self.arena.free_key(self.keys[doc])
        self.close()
        self.progress.finish(
            f"filter ({len(self.tails)} stages) done",
            f"{self.tokens:,} fresh tokens")
        self.done = True

    def next(self, evict_retained=False):
        """Run one chunk. Returns (held (key, prefix) pairs, blocked).

        Reads the previous chunk's answers while the new one runs.
        blocked means fresh admission is short of arena pages; with
        held survivors the consumer has to free some before the chain
        can admit again. evict_retained lets the chain evict retained
        KV of earlier operators for the shortfall instead of returning
        blocked, which a chain without held survivors always does.
        """
        items = []
        if self.done:
            return items, False
        sched = self.sched
        arena = self.arena
        timing = self.timing
        while True:
            if sched.done():
                self._finish(items)
                return items, False
            if self.hold:
                # the consumer frees and claims pages between chunks
                sched.free_pages = arena.free_pages
            t = time.perf_counter() if timing is not None else 0.0
            groups = sched.next_chunk()
            t = _tick(timing, "next_chunk", t)
            if groups:
                break
            if self.outstanding:
                self._report(self.outstanding.pop(0), items)
                continue
            if sched.blocked_pages and (evict_retained or not self.hold):
                before = arena.free_pages
                arena.evict_retained(sched.blocked_pages)
                freed = arena.free_pages - before
                if freed:
                    if not self.hold:
                        sched.add_free_pages(freed)
                    continue
            if self.hold and sched.blocked_pages:
                return items, True
            raise AssertionError("nothing buildable and nothing in flight")
        for doc, stage, fresh in groups:
            if fresh and self.arena_writes:
                logical = len(self.doc_ids[doc]) + self.preamble
                got = arena.activate(
                    self.keys[doc], logical,
                    capacity_tokens=len(self.doc_ids[doc])
                    + self.capacity_extra,
                    base_tokens=len(self.doc_ids[doc]))
                assert got is not None, \
                    "scheduler admitted a doc the arena cannot hold"
        t = _tick(timing, "alloc", t)
        chunk = pack_chunk(self.torch, arena,
                           [self._spec(*g) for g in groups],
                           timing=timing, pinned=self.pinned,
                           attention_mode=self.attention_mode,
                           canvas=self.canvas, answer_row=self.answer_row)
        t = _tick(timing, "pack", t)
        self.tokens += chunk.tokens
        torch = self.torch
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normed = _forward(self.pipeline, arena, chunk)
        e1.record()
        # a fresh document's sliding pages before its window origin
        # were for this pass only
        for doc, _stage, fresh in groups:
            if fresh and self.arena_writes:
                sched.trim(doc, arena.trim_window(self.keys[doc]))
        t = _tick(timing, "forward_launch", t)
        self.spans.append((0, e0, e1))
        self.outstanding.append((groups, self.async_ans.submit(normed)))
        self.chunks += 1
        t = _tick(timing, "submit", t)
        if timing is not None:
            timing["n_chunks"] = timing.get("n_chunks", 0) + 1
        # read the previous chunk's answers while this one runs
        while len(self.outstanding) > 1:
            self._report(self.outstanding.pop(0), items)
        return items, bool(self.hold and sched.blocked_pages)


def run_filter(torch, arena, pipeline, async_ans, doc_ids,
               question_ids, budget, timing=None,
               pinned=True, limit=None, *, arena_writes,
               arena_keys=None, retain_survivors=(), attention_mode=None,
               images=None):
    """The filter chain run to the end; see FilterStream for the arguments.

    Returns:
        (answers, spans, tokens): answers[d] = 0/1 list up to the
        first FALSE; spans and tokens as in run_join.
    """
    stream = FilterStream(
        torch, arena, pipeline, async_ans, doc_ids, question_ids, budget,
        timing=timing, pinned=pinned, limit=limit,
        arena_writes=arena_writes, arena_keys=arena_keys,
        retain_survivors=retain_survivors, attention_mode=attention_mode,
        images=images)
    return run_stream(stream)


def run_stream(stream):
    """Run a FilterStream to the end and return (answers, spans, tokens).

    The stream's image source is closed whether the chain completes or
    raises.
    """
    try:
        while not stream.done:
            stream.next()
    finally:
        stream.close()
    return stream.answers, stream.spans, stream.tokens
