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

from quail.executor.attention import FILTER_ATTENTION, JOIN_ATTENTION
from quail.executor.model import answer_weights
from quail.executor.pack import FilterAdmission, JoinAdmission
from quail.logical import true_false_ids
from quail.progress import Progress, logger, quiet


def _tick(timing, key, t0):
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + time.perf_counter() - t0
    return time.perf_counter()


def _staged(torch, data, dtype, pinned=True):
    """Host data to device through pinned memory, non-blocking.

    pinned=False reverts to pageable blocking copies.
    """
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


def _staged_token_parts(torch, sequences, total, pinned=True):
    """Copy Arrow token views into one GPU input tensor."""
    host = torch.empty(total, dtype=torch.int64, pin_memory=pinned)
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
            else:
                source = torch.as_tensor(part)
            host[offset:offset + count].copy_(source)
            offset += count
    if offset != total:
        raise AssertionError(
            f"packed {offset} token ids into a {total}-token chunk")
    return host.to("cuda", non_blocking=pinned)


class Answerer:
    """TRUE/FALSE from final-position hidden states.

    Scored against only the allowed token rows - no full-vocabulary
    logits.
    """

    def __init__(self, torch, F, model, tokenizer):
        t_ids, f_ids = true_false_ids(tokenizer)
        self.F = F
        self.allowed = sorted(t_ids | f_ids)
        self.true_ids = t_ids
        self.weights = answer_weights(model, self.allowed)
        self.true_cols = torch.tensor(
            [i for i, t in enumerate(self.allowed) if t in t_ids],
            device="cuda")
        self.false_cols = torch.tensor(
            [i for i, t in enumerate(self.allowed) if t in f_ids],
            device="cuda")

    def __call__(self, normed):
        scores = self.F.linear(normed, self.weights)
        t = scores.index_select(1, self.true_cols).amax(dim=1)
        f = scores.index_select(1, self.false_cols).amax(dim=1)
        return (t > f).int().cpu().tolist()

class AsyncAnswers:
    """Non-blocking TRUE/FALSE readout.

    submit() returns an event and pinned host buffer; result() waits on
    the event and reads the answers without stalling the GPU stream.
    """

    def __init__(self, torch, answerer):
        self.torch = torch
        self.ans = answerer

    def submit(self, normed):
        torch, ans = self.torch, self.ans
        scores = ans.F.linear(normed, ans.weights)
        t = scores.index_select(1, ans.true_cols).amax(dim=1)
        f = scores.index_select(1, ans.false_cols).amax(dim=1)
        bits = (t > f).to(torch.uint8)
        host = torch.empty(bits.shape[0], dtype=torch.uint8,
                           pin_memory=True)
        host.copy_(bits, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return event, host

    @staticmethod
    def result(handle):
        event, host = handle
        event.synchronize()
        return [int(b) for b in host.tolist()]


# ------------------------------------------------------- chunk packing

def pack_chunk(torch, arena, groups, timing=None, pinned=True, *,
               attention_mode):
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
    """
    t = time.perf_counter() if timing is not None else 0.0
    # unified scatters every fresh row through its own src/dst map, so
    # the cross and kv_writes bookkeeping below is two-call only
    two_call = attention_mode != "unified"
    id_parts, token_count = [], 0
    pos, cu_a, finals = [], [0], []
    suffix_rows = []
    kv_writes, layout = [], []
    cross_keys, cross_used, cu_q = [], [], [0]
    max_q = 0
    unified_groups = []
    for g in groups:
        fresh = g.get("prefix") is not None
        f = g["f"]
        key = g["key"]
        paged = key in arena.accounting.owned
        row0 = token_count
        if fresh:
            if not paged and len(g["suffixes"]) > 1:
                raise ValueError(
                    f"group {key!r}: an unpaged fresh group packs "
                    f"[prefix | suffix] as one causal segment, which "
                    f"only one suffix may join - allocate pages or "
                    f"split the group")
            id_parts.append(g["prefix"])
            token_count += len(g["prefix"])
            pos.extend(range(len(g["prefix"])))
            if paged or not g["suffixes"]:
                cu_a.append(token_count)
            if paged and two_call:
                kv_writes.append((key, row0, row0 + len(g["prefix"]),
                                  0))
        prefix_end = token_count
        s_row0 = token_count
        suffix_spans = []
        for si, suf in enumerate(g["suffixes"]):
            srow = token_count
            id_parts.append(suf)
            token_count += len(suf)
            pos.extend(range(f, f + len(suf)))
            suffix_spans.append((srow, token_count))
            cu_a.append(token_count)
            finals.append(token_count - 1)
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
            cu_q=_staged(torch, cu_q, torch.int32, pinned),
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
        kv_page_rows = []
        kv_lengths = []
        cu_q = [0]
        activation_src = []
        kv_dst = []
        tail_src = []
        tail_dst = []

        page_tokens = arena.accounting.page_tokens

        def add_sequence(pages, kv_tokens, query_tokens):
            kv_page_rows.append(pages)
            kv_lengths.append(kv_tokens)
            cu_q.append(cu_q[-1] + query_tokens)

        try:
            for spec in unified_groups:
                key = spec["key"]
                f = spec["f"]
                r0, r1 = spec["row0"], spec["row1"]
                spans = spec["suffix_spans"]
                logical_start = 0 if spec["fresh"] else f
                rows = arena._capacity_rows[key]
                count = r1 - r0
                direct = len(spans) == 1 \
                    and logical_start + count <= rows.numel()
                if direct:
                    activation_src.extend(range(r0, r1))
                    kv_dst.extend(
                        rows[logical_start:logical_start + count].tolist())
                    add_sequence(
                        arena.accounting.owned[key],
                        f + sum(e - s for s, e in spans), count)
                    continue

                prefix_count = spec["prefix_end"] - r0
                if prefix_count:
                    activation_src.extend(range(r0, spec["prefix_end"]))
                    kv_dst.extend(rows[:prefix_count].tolist())
                    prefix_pages = arena.accounting.owned[key][
                        :arena.accounting.pages_needed(f)]
                    add_sequence(prefix_pages, f, prefix_count)

                anchor_pages = arena.accounting.owned[key]
                for s0, s1 in spans:
                    suffix_tokens = s1 - s0
                    remainder = f % page_tokens
                    got = arena.alloc_temporary(remainder + suffix_tokens)
                    if got is None:
                        raise RuntimeError(
                            "unified suffix pages exceed the free KV arena; "
                            "split the chunk")
                    temp_key, temp_pages = got
                    temporary_keys.append(temp_key)
                    temp_rows = arena._capacity_rows[temp_key]
                    activation_src.extend(range(s0, s1))
                    kv_dst.extend(
                        temp_rows[remainder:remainder + suffix_tokens]
                        .tolist())
                    if remainder:
                        anchor_page = anchor_pages[f // page_tokens]
                        base = anchor_page * page_tokens
                        tail_src.extend(range(base, base + remainder))
                        tail_dst.extend(temp_rows[:remainder].tolist())
                    full_pages = f // page_tokens
                    add_sequence(
                        anchor_pages[:full_pages] + temp_pages,
                        f + suffix_tokens, suffix_tokens)

            table = arena.block_table_rows(kv_page_rows)
            unified = dict(
                src=_staged(torch, activation_src, torch.int64, pinned),
                dst=_staged(torch, kv_dst, torch.int64, pinned),
                tail_src=(_staged(torch, tail_src, torch.int64, pinned)
                          if tail_src else None),
                tail_dst=(_staged(torch, tail_dst, torch.int64, pinned)
                          if tail_dst else None),
                cu_q=_staged(torch, cu_q, torch.int32, pinned),
                used=_staged(torch, kv_lengths, torch.int32, pinned),
                table=table,
                max_q=max(b - a for a, b in zip(cu_q, cu_q[1:])),
                max_used=max(kv_lengths))
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
            rows = arena._capacity_rows[key][dest:dest + (r1 - r0)]
            if rows.numel() != r1 - r0:
                raise AssertionError("KV write exceeds reserved rows")
            dst_parts.append(rows)
        kv_src = _staged(torch, src, torch.int64, pinned)
        kv_dst = _staged(torch, torch.cat(dst_parts), torch.int64,
                         pinned)
    t = _tick(timing, "pack_kv", t)

    meta = dict(
        layer=0, kv_src=kv_src, kv_dst=kv_dst, cross=cross,
        unified=unified,
        cu_a=_staged(torch, cu_a, torch.int32, pinned),
        max_a=max(cu_a[i + 1] - cu_a[i] for i in range(len(cu_a) - 1)))
    out = dict(
        input_ids=_staged_token_parts(
            torch, id_parts, token_count, pinned),
        positions=_staged(torch, pos, torch.int64, pinned),
        final_indices=_staged(torch, finals, torch.int64, pinned),
        meta=meta, tokens=token_count, layout=layout)
    if temporary_keys:
        out["temporary_keys"] = temporary_keys
    _tick(timing, "pack_h2d", t)
    return out


# ------------------------------------------------------------ the join

def run_join(torch, arena, pipeline, async_ans, anchor_prefixes,
             stage_suffixes, budget, stage_frames=None,
             anchor_keys=None, anchor_done=None, anchor_source=None,
             attention_mode=None):
    """The join driver: stream partner lists against anchors.

    Survivors are gated between stages.

    Args:
        torch: The torch module, imported by the caller.
        arena: KVArena holding the anchors' KV pages.
        pipeline: Pipeline that runs each packed forward chunk.
        async_ans: AsyncAnswers that reads TRUE/FALSE off the GPU.
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
        attention_mode: Attention path set on the pipeline before each
            join chunk; the pipeline's current mode when omitted.

    Returns:
        (ans, spans, tokens): ans[j][a] = 0/1 row over stage-j
        partners, with a in admission order; spans = (stage,
        start_event, end_event) per forward; tokens = fresh tokens
        packed.
    """
    k = len(stage_suffixes)
    if anchor_source is not None:
        if anchor_prefixes or anchor_keys:
            raise ValueError(
                "anchor_source fills anchor_prefixes and anchor_keys")
        prefixes = anchor_prefixes if isinstance(anchor_prefixes, list) \
            else []
        keys = anchor_keys if isinstance(anchor_keys, list) else []
    else:
        prefixes = list(anchor_prefixes)
        keys = (list(range(len(prefixes))) if anchor_keys is None
                else list(anchor_keys))
    if len(keys) != len(prefixes):
        raise ValueError("anchor_keys must match anchor_prefixes")
    frames = stage_frames or [[] for _ in range(k)]
    owned = arena.accounting.owned

    if k == 0 or not stage_suffixes[0] \
            or (anchor_source is None and not prefixes):
        # no tuple to evaluate. A streamed source still runs its
        # chain to the end, and every anchor it hands over settles
        # with an empty row at once.
        if anchor_source is not None:
            while not anchor_source.done:
                items, _ = anchor_source.next(evict_retained=True)
                for key, prefix in items:
                    keys.append(key)
                    prefixes.append(prefix)
                    if anchor_done is None:
                        arena.free_key(key)
                    else:
                        anchor_done(len(keys) - 1, [])
        return [dict() for _ in range(k)], [], 0
    frame_max = max(len(f) for f in frames)
    resident = {a: len(owned[keys[a]]) for a in range(len(keys))
                if keys[a] in owned}
    # Every resident anchor stays available for the whole join while
    # fresh admissions evict unrelated retained KV.
    for a in resident:
        arena.pin(keys[a])
    sched = JoinAdmission(
        [len(p) for p in prefixes],
        [[len(s) for s in sufs] for sufs in stage_suffixes],
        budget, arena.accounting.n_pages, arena.accounting.page_tokens,
        frame_tokens=[len(f) for f in frames], resident=resident)
    spans = []
    tokens = 0
    outstanding = []     # (groups, handle) in launch order
    progress = Progress(
        f"join ({k} stages)",
        total=None if anchor_source is not None else len(prefixes),
        unit="anchors")
    finished = [0]

    def admit(items):
        for key, prefix in items:
            if key not in owned:
                raise ValueError(
                    f"streamed anchor {key!r} has no KV in the arena")
            arena.pin(key)
            keys.append(key)
            prefixes.append(prefix)
            sched.admit(len(prefix), len(owned[key]))

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
            got = arena.activate(key, f, capacity_tokens=f + frame_max)
            assert got is not None, \
                "scheduler admitted an anchor the arena cannot hold"
            sufs = stage_suffixes[j][start:end]
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
        return pack_chunk(torch, arena, specs,
                          attention_mode=pipeline.attention_mode)

    def settle(anchor):
        if anchor_done is None:
            if keys[anchor] in owned:
                arena.free_key(keys[anchor])
        else:
            anchor_done(anchor, sched.answers[k - 1].get(anchor, []))

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
                if kind == "finished":
                    settle(anchor)
                    finished[0] += 1
                elif keys[anchor] in owned:
                    arena.free_key(keys[anchor])
            pos += cnt
        progress.update(finished[0])

    while True:
        if anchor_source is not None and not anchor_source.done:
            pull()
        if sched.done() and (anchor_source is None
                             or anchor_source.done):
            break
        if sched.blocked_pages:
            # the free list is short for the next fresh anchor:
            # retained KV nothing here reads makes room
            arena.evict_retained(sched.blocked_pages)
        groups = sched.next_chunk(arena.accounting.free_pages)
        if not groups:
            if outstanding:
                report(outstanding.pop(0))
                continue
            if anchor_source is not None and not anchor_source.done:
                # nothing to run here: the source has to move, and it
                # may evict retained KV to admit again
                if pull(evict_retained=True, force=True):
                    continue
            raise AssertionError("nothing buildable and nothing in flight")
        if attention_mode is not None:
            pipeline.attention_mode = attention_mode
        chunk = build(groups)
        tokens += chunk["tokens"]
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normed = pipeline.forward_chunk(chunk)
        e1.record()
        spans.append((groups[0][1], e0, e1))
        outstanding.append((groups, async_ans.submit(normed)))
        # read the previous chunk's answers while this one runs
        while len(outstanding) > 1:
            report(outstanding.pop(0))
    while outstanding:
        report(outstanding.pop(0))
    progress.finish(f"join ({k} stages) done", f"{tokens:,} fresh tokens")
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

# Chunk sizes (tokens) the tiny-chunk warmup ladder builds. A gated
# chain's trailing chunks are 100-500 tokens, a shape the full-size
# warm chunks do not cover, so each needs its own compile.
TINY_WARM_TOKENS = (64, 128, 256, 512, 1024, 2048)

# Bump when either pass covers a different set of shapes. A bumped
# version invalidates every marker, so the next boot re-runs the
# compile pass and re-commits the cache.
WARMUP_VERSION = 2


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
    original_mode = pipeline.attention_mode
    for mode in (FILTER_ATTENTION, JOIN_ATTENTION):
        logger.debug("kernels: warming %s attention, full chunk", mode)
        pipeline.attention_mode = mode
        run_filter(torch, arena, pipeline, async_ans, warm_docs,
                   [question], budget, arena_writes=True)
        # tiny chunks, one document each: the trailing-chunk shapes
        # of gated multi-stage runs (see TINY_WARM_TOKENS)
        for t in TINY_WARM_TOKENS:
            if t >= budget:
                continue
            logger.debug("kernels: warming %s attention, %s tokens", mode, t)
            body = (doc * (t // len(doc) + 1))[:max(8, t - q_max)]
            run_filter(torch, arena, pipeline, async_ans, [body],
                       [question], budget, arena_writes=True)
    if join_chunk:
        logger.debug("kernels: warming join forward pass")
        pipeline.attention_mode = JOIN_ATTENTION
        run_join(torch, arena, pipeline, async_ans, warm_docs,
                 [[question] * 8], budget)
    pipeline.attention_mode = original_mode
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
    from vllm.model_executor.warmup.deep_gemm_warmup import (
        _generate_optimal_warmup_m_values,
    )
    layer = pipeline.layers[0]
    linears = (layer.self_attn.qkv_proj, layer.self_attn.o_proj,
               layer.mlp.gate_up_proj, layer.mlp.down_proj)
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
            q, s = pipeline.quant(buf[:m])
            pipeline.gemm(q, s, lin)
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
        pipeline: Pipeline that runs each packed forward chunk.
        async_ans: AsyncAnswers that reads TRUE/FALSE off the GPU.
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
        attention_mode: Attention path set on the pipeline before each
            chunk; the pipeline's current mode when omitted. A chain
            interleaved with a join on one pipeline states its own.
    """

    def __init__(self, torch, arena, pipeline, async_ans, doc_ids,
                 question_ids, budget, timing=None, pinned=True,
                 limit=None, *, arena_writes, arena_keys=None,
                 retain_survivors=(), hold_survivors=False,
                 hold_extra_tokens=0, attention_mode=None):
        p = _shared_preamble_tokens(question_ids)
        keys = range(len(doc_ids)) if arena_keys is None else arena_keys
        if len(keys) != len(doc_ids):
            raise ValueError("arena_keys must match doc_ids")
        retain_all = retain_survivors is True
        retain = set() if retain_all else set(retain_survivors)
        if hold_survivors and (retain_all or retain):
            raise ValueError("held survivors are pinned, not retained")
        stage_tokens = [len(question_ids[0])] \
            + [len(q) - p for q in question_ids[1:]]
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
        self.attention_mode = attention_mode or pipeline.attention_mode
        unified = self.attention_mode == "unified"
        # unified scatters each stage's question tail (the tokens past
        # the kept preamble) into the doc's pages; capacity must cover
        # the longest tail, or zero when every question is pure preamble
        temp_tail = max(0, *(len(q) - p for q in question_ids)) \
            if unified and arena_writes else 0
        capacity_extra = p + temp_tail
        if hold_survivors:
            capacity_extra = max(capacity_extra, hold_extra_tokens)
        if arena.accounting.retention_cap_pages is None:
            arena.accounting.retention_cap_pages = max(
                0, arena.accounting.n_pages
                - arena.accounting.pages_needed(2 * budget))
        self.sched = FilterAdmission(
            [len(d) for d in doc_ids], stage_tokens, budget,
            arena_pages=(arena.accounting.n_pages if arena_writes
                         else None),
            page_tokens=arena.accounting.page_tokens,
            kept_extra_tokens=capacity_extra, limit=limit,
            available_pages=(arena.accounting.free_pages
                             if arena_writes else None))
        self.torch = torch
        self.arena = arena
        self.pipeline = pipeline
        self.async_ans = async_ans
        self.doc_ids = doc_ids
        self.keys = keys
        self.budget = budget
        self.timing = timing
        self.pinned = pinned
        self.arena_writes = arena_writes
        self.retain_all = retain_all
        self.retain = retain
        self.hold = hold_survivors
        self.preamble = p
        self.capacity_extra = capacity_extra
        self.stage_tokens = stage_tokens
        self.tails = tails
        self.spans = []
        self.tokens = 0
        self.chunks = 0
        self.done = False
        self.held = []          # positions handed over, in order
        self.outstanding = []   # (groups, handle) in launch order
        self.progress = Progress(
            f"filter ({len(question_ids)} stages)", total=len(doc_ids))
        self._finished = 0

    @property
    def answers(self):
        """Per document, its 0/1 answers up to the first FALSE."""
        return self.sched.answers

    def _spec(self, doc, stage, fresh):
        if fresh:
            return dict(key=self.keys[doc], prefix=self.doc_ids[doc],
                        f=len(self.doc_ids[doc]), suffixes=[self.tails[0]],
                        write_suffix_tokens=self.preamble)
        return dict(key=self.keys[doc], prefix=None,
                    f=len(self.doc_ids[doc]) + self.preamble,
                    suffixes=[self.tails[stage]])

    def _report(self, entry, items):
        timing = self.timing
        t = time.perf_counter() if timing is not None else 0.0
        groups, handle = entry
        bits = self.async_ans.result(handle)
        t = _tick(timing, "report_wait", t)
        last_stage = len(self.stage_tokens) - 1
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
                self.held.append(doc)
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
                sched.sync_free_pages(arena.accounting.free_pages)
            t = time.perf_counter() if timing is not None else 0.0
            groups = sched.next_chunk()
            t = _tick(timing, "next_chunk", t)
            if groups:
                break
            if self.outstanding:
                self._report(self.outstanding.pop(0), items)
                continue
            if sched.blocked_pages and (evict_retained or not self.hold):
                before = arena.accounting.free_pages
                arena.evict_retained(sched.blocked_pages)
                freed = arena.accounting.free_pages - before
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
                    + self.capacity_extra)
                assert got is not None, \
                    "scheduler admitted a doc the arena cannot hold"
        t = _tick(timing, "alloc", t)
        self.pipeline.attention_mode = self.attention_mode
        chunk = pack_chunk(self.torch, arena,
                           [self._spec(*g) for g in groups],
                           timing=timing, pinned=self.pinned,
                           attention_mode=self.attention_mode)
        t = _tick(timing, "pack", t)
        self.tokens += chunk["tokens"]
        torch = self.torch
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normed = self.pipeline.forward_chunk(chunk)
        e1.record()
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
               arena_keys=None, retain_survivors=()):
    """The filter chain run to the end; see FilterStream for the arguments.

    Returns:
        (answers, spans, tokens): answers[d] = 0/1 list up to the
        first FALSE; spans and tokens as in run_join.
    """
    stream = FilterStream(
        torch, arena, pipeline, async_ans, doc_ids, question_ids, budget,
        timing=timing, pinned=pinned, limit=limit,
        arena_writes=arena_writes, arena_keys=arena_keys,
        retain_survivors=retain_survivors)
    while not stream.done:
        stream.next()
    return stream.answers, stream.spans, stream.tokens
