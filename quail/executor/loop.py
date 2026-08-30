"""The overlapped chunk loop: pack on CPU while the GPU runs, gate
answers, free pages immediately.

- run_join: a known pair list, brim-packed by pack_stream.
- run_filter: continuous admission with FilterAdmission, pages freed
  on FALSE or after the last stage.

torch is imported lazily; this module runs only inside the Modal
image.
"""

import time

from quail.executor.attention import FILTER_ATTENTION, JOIN_ATTENTION
from quail.executor.pack import (
    FilterAdmission,
    pack_stream,
    partition_anchor_groups,
)
from quail.executor.retention import RetainedPool


def _tick(timing, key, t0):
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + time.perf_counter() - t0
    return time.perf_counter()


def _staged(torch, data, dtype, pinned=True):
    """Host data to device through pinned memory, non-blocking.

    pinned=False reverts to pageable blocking copies."""
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


def true_false_ids(tok):
    """The token ids that mean TRUE and FALSE."""
    true, false = set(), set()
    for w in ("TRUE", " TRUE", "True", " True"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            true.add(ids[0])
    for w in ("FALSE", " FALSE", "False", " False"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            false.add(ids[0])
    return true, false


class Answerer:
    """TRUE/FALSE from final-position hidden states, scored against
    only the allowed token rows - no full-vocabulary logits."""

    def __init__(self, torch, F, model, tokenizer):
        t_ids, f_ids = true_false_ids(tokenizer)
        self.F = F
        self.allowed = sorted(t_ids | f_ids)
        self.true_ids = t_ids
        # the head weight lives on the CPU when untied (moved there at
        # load); slice where it lives, keep only the slice on the GPU
        weight = model.lm_head.weight
        sel = torch.tensor(self.allowed, device=weight.device)
        self.weights = weight.index_select(0, sel).to(
            device="cuda", dtype=torch.bfloat16)
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

    def margins(self, normed):
        scores = self.F.linear(normed, self.weights)
        t = scores.index_select(1, self.true_cols).amax(dim=1)
        f = scores.index_select(1, self.false_cols).amax(dim=1)
        return (t - f).float().cpu().tolist()


class AsyncAnswers:
    """Non-blocking TRUE/FALSE readout. submit() returns an event and
    pinned host buffer; result() waits on the event and reads the
    answers without stalling the GPU stream."""

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
             stage_suffixes, budget, group_size=None,
             stage_frames=None, anchor_keys=None, anchor_done=None):
    """The join driver: stream partner lists against anchors, gating
    survivors between stages.

    Args:
        anchor_prefixes: Anchor id (list index) -> prefix token list.
        stage_suffixes: Per stage, the partner suffix token lists.
        budget: Chunk token budget.
        group_size: Maximum anchors gated together between stages. None
            uses only the KV capacity limit.
        stage_frames: Per stage, task framing token list written into
            each anchor's kept KV after the document rows.
        anchor_keys: Stable arena key for each anchor. List positions are
            used when omitted.
        anchor_done: Optional callback(anchor position, final answer row).
            It owns the anchor's final retain or free decision.

    Returns:
        (ans, spans, tokens): ans[j][a] = 0/1 row over stage-j
        partners; spans = (stage, start_event, end_event) per forward;
        tokens = fresh tokens packed.
    """
    k = len(stage_suffixes)
    n = len(anchor_prefixes)
    if n == 0 or k == 0:
        return [dict() for _ in range(k)], [], 0
    suffix_lens = [[len(s) for s in sufs] for sufs in stage_suffixes]
    frames = stage_frames or [[] for _ in range(k)]
    keys = list(range(n)) if anchor_keys is None else list(anchor_keys)
    if len(keys) != n:
        raise ValueError("anchor_keys must match anchor_prefixes")
    frame_max = max(len(f) for f in frames)
    groups = partition_anchor_groups(
        [len(prefix) for prefix in anchor_prefixes],
        arena.accounting.n_pages,
        arena.accounting.page_tokens,
        extra_tokens=frame_max,
        max_group_size=group_size)
    group_pages = [sum(
        arena.accounting.pages_needed(
            len(anchor_prefixes[a]) + frame_max)
        for a in members)
        for members in groups]
    ans = [dict() for _ in range(k)]
    spans = []
    tokens = 0

    def plan_stage(members, j):
        live = [a for a in members
                if j == 0 or any(ans[j - 1].get(a, []))]
        if not live:
            return [], [], set()
        # each anchor's FIRST group of the stage carries the frame at
        # the head of its first suffix (written into kept KV there),
        # so the first suffix length is inflated by the frame
        lens = suffix_lens[j]
        if frames[j] and lens:
            lens = [len(frames[j]) + lens[0]] + lens[1:]
        spec = [(len(anchor_prefixes[a]), lens) for a in live]
        keep_loc = set(range(len(live))) if j + 1 < k else set()
        already_loc = {i for i, a in enumerate(live)
                       if keys[a] in arena.accounting.owned}
        plan, to_cache = pack_stream(spec, budget, keep=keep_loc,
                                     already_kept=already_loc)
        return live, plan, to_cache

    def build(j, idx, chunk_groups):
        frame = frames[j]
        specs = []
        for a, start, end, carried in chunk_groups:
            anchor = idx[a]
            key = keys[anchor]
            f = len(anchor_prefixes[anchor])
            if key in arena.accounting.owned or carried:
                got = arena.activate(
                    key, f, capacity_tokens=f + frame_max)
                assert got is not None, "arena underprovisioned"
            sufs = stage_suffixes[j][start:end]
            if frame and start == 0 and sufs:
                # frame entry: scatter the frame into KV after the
                # document rows. The pair entry reads doc + frame.
                # The frame entry's answer bit is skipped by scatter().
                specs.append(dict(
                    key=key,
                    prefix=anchor_prefixes[anchor] if carried else None,
                    f=f, suffixes=[frame],
                    write_suffix_tokens=len(frame)))
                specs.append(dict(
                    key=key, prefix=None, f=f + len(frame),
                    suffixes=sufs))
            else:
                specs.append(dict(
                    key=key,
                    prefix=anchor_prefixes[anchor] if carried else None,
                    f=f + (len(frame) if frame else 0),
                    suffixes=sufs))
        return pack_chunk(torch, arena, specs,
                          attention_mode=pipeline.attention_mode)

    def launch(j, chunk):
        nonlocal tokens
        tokens += chunk["tokens"]
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normed = pipeline.forward_chunk(chunk)
        e1.record()
        spans.append((j, e0, e1))
        return async_ans.submit(normed)

    def scatter(j, idx, chunk_groups, bits):
        pos = 0
        for a, start, end, _ in chunk_groups:
            if frames[j] and start == 0 and end > start:
                pos += 1        # the frame entry's bit means nothing
            cnt = end - start
            ans[j].setdefault(idx[a], []).extend(bits[pos:pos + cnt])
            pos += cnt

    def free_if_owned(anchor):
        key = keys[anchor]
        if key in arena.accounting.owned:
            arena.free_key(key)

    prefetch = None     # (idx, plan, handle0) of the next group's
    #                     stage 0, chunk 0 already launched
    for g, members in enumerate(groups):
        # Every resident key named by this group must remain available
        # while missing group keys evict unrelated retained KV.
        for a in members:
            key = keys[a]
            if key in arena.accounting.owned:
                arena.pin(key)
        for j in range(k):
            if j == 0 and prefetch is not None:
                idx, plan, h0 = prefetch
                prefetch = None
            else:
                idx, plan, _ = plan_stage(members, j)
                h0 = launch(j, build(j, idx, plan[0])) if plan else None
            if not plan:
                continue
            last_chunk = {}
            for t, cg in enumerate(plan):
                for a_l, _, _, _ in cg:
                    last_chunk[a_l] = t
            completed = set()

            def finish(handle, t):
                scatter(j, idx, plan[t], async_ans.result(handle))
                if j == k - 1:
                    for a_l, _, _, _ in plan[t]:
                        anchor = idx[a_l]
                        if last_chunk[a_l] != t or anchor in completed:
                            continue
                        if anchor_done is None:
                            free_if_owned(anchor)
                        else:
                            anchor_done(anchor, ans[j].get(anchor, []))
                        completed.add(anchor)

            handles = [(h0, 0)]
            if (j == 0 and k > 1 and g + 1 < len(groups)
                    and group_pages[g] + group_pages[g + 1]
                    <= arena.accounting.n_pages):
                # the gate below cannot be planned past; keep the
                # GPU fed with the next group's gate-free stage 0
                nidx, nplan, _ = plan_stage(groups[g + 1], 0)
                if nplan:
                    nh = launch(0, build(0, nidx, nplan[0]))
                    prefetch = (nidx, nplan, nh)
            for t in range(1, len(plan)):
                c = build(j, idx, plan[t])   # CPU, GPU busy
                h = launch(j, c)
                h_prev, t_prev = handles.pop(0)
                finish(h_prev, t_prev)
                handles.append((h, t))
            while handles:
                h, t = handles.pop(0)
                finish(h, t)
            # free pages nothing later reads: after the last stage
            # everything in the group is done; between stages, the
            # gate's casualties are done
            if j == k - 1:
                for a in members:
                    if (a not in completed
                            and keys[a] in arena.accounting.owned):
                        if anchor_done is None:
                            free_if_owned(a)
                        else:
                            anchor_done(a, ans[j].get(a, []))
            else:
                for a in idx:
                    if not any(ans[j].get(a, [])):
                        free_if_owned(a)
    return ans, spans, tokens


# ------------------------------------------------------------- warmup
#
# Three cost tiers:
#   1. JIT compile (nvcc/Triton) of a kernel configuration: paid once
#      ever per (software stack, GPU, model, budget) by compile_kernels.
#      Cached on the kernel cache volume (DG_CACHE_DIR / TRITON_CACHE_DIR);
#      a marker file records that the pass ran.
#   2. Loading a cached binary into the process: paid once per container
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
WARMUP_VERSION = 1


def _warm_inputs(budget):
    """Synthetic warmup tokens: a 512-id document and a 16-id
    question suffix, cycled to any length the passes need. Fixed
    small ids; only the counts matter to the kernels."""
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
    """Run real forward passes over every attention path: both modes
    with arena writes, plus the unpaged causal fast path. join_chunk
    adds one run_join call."""
    warm_docs, question, doc = _warm_inputs(budget)
    q_max = len(question)
    original_mode = pipeline.attention_mode
    for mode in (FILTER_ATTENTION, JOIN_ATTENTION):
        pipeline.attention_mode = mode
        run_filter(torch, arena, pipeline, async_ans, warm_docs,
                   [question], budget, arena_writes=True)
        # tiny chunks, one document each: the trailing-chunk shapes
        # of gated multi-stage runs (see TINY_WARM_TOKENS)
        for t in TINY_WARM_TOKENS:
            if t >= budget:
                continue
            body = (doc * (t // len(doc) + 1))[:max(8, t - q_max)]
            run_filter(torch, arena, pipeline, async_ans, [body],
                       [question], budget, arena_writes=True)
    if join_chunk:
        pipeline.attention_mode = JOIN_ATTENTION
        run_join(torch, arena, pipeline, async_ans, warm_docs,
                 [[question] * 8], budget)
    pipeline.attention_mode = original_mode
    run_filter(torch, arena, pipeline, async_ans, warm_docs,
               [question], budget, arena_writes=False)


def compile_kernels(torch, arena, pipeline, async_ans, budget):
    """Build every DeepGEMM kernel configuration up to the budget,
    then run forward passes over every attention-path shape.

    Runs once per (software stack, GPU, model, budget). Uses vLLM's
    config heuristic generator to enumerate every token count at
    which the chosen GEMM configuration changes."""
    from vllm.model_executor.warmup.deep_gemm_warmup import (
        _generate_optimal_warmup_m_values)
    layer = pipeline.layers[0]
    linears = (layer.self_attn.qkv_proj, layer.self_attn.o_proj,
               layer.mlp.gate_up_proj, layer.mlp.down_proj)
    work = [(m, lin) for lin in linears
            for m in _generate_optimal_warmup_m_values(
                budget, lin.weight.shape[0], torch.device("cuda"))]
    try:
        from tqdm import tqdm
        work = tqdm(work, desc="quail kernel compile pass",
                    unit="gemm")
    except ImportError:
        pass
    with torch.inference_mode():
        # one buffer per linear, row-sliced per call; the sweep only
        # needs each kernel launched once
        cur, buf = None, None
        for m, lin in work:
            if lin is not cur:
                buf = torch.randn(budget, lin.weight.shape[1],
                                  device="cuda",
                                  dtype=torch.bfloat16)
                cur = lin
            q, s = pipeline.quant(buf[:m])
            pipeline.gemm(q, s, lin)
        buf = None
        torch.cuda.synchronize()
    _forward_warm(torch, arena, pipeline, async_ans, budget,
                  join_chunk=True)


def touch_kernels(torch, arena, pipeline, async_ans, budget):
    """Run each hot kernel once per container so cached binaries load
    at boot instead of mid-run."""
    _forward_warm(torch, arena, pipeline, async_ans, budget,
                  join_chunk=False)


def _marker_path(model_name, budget):
    import os
    dg = os.environ.get("DG_CACHE_DIR")
    root = (os.path.dirname(dg) if dg
            else os.path.expanduser("~/.cache/quail-kernels"))
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
    """Boot-time warmup: compile pass once per (stack, GPU, model,
    budget), touch pass every container after that.

    A marker file records the identity the compile pass ran for.
    Identity match -> touch; mismatch or absent -> compile.

    Returns dict(tier="compile"|"touch", warm_s=seconds)."""
    import json
    import os

    path = _marker_path(model_name, budget)
    identity = _marker_identity(torch, model_name, budget)
    on_disk = None
    try:
        with open(path) as f:
            on_disk = json.load(f)
    except (OSError, ValueError):
        pass
    t0 = time.perf_counter()
    if force_compile or on_disk != identity:
        compile_kernels(torch, arena, pipeline, async_ans, budget)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(identity, f, indent=1)
        os.replace(tmp, path)
        tier = "compile"
    else:
        touch_kernels(torch, arena, pipeline, async_ans, budget)
        tier = "touch"
    torch.cuda.synchronize()
    return dict(tier=tier, warm_s=round(time.perf_counter() - t0, 2))


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


def run_filter(torch, arena, pipeline, async_ans, doc_ids,
               question_ids, budget, trace=None, timing=None,
               pinned=True, limit=None, *, arena_writes,
               arena_keys=None, retain_survivors=(),
               retention_values=None):
    """The filter chain: continuous admission, survivor priority, pages
    freed on FALSE or after the last stage.

    Args:
        doc_ids: Per-document token lists.
        question_ids: Per-stage question token lists.
        budget: Chunk token budget.
        trace: When given, one dict per chunk is appended with tokens,
            groups, and fresh admission counts.
        timing: CPU seconds per loop phase accumulate into it.
        pinned: False for pageable blocking copies.
        arena_writes: Whether document KV is written to the arena.
            Must be True with multiple stages.
        arena_keys: Stable arena key for each document. List positions are
            used when omitted.
        retain_survivors: Passing document positions to keep for joins.
            Kept KV goes into a RetainedPool capped at the arena minus
            the loop's two-chunk working reservation, so retention
            never starves admission; once full, longer documents
            displace shorter ones by saved recompute per page.
        retention_values: Saved recomputation seconds by document position.

    Returns:
        (answers, spans, tokens): answers[d] = 0/1 list up to the
        first FALSE; spans and tokens as in run_join.
    """
    p = _shared_preamble_tokens(question_ids)
    keys = (list(range(len(doc_ids))) if arena_keys is None
            else list(arena_keys))
    if len(keys) != len(doc_ids):
        raise ValueError("arena_keys must match doc_ids")
    retain = (set(range(len(doc_ids))) if retain_survivors is True
              else set(retain_survivors))
    values = retention_values or {}
    stage_tokens = [len(question_ids[0])] \
        + [len(q) - p for q in question_ids[1:]]
    tails = [question_ids[0]] + [q[p:] for q in question_ids[1:]]
    for i, t in enumerate(tails):
        if not t:
            raise ValueError(
                f"stage {i} question has no tokens beyond the shared "
                f"preamble ({p} tokens)")
    if not arena_writes and (len(question_ids) > 1 or retain):
        # a later stage re-reads the KV, which needs the pages this
        # switch skips
        raise ValueError("arena_writes=False needs a single stage")
    unified = pipeline.attention_mode == "unified"
    # unified scatters each stage's question tail (the tokens past the
    # kept preamble) into the doc's pages; capacity must cover the
    # longest tail, or zero when every question is pure preamble
    temp_tail = max(0, *(len(q) - p for q in question_ids)) \
        if unified and arena_writes else 0
    pool = None
    if retain:
        # The scan ring: the loop keeps up to two chunks of document
        # KV in flight (one running while the next packs), so those
        # pages are reserved before anything is retained, and the pool
        # gets only what is left. Retention then never starves
        # admission, and the filter runs full chunks the whole scan.
        ring_pages = arena.accounting.pages_needed(2 * budget)
        short = ring_pages - arena.accounting.free_pages
        if short > 0:
            # an earlier operator's retained KV crowds the ring:
            # release the least valuable prefixes once, up front
            arena.evict_retained(short)
        pool = RetainedPool(
            max(0, arena.accounting.free_pages - ring_pages))
    sched = FilterAdmission(
        [len(d) for d in doc_ids], stage_tokens, budget,
        arena_pages=(arena.accounting.n_pages if arena_writes
                     else None),
        page_tokens=arena.accounting.page_tokens,
        kept_extra_tokens=p + temp_tail, limit=limit,
        available_pages=(arena.accounting.free_pages
                         if arena_writes else None))
    spans, tokens = [], 0
    outstanding = []     # (groups, handle) in launch order

    def to_spec(doc, stage, fresh):
        if fresh:
            return dict(key=keys[doc], prefix=doc_ids[doc],
                        f=len(doc_ids[doc]), suffixes=[tails[0]],
                        write_suffix_tokens=p)
        return dict(key=keys[doc], prefix=None, f=len(doc_ids[doc]) + p,
                    suffixes=[tails[stage]])

    def report(entry):
        t = time.perf_counter() if timing is not None else 0.0
        groups, handle = entry
        bits = async_ans.result(handle)
        t = _tick(timing, "report_wait", t)
        for (doc, stage, _fresh), bit in zip(groups, bits):
            passed = bool(bit)
            last = stage == len(stage_tokens) - 1
            keep = passed and last and doc in retain
            if keep:
                keep, victims = pool.offer(
                    keys[doc],
                    arena.accounting.pages_needed(len(doc_ids[doc])),
                    float(values.get(doc, 0.0)))
                for v in victims:
                    sched.add_free_pages(arena.free_key(v))
            for d in sched.report(doc, stage, passed,
                                  release=not keep):
                arena.free_key(keys[d])
            if keep:
                arena.retain(keys[doc], len(doc_ids[doc]),
                             float(values.get(doc, 0.0)))
        _tick(timing, "report_rest", t)

    while not sched.done():
        t = time.perf_counter() if timing is not None else 0.0
        groups = sched.next_chunk()
        t = _tick(timing, "next_chunk", t)
        if not groups:
            if outstanding:
                report(outstanding.pop(0))
                continue
            if sched.blocked_pages:
                before = arena.accounting.free_pages
                evicted = arena.evict_retained(sched.blocked_pages)
                if pool is not None:
                    for k in evicted:
                        pool.discard(k)
                freed = arena.accounting.free_pages - before
                if freed:
                    sched.add_free_pages(freed)
                    continue
            raise AssertionError("nothing buildable and nothing in flight")
        for doc, stage, fresh in groups:
            if fresh and arena_writes:
                logical = len(doc_ids[doc]) + p
                got = arena.activate(
                    keys[doc], logical,
                    capacity_tokens=logical + temp_tail)
                assert got is not None, \
                    "scheduler admitted a doc the arena cannot hold"
        t = _tick(timing, "alloc", t)
        chunk = pack_chunk(torch, arena, [to_spec(*g) for g in groups],
                           timing=timing, pinned=pinned,
                           attention_mode=pipeline.attention_mode)
        t = _tick(timing, "pack", t)
        tokens += chunk["tokens"]
        if trace is not None:
            trace.append(dict(
                tokens=chunk["tokens"], groups=len(groups),
                fresh=sum(1 for _, _, f in groups if f)))
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normed = pipeline.forward_chunk(chunk)
        e1.record()
        t = _tick(timing, "forward_launch", t)
        spans.append((0, e0, e1))
        outstanding.append((groups, async_ans.submit(normed)))
        t = _tick(timing, "submit", t)
        if timing is not None:
            timing["n_chunks"] = timing.get("n_chunks", 0) + 1
        # read the previous chunk's answers while this one runs
        while len(outstanding) > 1:
            report(outstanding.pop(0))
    while outstanding:
        report(outstanding.pop(0))
    for doc in sched.drain_ready():
        arena.free_key(keys[doc])
    return sched.answers, spans, tokens
