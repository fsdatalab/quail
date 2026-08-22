"""The overlapped chunk loop: pack on CPU while the GPU runs, gate the
moment answers land, free pages the instant nothing later can read
them. Two drivers over one substrate:

- run_join: a known pair list, brim-packed by pack_stream (the
  exploration's measured join loop, arena-backed).
- run_filter: continuous admission (FilterAdmission) - the filter
  chain as the degenerate join: anchor = the document, suffix = the
  question, stage j+1 attaches a fresh suffix to the same kept KV.
  No question KV is ever written except the shared question preamble,
  which joins the anchor KV after stage 1 (exactly what chain mode's
  rewind kept resident).

torch is imported lazily; this module runs only inside the Modal
image.
"""

import time

from quail.executor.pack import FilterAdmission, pack_stream


def _tick(timing, key, t0):
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + time.perf_counter() - t0
    return time.perf_counter()


def _staged(torch, data, dtype, pinned=True):
    """Host data -> device through pinned memory, non-blocking.

    torch.tensor(list, device='cuda') from pageable memory blocks the
    CPU until the stream drains the chunk still running; the pinned
    stage never does. The pinned tensor may be dropped right away:
    the caching host allocator defers its reuse until the copy's
    stream event fires.

    pinned=False reverts to the pageable blocking copies - the
    ablation ladder's pre-#12 rung (issue #9 measured what the pinned
    stage is worth)."""
    if torch.is_tensor(data):
        if pinned:
            return data.pin_memory().to("cuda", non_blocking=True)
        return data.to("cuda")
    if pinned:
        return torch.tensor(data, dtype=dtype, pin_memory=True).to(
            "cuda", non_blocking=True)
    return torch.tensor(data, dtype=dtype, device="cuda")


def true_false_ids(tok):
    """The token ids that mean TRUE and FALSE. Constraining the answer
    to their union makes every stage answer in exactly one token: the
    answer is read from the prefill pass and no decode step ever
    runs."""
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
        sel = torch.tensor(self.allowed, device="cuda")
        self.weights = model.lm_head.weight.index_select(0, sel).to(
            torch.bfloat16)
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
    """TRUE/FALSE readout that does not stall the stream: submit()
    computes the bits on GPU, enqueues a copy to pinned host memory,
    and records an event; result() waits only for that event - ops
    enqueued after the event (the next chunk's forward) keep the GPU
    busy while the CPU reads the answers."""

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

def pack_chunk(torch, arena, groups, timing=None, pinned=True):
    """Tensors for one chunk, built from groups in chunk order.

    Each group is a dict:
      key       arena key for KV writes and kept-context reads
      prefix    fresh prefix token list, packed into the chunk - or
                None when the key's KV is already in the arena
      f         kept-context length: suffix positions start here, and
                call B reads this many arena rows (for a fresh group
                this is len(prefix); for a kept group, whatever the
                arena holds for the key - document plus any kept
                preamble)
      suffixes  list of suffix token lists (may be empty for a
                cache-only group)
      write_suffix_tokens   scatter this many leading rows of the
                FIRST suffix into the key's pages at offset
                len(prefix): the shared question preamble joining the
                anchor KV after stage 1

    A fresh prefix is scattered into its pages when the key owns pages
    (allocated by the caller before packing); a fresh group without
    pages runs self-attention only (the probe's unpacked reference).
    """
    t = time.perf_counter() if timing is not None else 0.0
    ids, pos, cu_a, finals = [], [], [0], []
    suffix_rows = []
    kv_writes, layout = [], []
    cross_keys, cross_used, cu_q = [], [], [0]
    max_q = 0
    for g in groups:
        fresh = g.get("prefix") is not None
        f = g["f"]
        key = g["key"]
        paged = key in arena.accounting.owned
        row0 = len(ids)
        if fresh:
            ids.extend(g["prefix"])
            pos.extend(range(len(g["prefix"])))
            cu_a.append(len(ids))
            if paged:
                kv_writes.append((key, row0, row0 + len(g["prefix"]),
                                  0))
        s_row0 = len(ids)
        for si, suf in enumerate(g["suffixes"]):
            srow = len(ids)
            ids.extend(suf)
            pos.extend(range(f, f + len(suf)))
            cu_a.append(len(ids))
            finals.append(len(ids) - 1)
            wst = g.get("write_suffix_tokens", 0)
            if si == 0 and wst and paged:
                # the shared question preamble joins the kept KV right
                # after the document rows: after the fresh prefix, or
                # after a restored document's f rows
                dest = len(g["prefix"]) if fresh else f
                kv_writes.append((key, srow, srow + wst, dest))
        s_count = len(ids) - s_row0
        layout.append((key, len(g["suffixes"])))
        if s_count and f and paged:
            suffix_rows.extend(range(s_row0, len(ids)))
            cu_q.append(cu_q[-1] + s_count)
            cross_keys.append(key)
            cross_used.append(f)
            max_q = max(max_q, s_count)

    t = _tick(timing, "pack_py", t)
    cross = None
    if cross_keys:
        table, _ = arena.block_table(cross_keys)
        t = _tick(timing, "pack_blocktable", t)
        used = _staged(torch, cross_used, torch.int32, pinned)
        cu_k = _staged(torch, [0] + list(_cumsum(cross_used)),
                       torch.int32, pinned)
        cross = dict(
            rows=_staged(torch, suffix_rows, torch.int64, pinned),
            cu_q=_staged(torch, cu_q, torch.int32, pinned),
            max_q=max_q, keys=cross_keys, used=used,
            max_used=max(cross_used), table=table, cu_k=cu_k)
    t = _tick(timing, "pack_cross", t)

    # all of the chunk's KV writes as one list of (source, destination)
    # row pairs: the attention pass scatters them with a single kernel
    # launch per layer (kv_row_scatter)
    kv_src = kv_dst = None
    if kv_writes:
        src = []
        for _, r0, r1, _ in kv_writes:
            src.extend(range(r0, r1))
        kv_src = _staged(torch, src, torch.int64, pinned)
        kv_dst = _staged(torch, torch.cat(
            [arena._rows[key][dest:dest + (r1 - r0)]
             for key, r0, r1, dest in kv_writes]), torch.int64, pinned)
    t = _tick(timing, "pack_kv", t)
    meta = dict(
        layer=0, kv_src=kv_src, kv_dst=kv_dst, cross=cross, paged=True,
        cu_a=_staged(torch, cu_a, torch.int32, pinned),
        max_a=max(cu_a[i + 1] - cu_a[i] for i in range(len(cu_a) - 1)))
    out = dict(
        input_ids=_staged(torch, ids, torch.int64, pinned),
        positions=_staged(torch, pos, torch.int64, pinned),
        final_indices=_staged(torch, finals, torch.int64, pinned),
        meta=meta, tokens=len(ids), layout=layout)
    _tick(timing, "pack_h2d", t)
    return out


def _cumsum(xs):
    total = 0
    for x in xs:
        total += x
        yield total


# ------------------------------------------------------------ the join

def run_join(torch, arena, pipeline, async_ans, anchor_prefixes,
             stage_suffixes, budget, group_size=None, store=None,
             store_hash=None, store_min_tokens=1, store_ids=None,
             stats=None, stage_frames=None):
    """The join driver: every stage streams a partner list against the
    anchor side; gated anchors advance between stages.

    anchor_prefixes: anchor id (list index) -> prefix token list.
    stage_suffixes: per stage, the partner suffix token lists.
    budget: the chunk token budget.
    group_size: anchors gated together between stages; None = all
    anchors in one group (right for one stage, where no gate runs).
    stage_frames: per stage, the task framing token list, written
    into each anchor's kept KV right after the document rows (the
    filter's shared-question-preamble mechanism). Every pair suffix
    of that stage reads the frame from KV instead of carrying its
    tokens, so framing costs tokens per anchor, not per pair. A
    later stage's frame overwrites the earlier one's rows - each
    stage's pairs attend only their own frame. The frame rows are
    NOT part of the stored prefix, so the store stays
    query-independent.

    Pipelining, one rule: while the GPU runs a chunk, the CPU builds
    the next buildable one. Within a stage that is the next chunk of
    the plan; at a gate, whose next chunk cannot be built until the
    group's answers arrive, it is the next group's stage-0 chunk
    (which depends on nothing) - launched before the gate resolves,
    so the GPU never drains. Answers travel as event-synced pinned
    copies (AsyncAnswers); an anchor's prefix KV is written to its
    arena pages exactly once and the pages free when its group leaves
    its last stage. Suffix KV is never written anywhere.

    store / store_hash / store_min_tokens / store_ids: the pinned KV
    store, exactly run_filter's contract. An anchor whose prefix is
    already stored restores into its pages instead of computing (its
    chunks carry only partner suffixes); an anchor leaving its last
    stage - gated out or finished - is copied out on the side stream
    when its prefix is at least store_min_tokens long, its pages
    returned only after the copy lands. store_ids maps anchor list
    positions to stable store key ids (the anchor's global document
    index, the same key a filter scan of the same corpus uses).
    stats, when given, is filled with restored/stored counts.

    Returns (ans, spans, tokens): ans[j][a] = 0/1 row over stage-j
    partners, present only for anchors that reached stage j; spans =
    (stage, start_event, end_event) per forward for GPU-time sums;
    tokens = fresh tokens packed.
    """
    k = len(stage_suffixes)
    n = len(anchor_prefixes)
    if n == 0:
        # An upstream filter chain can legitimately reduce the anchor
        # side to nothing before a join runs (e.g. a restrictive
        # multi-filter chain with no survivors). group_size would
        # otherwise fall back to n (0), and range(0, 0, 0) is a
        # ValueError - "arg 3 must not be zero" - not an empty range.
        # Zero anchors means zero pairs, unconditionally: nothing to
        # pack, launch, or gate.
        if stats is not None:
            stats.update(restored_docs=0, restored_tokens=0,
                         stored_docs=0, stored_tokens=0)
        return [dict() for _ in range(k)], [], 0
    group_size = n if group_size is None else group_size
    groups = [list(range(i, min(i + group_size, n)))
              for i in range(0, n, group_size)]
    suffix_lens = [[len(s) for s in sufs] for sufs in stage_suffixes]
    frames = stage_frames or [[] for _ in range(k)]
    frame_max = max(len(f) for f in frames)
    ans = [dict() for _ in range(k)]
    spans = []
    tokens = 0

    def skey(a):
        return (store_hash, store_ids[a] if store_ids else a)

    restored = set()
    if store is not None:
        restored = {a for a in range(n) if skey(a) in store}
    load_events = {}     # anchor -> store load event, awaited pre-launch
    pending_saves = []   # (anchor, event): pages held until the copy lands
    saving = set()       # anchors with an in-flight save: drain frees them
    save_after = None    # an event recorded after the newest forward;
    #                      any such event orders a save behind the
    #                      compute that wrote the anchor's pages
    if stats is not None:
        stats.update(restored_docs=len(restored),
                     restored_tokens=sum(len(anchor_prefixes[a])
                                         for a in restored),
                     stored_docs=0, stored_tokens=0)

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
        # restored anchors never pack prefix tokens: their KV loads
        # from the store when their pages allocate
        already_loc = {i for i, a in enumerate(live)
                       if a in arena.accounting.owned or a in restored}
        plan, to_cache = pack_stream(spec, budget, keep=keep_loc,
                                     already_kept=already_loc)
        return live, plan, to_cache

    def drain_saves(block=False):
        rest = []
        for a, ev in pending_saves:
            if block:
                ev.synchronize()
            if ev.query():
                arena.free_key(a)
                saving.discard(a)
            else:
                rest.append((a, ev))
        pending_saves[:] = rest

    def build(j, idx, chunk_groups):
        frame = frames[j]
        specs = []
        for a, start, end, carried in chunk_groups:
            key = idx[a]
            f = len(anchor_prefixes[key])
            if key not in arena.accounting.owned \
                    and (carried or key in restored):
                got = arena.alloc(key, f + frame_max)
                if got is None and pending_saves:
                    # pages held only by in-flight store saves
                    drain_saves(block=True)
                    got = arena.alloc(key, f + frame_max)
                assert got is not None, "arena underprovisioned"
                if not carried:
                    load_events[key] = store.load(skey(key), arena, key)
            sufs = stage_suffixes[j][start:end]
            if frame and start == 0 and sufs:
                # the anchor's first group of this stage: the frame
                # gets its own entry whose rows are written after the
                # document (dest = f, overwriting any earlier stage's
                # frame), and the pair entry in the same chunk reads
                # doc + frame - the same write-then-read the fresh
                # document + question path already does. The frame
                # entry's answer bit is skipped by scatter().
                specs.append(dict(
                    key=key,
                    prefix=anchor_prefixes[key] if carried else None,
                    f=f, suffixes=[frame],
                    write_suffix_tokens=len(frame)))
                specs.append(dict(
                    key=key, prefix=None, f=f + len(frame),
                    suffixes=sufs))
            else:
                specs.append(dict(
                    key=key,
                    prefix=anchor_prefixes[key] if carried else None,
                    f=f + (len(frame) if frame else 0),
                    suffixes=sufs))
        return pack_chunk(torch, arena, specs)

    def launch(j, chunk):
        nonlocal tokens, save_after
        tokens += chunk["tokens"]
        for key, _ in chunk["layout"]:
            ev = load_events.pop(key, None)
            if ev is not None:
                torch.cuda.current_stream().wait_event(ev)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normed = pipeline.forward_chunk(chunk)
        e1.record()
        spans.append((j, e0, e1))
        save_after = e1
        return async_ans.submit(normed)

    def scatter(j, idx, chunk_groups, bits):
        pos = 0
        for a, start, end, _ in chunk_groups:
            if frames[j] and start == 0 and end > start:
                pos += 1        # the frame entry's bit means nothing
            cnt = end - start
            ans[j].setdefault(idx[a], []).extend(bits[pos:pos + cnt])
            pos += cnt

    def free_if_owned(key):
        """Free an anchor's pages - after copying them to the store
        when it qualifies (long enough, not already stored). Pages
        with an in-flight save are freed by drain_saves instead."""
        if key not in arena.accounting.owned or key in saving:
            return
        if (store is not None and save_after is not None
                and len(anchor_prefixes[key]) >= store_min_tokens
                and skey(key) not in store):
            ev = store.save(skey(key), arena, key,
                            len(anchor_prefixes[key]),
                            after_event=save_after)
            if ev is not None:
                saving.add(key)
                pending_saves.append((key, ev))
                if stats is not None:
                    stats["stored_docs"] += 1
                    stats["stored_tokens"] += len(anchor_prefixes[key])
                return
        arena.free_key(key)

    prefetch = None     # (idx, plan, handle0) of the next group's
    #                     stage 0, chunk 0 already launched
    for g, members in enumerate(groups):
        for j in range(k):
            drain_saves()
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

            def finish(handle, t):
                scatter(j, idx, plan[t], async_ans.result(handle))
                if j == k - 1:
                    # a cut anchor's pages have no reader past its
                    # last chunk of its final stage; freeing at stage
                    # end instead held ~76 anchors' KV (the measured
                    # 48.3 GiB peak in the exploration)
                    for a_l, _, _, _ in plan[t]:
                        if last_chunk[a_l] == t:
                            free_if_owned(idx[a_l])

            handles = [(h0, 0)]
            if j == 0 and k > 1 and g + 1 < len(groups):
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
                    free_if_owned(a)
            else:
                for a in idx:
                    if not any(ans[j][a]):
                        free_if_owned(a)
    drain_saves(block=True)
    return ans, spans, tokens


# ------------------------------------------------------------- warmup

def warm_kernels(torch, arena, pipeline, async_ans, doc_ids,
                 question_ids, budget):
    """Compile every kernel configuration a run can hit, at boot,
    outside measured walls (the protocol measures with kernels
    compiled).

    DeepGEMM picks a kernel configuration per token count and
    JIT-compiles each one on first sight (~3-10 s per configuration).
    The configuration (its JIT-debug log shows block widths like 112)
    varies with the token count FINER than powers of two - a
    geometric sweep left ~41 s of compile lumps inside the first
    measured run - so the sweep is dense at small counts and steps
    at large ones, the same reason vLLM's DeepGEMM warmup iterates
    the token dimension. Compiled artifacts land in the kernel cache
    (a volume in the Modal images), so all of this costs real time
    once per software stack, ever.

    After the sweep, one budget-sized chunk warms the Triton kernels
    and the attention path. doc_ids are cycled, so small corpora
    still warm full-size shapes."""
    layer = pipeline.layers[0]
    linears = (layer.self_attn.qkv_proj, layer.self_attn.o_proj,
               layer.mlp.gate_up_proj, layer.mlp.down_proj)
    # granular at every size, including the big ones: DeepGEMM's
    # config choice (its debug log shows block widths like 112)
    # tracks the token count finer than powers of two, and the
    # measured lumps came from mid-size drain chunks, so there is no
    # "big sizes are covered by the sweep" shortcut
    sizes = sorted(
        {m for m in range(64, 4097, 256)}
        | {m for m in range(4096, 32769, 1024)}
        | {m for m in range(32768, budget + 1, 2048)}
        | {budget})
    work = [(m, lin) for m in sizes for lin in linears]
    try:
        from tqdm import tqdm
        work = tqdm(work, desc="quail kernel warmup", unit="gemm")
    except ImportError:
        pass
    with torch.inference_mode():
        for m, lin in work:
            x = torch.randn(m, lin.weight.shape[1], device="cuda",
                            dtype=torch.bfloat16)
            q, s = pipeline.quant(x)
            pipeline.gemm(q, s, lin)
        torch.cuda.synchronize()

    q_max = max(len(q) for q in question_ids)
    warm_docs, used = [], 0
    i = 0
    while True:
        d = doc_ids[i % len(doc_ids)]
        if used + len(d) + q_max > budget:
            break
        warm_docs.append(d)
        used += len(d) + q_max
        i += 1
    run_filter(torch, arena, pipeline, async_ans, warm_docs,
               question_ids, budget)


# ---------------------------------------------------------- the filter

def _shared_preamble_tokens(question_ids):
    """Longest common token prefix across the stage questions - the
    part chain mode kept resident between rewinds."""
    if len(question_ids) < 2:
        return 0
    p = 0
    while all(len(q) > p and q[p] == question_ids[0][p]
              for q in question_ids):
        p += 1
    return p


def run_filter(torch, arena, pipeline, async_ans, doc_ids,
               question_ids, budget, trace=None, store=None,
               store_hash=None, store_min_tokens=1, stats=None,
               store_ids=None, timing=None, pinned=True,
               limit=None):
    """The filter chain on the packed executor: continuous admission,
    survivor priority, pages freed on NO or after the last stage.

    doc_ids: per-document token lists (the planted flag line included).
    question_ids: per-stage question token lists, planner order.
    trace: optional list; when given, one dict of chunk composition
    (tokens, groups, fresh admissions) is appended per chunk, aligned
    with spans - the shape data for chasing per-chunk anomalies.

    store / store_hash / store_min_tokens: the pinned KV store. A
    document already in the store restores into its pages instead of
    computing (its stage-1 chunk carries only the question); a
    document leaving its last stage is copied out on the side stream
    when it is at least store_min_tokens long, its pages returned
    only after the copy lands. stats, when given, is filled with
    restored/stored counts. store_ids maps local document positions
    to stable store key ids (a sharded worker keys by global index,
    so its store slice survives across queries).

    timing: optional dict; when given, CPU seconds per loop phase
    (next_chunk, alloc, pack and its sub-phases, forward launch,
    submit, report wait/scatter, drain_saves) accumulate into it.
    Timing is host-side only and does not change what runs.

    pinned: pass False to build chunk tensors with pageable blocking
    copies (the pre-#12 path; the ablation ladder's staging rung).

    Returns (answers, spans, tokens): answers[d] = 0/1 list up to the
    first NO (gated); spans and tokens as in run_join.
    """
    p = _shared_preamble_tokens(question_ids)
    stage_tokens = [len(question_ids[0])] \
        + [len(q) - p for q in question_ids[1:]]
    tails = [question_ids[0]] + [q[p:] for q in question_ids[1:]]
    def skey(d):
        return (store_hash, store_ids[d] if store_ids else d)

    restored = set()
    if store is not None:
        restored = {d for d in range(len(doc_ids))
                    if skey(d) in store}
    sched = FilterAdmission(
        [len(d) for d in doc_ids], stage_tokens, budget,
        arena_pages=arena.accounting.n_pages,
        page_tokens=arena.accounting.page_tokens,
        kept_extra_tokens=p, restored=restored, limit=limit)
    spans, tokens = [], 0
    outstanding = []     # (groups, handle) in launch order
    load_events = {}     # doc -> store load event, awaited pre-launch
    pending_saves = []   # (doc, event): pages held until the copy lands
    if stats is not None:
        stats.update(restored_docs=len(restored),
                     restored_tokens=sum(len(doc_ids[d])
                                         for d in restored),
                     stored_docs=0, stored_tokens=0)

    def to_spec(doc, stage, fresh):
        if fresh:
            if doc in restored:
                # KV loads from the store; stage 1 is question-only
                return dict(key=doc, prefix=None,
                            f=len(doc_ids[doc]), suffixes=[tails[0]],
                            write_suffix_tokens=p)
            return dict(key=doc, prefix=doc_ids[doc],
                        f=len(doc_ids[doc]), suffixes=[tails[0]],
                        write_suffix_tokens=p)
        return dict(key=doc, prefix=None, f=len(doc_ids[doc]) + p,
                    suffixes=[tails[stage]])

    def drain_saves(block=False):
        t = time.perf_counter() if timing is not None else 0.0
        rest = []
        for doc, ev in pending_saves:
            if block:
                ev.synchronize()
            if ev.query():
                arena.free_key(doc)
                sched.release(doc)
            else:
                rest.append((doc, ev))
        pending_saves[:] = rest
        _tick(timing, "drain_saves", t)

    def report(entry):
        t = time.perf_counter() if timing is not None else 0.0
        groups, handle = entry
        bits = async_ans.result(handle)
        t = _tick(timing, "report_wait", t)
        for (doc, stage, _fresh), bit in zip(groups, bits):
            yes = bool(bit)
            last = stage == len(stage_tokens) - 1
            leaving = (not yes) or last
            save = (leaving and store is not None
                    and len(doc_ids[doc]) >= store_min_tokens
                    and skey(doc) not in store)
            if save:
                ev = store.save(skey(doc), arena, doc,
                                len(doc_ids[doc]),
                                after_event=handle[0])
                if ev is not None:
                    sched.report(doc, stage, yes, release=False)
                    pending_saves.append((doc, ev))
                    if stats is not None:
                        stats["stored_docs"] += 1
                        stats["stored_tokens"] += len(doc_ids[doc])
                    continue
            sched.report(doc, stage, yes)
            if doc not in sched.resident \
                    and doc in arena.accounting.owned:
                arena.free_key(doc)
        _tick(timing, "report_rest", t)

    while not sched.done():
        drain_saves()
        t = time.perf_counter() if timing is not None else 0.0
        groups = sched.next_chunk()
        t = _tick(timing, "next_chunk", t)
        if not groups:
            if outstanding:
                report(outstanding.pop(0))
            else:
                # nothing in flight: only pending saves hold pages
                assert pending_saves, \
                    "nothing buildable and nothing in flight"
                drain_saves(block=True)
            continue
        for doc, stage, fresh in groups:
            if fresh:
                got = arena.alloc(doc, len(doc_ids[doc]) + p)
                assert got is not None, \
                    "scheduler admitted a doc the arena cannot hold"
                if doc in restored:
                    load_events[doc] = store.load(skey(doc), arena,
                                                  doc)
        t = _tick(timing, "alloc", t)
        chunk = pack_chunk(torch, arena, [to_spec(*g) for g in groups],
                           timing=timing, pinned=pinned)
        t = _tick(timing, "pack", t)
        tokens += chunk["tokens"]
        if trace is not None:
            trace.append(dict(
                tokens=chunk["tokens"], groups=len(groups),
                fresh=sum(1 for _, _, f in groups if f)))
        for doc, _, fresh in groups:
            if fresh and doc in load_events:
                torch.cuda.current_stream().wait_event(
                    load_events.pop(doc))
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
    drain_saves(block=True)
    return sched.answers, spans, tokens
