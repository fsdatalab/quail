"""Run filter chains and joins through the overlapped stage scheduler.

These adapters turn operator inputs into stages. The shared helpers select
attention paths and manage the arena pages used by each forward pass.
"""


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
    """Stream each stage's selected partner requests against anchor documents.

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
        A tuple of per-stage partner answers, chunk timing spans, and fresh
        token count. Each stage maps anchor indices to answer rows for their
        selected partners. Timing spans use the format from run_stages().
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
    """Run a sequence of Boolean filters with the shared stage scheduler.

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
        A tuple of per-document Boolean answers, chunk timing spans, and
        fresh token count. Each document's answers end at its first failed
        filter. Timing spans use the format returned by run_stages().
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
