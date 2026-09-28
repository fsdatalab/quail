"""One scheduler for every model operator on a table's documents.

A model operator asks something of each document: a filter asks a
question, a join asks about each partner, a classification scores
label suffixes. Each of those is a Stage: the frame it writes after
the document, the suffixes it sends, the readout that turns the final
hidden rows into answers, and the decision that reads a document's
answers and says whether it goes on to the next stage.

run_stages drives a list of stages over a set of documents with one
admission (JoinAdmission): a document's head and document tokens are
computed once and stay in KV while it moves through the stages, a
chunk mixes documents at different stages, and a document's pages
go free as soon as a decision drops it or its last stage answers.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from quail.backends.quail.executor.pack import JoinAdmission, partner_pages
from quail.progress import Progress, logger


@dataclass
class Stage:
    """What one model operator asks of each document at one step.

    Attributes:
        suffixes: The stage's request token lists, shared by every
            document; each starts right after the frame.
        readout: Turns final hidden rows into answers: AsyncAnswers
            for TRUE/FALSE bits, AsyncScores for numbers,
            AsyncLabelLogprobs for label log probabilities. Its dtype
            is the answer array type per document; None keeps 0/1
            lists.
        frame: Tokens written into the document's KV right after the
            document before this stage's suffixes. A frame equal to
            the stage before's is already there and is not written
            again.
        requests: Callable(document key) -> indices into suffixes the
            document sends, or None for all of them. None sends every
            suffix to every document.
        decide: Callable(document index, row) -> whether the document
            goes on to the next stage, or survives the last one. None
            takes any true answer.
        read_all_rows: Whether every row of a suffix feeds the readout,
            not only its last.
        label: The stage's name in progress lines.
    """

    suffixes: list
    readout: Any
    frame: list = field(default_factory=list)
    requests: Callable | None = None
    decide: Callable | None = None
    read_all_rows: bool = False
    label: str = ""


def frame_writes(stages) -> list:
    """Per stage, whether its frame is written: not when the stage before wrote it."""
    writes = []
    previous = None
    for stage in stages:
        frame = list(stage.frame)
        writes.append(bool(frame) and frame != previous)
        previous = frame
    return writes


def run_stages(torch, arena, pipeline, stages, anchor_prefixes, budget, *,
               anchor_keys=None, on_settled=None, anchor_source=None,
               anchor_batch=None, staging=None, attention_mode=None,
               prefix_tree=None, stats=None, limit=None, paged=True,
               unit="anchors", label=None):
    """Run every stage over the documents with one admission.

    Args:
        torch: The torch module, imported by the caller.
        arena: KVArena holding the documents' KV pages.
        pipeline: ModelPipeline that runs each packed forward chunk.
        stages: The Stage list, in order.
        anchor_prefixes: Document index -> its prefix token list (head
            and document). With anchor_source it must be an empty
            list; the run appends each streamed document's prefix.
        budget: Chunk token budget.
        anchor_keys: Stable arena key per document; list positions
            when omitted. With anchor_source it must be an empty list.
        on_settled: Optional callable(document index, survived, last
            row) run when a document finishes; it owns the document's
            retain or free decision. Without it a finished document's
            pages are freed.
        anchor_source: Optional stream admitting documents while the
            run goes, as run_join takes it.
        anchor_batch: Optional callable(keys) -> the keys to admit from
            a batch the source hands over.
        staging: Optional reusable input transfer buffers.
        attention_mode: "tree" or "unified" as the plan chose; None
            and a model without tree attention run unified.
        prefix_tree: A PrefixTree over anchor_prefixes, or None.
        stats: When given, receives borrowed_tokens.
        limit: Stop admitting documents once this many survived the
            last stage; the rest never run.
        paged: Whether documents' KV is written to arena pages. False
            runs one stage of one suffix per document as a single
            causal segment without pages: the filter fast path.
        unit: The progress unit.
        label: The progress label; None names the stages.

    Returns:
        (answers, spans, tokens): answers[j][a] = the row of stage j's
        answers for document a, in admission order; spans =
        (stage, start_event, end_event) per forward; tokens = fresh
        tokens packed.
    """
    from quail.backends.quail.executor.loop import (
        ArenaFullError,
        _forward,
        attention_path,
        borrow_check,
        lowest_borrows,
        pack_chunk,
    )

    k = len(stages)
    if k == 0:
        return [], [], 0
    if not paged and (k > 1 or anchor_source is not None
                      or prefix_tree is not None
                      or any(len(stage.suffixes) != 1 for stage in stages)):
        raise ValueError("the unpaged path runs one stage of one suffix")
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
    frames = [list(stage.frame) for stage in stages]
    writes = frame_writes(stages)
    stage_suffixes = [stage.suffixes for stage in stages]
    # a stage's suffixes read their document: tree unless the plan says
    mode = attention_path(pipeline, attention_mode, default="tree")
    canvas = tuple(pipeline.canvas_ids)
    answer_row = pipeline.canvas_answer_row

    def entry_rows(a, j, start, end, carried):
        f = len(prefixes[a])
        sufs = [stage_suffixes[j][i]
                for i in sched.partner_indices(a, j, start, end)]
        rows = (f if carried else 0) + sum(len(s) + len(canvas) for s in sufs)
        if writes[j] and start == 0:
            rows += len(frames[j]) + len(canvas)
        return rows

    # a frame entry's canvas rows land in the document's pages after
    # the frame, so the pages cover them
    frame_max = max(len(f) + (len(canvas) if f else 0) for f in frames)
    if anchor_source is not None and mode == "unified":
        # the source leaves room for one suffix's temporary rows, or
        # held documents could fill the arena before the run can go
        longest = [max((len(s) for s in suffixes), default=0) + len(canvas)
                   for suffixes in stage_suffixes]
        anchor_source.set_reserve(max(
            (partner_pages(arena.page_cost, arena.page_tokens, len(doc),
                           len(frame), rows)
             for doc in anchor_source.doc_ids
             for frame, rows in zip(frames, longest)), default=0))

    def held_pages(key, prefix_tokens):
        # the admission prices a document at page_cost(prefix + frames)
        # less what it holds; a trimmed window holds fewer sliding
        # pages than that price assumes, so count what growing takes
        capacity = prefix_tokens + frame_max
        return (arena.page_cost(capacity)
                - arena.growth_cost(key, capacity))

    resident = ({a: held_pages(keys[a], len(prefixes[a]))
                 for a in range(len(keys)) if arena.is_resident(keys[a])}
                if paged else {})
    # every resident document stays available for the whole run while
    # fresh admissions evict unrelated retained KV
    for a in resident:
        arena.pin(keys[a])

    def requests_of(key):
        lists = [None if stage.requests is None else stage.requests(key)
                 for stage in stages]
        return None if all(lst is None for lst in lists) else lists

    if prefix_tree is not None and not prefix_tree.shared_tokens:
        prefix_tree = None

    def advance(a, j, row):
        decide = stages[j].decide
        if decide is None:
            return bool(any(row) if stages[j].readout.dtype is None
                        else np.any(row))
        return bool(decide(a, row))

    sched = JoinAdmission(
        [len(p) for p in prefixes],
        [[len(s) for s in sufs] for sufs in stage_suffixes],
        budget,
        arena.n_pages if paged else 1 << 62,
        arena.page_tokens,
        frame_tokens=[len(f) for f in frames], resident=resident,
        anchor_partners={a: requests_of(keys[a]) for a in range(len(keys))},
        # a windowed model may pack any chunk unified, so it reserves
        # the unified path's temporary pages throughout
        temporary_suffix_pages=paged and mode == "unified",
        answer_dtype=[stage.readout.dtype for stage in stages],
        canvas_tokens=len(canvas),
        page_cost=arena.page_cost if paged else (lambda *args: 0),
        tree=prefix_tree,
        advance=advance,
        frame_writes=writes,
        limit=limit,
    )
    borrowing = sched.borrowing
    borrowing.can_borrow = borrow_check(arena, keys, borrowing)
    # counted after resident documents detach: they pack no prefix
    borrowers = borrowing.borrowers()
    lowest_borrow = lowest_borrows(borrowing)
    held = [keys[a] for a, n in enumerate(borrowers) if n]
    spans = []
    tokens = 0
    outstanding = []     # (groups, entries, handles) in launch order
    if label is None:
        label = " > ".join(stage.label or f"stage {j}"
                           for j, stage in enumerate(stages))
    counting_answers = unit != "anchors"
    total = (sum(sched._count(a, j) for a in range(len(prefixes))
                 for j in range(k))
             if counting_answers else len(prefixes))
    progress = Progress(
        label, total=None if anchor_source is not None else total, unit=unit)
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
                    f"streamed document {key!r} has no KV in the arena")
            arena.pin(key)
            keys.append(key)
            prefixes.append(prefix)
            sched.admit(len(prefix), held_pages(key, len(prefix)),
                        partners=requests_of(key))

    def pull(evict_retained=False, force=False):
        """Run source chunks until a chunk can fill or the source blocks."""
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
        """Pack one chunk; returns it with (stage, answer rows) per entry."""
        specs = []
        entries = []
        for a, j, start, end, carried in chunk_groups:
            key = keys[a]
            f = len(prefixes[a])
            frame = frames[j]
            parent, shared = ((borrowing.parent(a), borrowing.shared(a))
                              if carried else (None, 0))
            if paged:
                fresh = not arena.is_resident(key)
                got = arena.activate(
                    key, f, capacity_tokens=f + frame_max, base_tokens=f,
                    borrow=(keys[parent], shared) if shared else None)
                assert got is not None, \
                    "the admission placed a document the arena cannot hold"
                if fresh and a < len(borrowers) and borrowers[a]:
                    arena.hold(key, borrowers[a])
                    arena.keep_window(key, lowest_borrow[a])
            prefix = None
            if carried:
                prefix = prefixes[a][shared:] if shared else prefixes[a]
            sufs = [stage_suffixes[j][i]
                    for i in sched.partner_indices(a, j, start, end)]
            read_all = stages[j].read_all_rows
            rows = sum(map(len, sufs)) if read_all else len(sufs)
            if writes[j] and start == 0:
                # frame entry: scatter the frame into KV after the
                # document rows; its own answer row means nothing
                specs.append(dict(
                    key=key, prefix=prefix, start=shared,
                    f=f, suffixes=[frame],
                    write_suffix_tokens=len(frame)))
                entries.append((j, 1))
                specs.append(dict(
                    key=key, prefix=None, f=f + len(frame),
                    suffixes=sufs, read_all_rows=read_all))
            else:
                specs.append(dict(
                    key=key, prefix=prefix, start=shared,
                    f=f + len(frame),
                    suffixes=sufs, read_all_rows=read_all))
            entries.append((j, rows))
        chunk = pack_chunk(torch, arena, specs, attention_mode=mode,
                           staging=staging, canvas=canvas,
                           answer_row=answer_row)
        return chunk, entries

    def settle(anchor):
        survived = sched._true[anchor][k - 1]
        if on_settled is None:
            if arena.is_resident(keys[anchor]):
                arena.free_key(keys[anchor])
        else:
            on_settled(anchor, survived, sched.answers[k - 1].get(anchor, []))

    def event(kind, anchor):
        if kind == "finished":
            settle(anchor)
            finished[0] += 1
        elif arena.is_resident(keys[anchor]):
            arena.free_key(keys[anchor])

    def submit(normed, entries, rows_per_answer):
        """Hand each stage's answer rows to its readout; returns handles."""
        by_stage = {}
        row = 0
        for j, rows in entries:
            by_stage.setdefault(j, []).append((row, row + rows))
            row += rows
        if len(by_stage) == 1:
            (j,) = by_stage
            readout = stages[j].readout
            return {j: (readout.submit(normed, rows_per_answer=rows_per_answer)
                        if rows_per_answer else readout.submit(normed))}
        handles = {}
        per_answer = list(rows_per_answer) if rows_per_answer else None
        for j, spans_j in by_stage.items():
            index = torch.tensor(
                [r for start, end in spans_j for r in range(start, end)],
                device=normed.device)
            part = normed.index_select(0, index)
            readout = stages[j].readout
            if per_answer is None:
                handles[j] = readout.submit(part)
            else:
                # answers of this stage, in entry order
                counts = []
                answer = 0
                row = 0
                for jj, rows in entries:
                    taken = 0
                    while taken < rows:
                        taken += per_answer[answer]
                        if jj == j:
                            counts.append(per_answer[answer])
                        answer += 1
                handles[j] = readout.submit(part, rows_per_answer=counts)
        return handles

    def report(entry):
        groups, entries, handles = entry
        values = {j: stages[j].readout.result(handle)
                  for j, handle in handles.items()}
        pos = {j: 0 for j in handles}
        for a, j, start, end, _ in groups:
            if writes[j] and start == 0:
                pos[j] += 1        # the frame entry's answer means nothing
            cnt = end - start
            for kind, anchor in sched.report(
                    a, j, start, end, values[j][pos[j]:pos[j] + cnt]):
                event(kind, anchor)
            pos[j] += cnt
        progress.update(
            progress.done + sum(end - start for _, _, start, end, _ in groups)
            if counting_answers else finished[0])

    def run_part(part):
        """Run one chunk of groups, halving it when its pages do not fit."""
        try:
            run_one(part)
        except ArenaFullError as error:
            # retained KV nothing here reads makes room first
            need = arena.page_cost(sum(entry_rows(*e) for e in part))
            if arena.evict_retained(need):
                logger.info("chunk of %d groups retried after evicting "
                            "retained KV", len(part))
                run_part(part)
                return
            if len(part) < 2:
                raise
            logger.info("chunk of %d groups split: %s", len(part), error)
            half = len(part) // 2
            run_part(part[:half])
            run_part(part[half:])

    def run_one(part):
        nonlocal tokens
        chunk, entries = build(part)
        tokens += chunk.tokens
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normed = _forward(pipeline, arena, chunk)
        e1.record()
        for key in chunk.fresh_keys:
            arena.trim_window(key)
        spans.append((part[0][1], e0, e1))
        handles = submit(normed, entries, chunk.rows_per_answer)
        outstanding.append((part, entries, handles))
        # read the previous chunk's answers while this one runs
        while len(outstanding) > 1:
            report(outstanding.pop(0))

    while True:
        if anchor_source is not None and not anchor_source.done:
            pull()
        # documents with nothing to send at their first stage never run
        for kind, anchor in sched.take_settled():
            event(kind, anchor)
        if sched.done() and (anchor_source is None
                             or anchor_source.done
                             or sched.limit_reached()):
            break
        if sched.blocked_pages:
            # the free list is short for the next fresh document:
            # retained KV nothing here reads makes room
            arena.evict_retained(sched.blocked_pages)
        groups = sched.next_chunk(arena.free_pages if paged else 1 << 62)
        if not groups:
            if outstanding:
                report(outstanding.pop(0))
                continue
            if anchor_source is not None and not anchor_source.done:
                # the source has to move; it may evict retained KV to admit
                if pull(evict_retained=True, force=True):
                    continue
            if sched.blocked_pages and arena.evict_retained(sched.blocked_pages):
                continue
            # parents freed but held for queued children hold the
            # pages: the children pack their whole prefixes
            if arena.drop_holds(held):
                continue
            raise AssertionError("nothing buildable and nothing in flight")
        run_part(groups)
        # the packed children hold their parents' pages now
        for a, _, start, _, carried in groups:
            parent = (borrowing.tree_parent[a]
                      if a < len(borrowing.tree_parent) else None)
            if carried and start == 0 and parent is not None:
                arena.release(keys[parent])
    while outstanding:
        report(outstanding.pop(0))
    # a limit ends the run with documents still queued
    for anchor in sched.drain():
        if arena.is_resident(keys[anchor]):
            arena.free_key(keys[anchor])
    progress.finish(f"{label} done", f"{tokens:,} fresh tokens")
    arena.drop_holds(held)
    if stats is not None:
        stats["borrowed_tokens"] = borrowing.borrowed_tokens
    return sched.answers, spans, tokens

