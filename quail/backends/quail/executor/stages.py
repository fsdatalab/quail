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

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from quail.backends.quail.executor.pack import DROP, SKIP, JoinAdmission
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
            document sends, None for all of them, DROP to take the
            document out of the run at this stage, or SKIP to pass it
            on to the next stage with nothing asked. Asked when the
            document reaches the stage. None sends every suffix to
            every document.
        decide: Callable(document index, row) -> whether the document
            goes on to the next stage, or survives the last one. None
            takes any true answer.
        read_all_rows: Whether every row of a suffix feeds the readout,
            not only its last.
        read_rows: Per suffix, how many of its last rows feed the
            readout; None reads by ``read_all_rows``. A pair
            classification's suffixes carry a partner document before
            the label path and read the path's rows only.
        single: The stage sends its one suffix to every document, as a
            filter does: the frame and the suffix pack as one entry
            written straight into the document's pages, and under tree
            attention a borrowing document reads its parent's pages
            stacked with its siblings.
        label: The stage's name in progress lines.
        chains: For a stage whose one suffix packs a label trie, the
            chains from ``trie_chains``: the packer makes each a causal
            segment and gathers the ancestors above it.
        canvas: For a diffusion model's denoising step, Callable(document
            index) -> (token ids, conditioning row): the canvas the
            document packs after each suffix in place of the pipeline's,
            and the first of its rows in the run's conditioning rows.
            Asked when the document's group is packed. None packs the
            pipeline's canvas.
        canvas_rows: The rows of every canvas ``canvas`` returns.
        step: For a denoising step, its number, which sets the step's
            temperature.
    """

    DROP = DROP
    SKIP = SKIP

    suffixes: list
    readout: Any
    frame: list = field(default_factory=list)
    requests: Callable | None = None
    decide: Callable | None = None
    read_all_rows: bool = False
    read_rows: Any = None
    single: bool = False
    label: str = ""
    chains: list | None = None
    canvas: Callable | None = None
    canvas_rows: int = 0
    step: int = 0


def shared_preamble_tokens(question_ids) -> int:
    """Longest common token prefix across the stage questions."""
    if len(question_ids) < 2:
        return 0
    p = 0
    while all(len(q) > p and q[p] == question_ids[0][p]
              for q in question_ids):
        p += 1
    return p


def filter_stages(question_ids, readout) -> list:
    """One Stage per filter question: the shared preamble as a frame, then the tail.

    The preamble common to every question is written into the
    document's KV once, at the first stage; each stage's suffix is its
    question past the preamble.

    Raises:
        ValueError: A question has no tokens past the shared preamble.
    """
    p = shared_preamble_tokens(question_ids)
    frame = list(question_ids[0][:p])
    stages = []
    for index, question in enumerate(question_ids):
        tail = list(question[p:])
        if not tail:
            raise ValueError(
                f"stage {index} question has no tokens beyond the shared "
                f"preamble ({p} tokens)")
        stages.append(Stage(suffixes=[tail], readout=readout, frame=frame,
                            single=True, label=f"filter stage {index}"))
    return stages


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
               anchor_keys=None, on_settled=None, staging=None,
               attention_mode=None, prefix_tree=None, stats=None,
               limit=None, paged=True, unit="anchors", label=None,
               default_attention="tree", on_chunk=None, conditioning=None):
    """Run every stage over the documents with one admission.

    Args:
        torch: The torch module, imported by the caller.
        arena: KVArena holding the documents' KV pages.
        pipeline: ModelPipeline that runs each packed forward chunk.
        stages: The Stage list, in order.
        anchor_prefixes: Document index -> its prefix token list (head
            and document).
        budget: Chunk token budget.
        anchor_keys: Stable arena key per document; list positions
            when omitted.
        on_settled: Optional callable(document index, survived, last
            row) run when a document finishes; it owns the document's
            retain or free decision. Without it a finished document's
            pages are freed.
        staging: Optional reusable input transfer buffers.
        attention_mode: "tree" or "unified" as the plan chose; None
            and a model without tree attention run unified.
        prefix_tree: A PrefixTree over anchor_prefixes, or None.
        stats: When given, receives borrowed_tokens and pack_s, the
            host seconds spent building chunks.
        limit: Stop admitting documents once this many survived the
            last stage; the rest never run.
        paged: Whether documents' KV is written to arena pages. False
            runs one stage of one suffix per document as a single
            causal segment without pages: the filter fast path.
        unit: The progress unit.
        label: The progress label; None names the stages.
        default_attention: The attention path when the plan chose none
            and the model has tree attention.
        on_chunk: Optional callable run after each chunk's answers are
            read with the documents that completed a stage in it, as
            (document index, stage, passed) tuples: passed says the
            document went on to the next stage, or survived the last.
        conditioning: For denoising steps, the ConditioningRows every
            stage's canvas names rows in; None otherwise.

    Returns:
        (answers, spans, tokens): answers[j][a] = the row of stage j's
        answers for document a, in admission order; spans =
        (stage, start_event, end_event) per forward; tokens = fresh
        tokens packed.
    """
    from quail.backends.quail.executor.loop import (
        ArenaFullError,
        Suffixes,
        _forward,
        attention_path,
        borrow_check,
        lowest_borrows,
        pack_chunk,
    )

    k = len(stages)
    if k == 0:
        return [], [], 0
    if not paged and (k > 1 or prefix_tree is not None
                      or any(len(stage.suffixes) != 1 for stage in stages)):
        raise ValueError("the unpaged path runs one stage of one suffix")
    prefixes = anchor_prefixes
    keys = (list(range(len(prefixes))) if anchor_keys is None
            else anchor_keys)
    if len(keys) != len(prefixes):
        raise ValueError("anchor_keys must match anchor_prefixes")
    frames = [list(stage.frame) for stage in stages]
    frame_ids = [np.asarray(frame, dtype=np.int64) for frame in frames]
    writes = frame_writes(stages)
    suffixes = [Suffixes.of(stage.suffixes) for stage in stages]
    read_rows = [None if stage.read_rows is None
                 else np.asarray(stage.read_rows, dtype=np.int64)
                 for stage in stages]
    # a stage's suffixes read their document: tree unless the plan says;
    # a stage packing chains needs the tree path whatever the plan said
    if any(stage.chains is not None for stage in stages):
        attention_mode = "tree"
    mode = attention_path(pipeline, attention_mode, default=default_attention)
    canvas = tuple(pipeline.canvas_ids)
    answer_row = pipeline.canvas_answer_row
    # per stage, the canvas rows after each suffix
    canvas_rows = [len(canvas) if stage.canvas is None else stage.canvas_rows
                   for stage in stages]

    def entry_rows(a, j, start, end, carried):
        f = len(prefixes[a])
        indices = sched.partner_indices(a, j, start, end)
        lengths = suffixes[j].lengths_at(indices)
        rows = ((f if carried else 0) + int(lengths.sum())
                + canvas_rows[j] * len(lengths))
        if writes[j] and start == 0:
            rows += len(frames[j]) + len(canvas)
        return rows

    # a frame entry's canvas rows land in the document's pages after
    # the frame, so the pages cover them
    frame_max = max(len(f) + (len(canvas) if f else 0) for f in frames)

    # a stage sending one suffix to every document writes it straight
    # into the document's own pages on the unified path, so the pages
    # cover the frame and that suffix; other stages' suffixes take
    # temporary pages
    capacity_extra = frame_max
    if paged and mode == "unified":
        capacity_extra = max(
            [frame_max] + [len(frame) + len(stage.suffixes[0]) + rows
                           for frame, stage, rows in zip(frames, stages,
                                                         canvas_rows)
                           if stage.single])

    def held_pages(key, prefix_tokens):
        # the admission prices a document at page_cost(prefix + extra)
        # less what it holds; a trimmed window holds fewer sliding
        # pages than that price assumes, so count what growing takes
        capacity = prefix_tokens + capacity_extra
        return (arena.page_cost(capacity)
                - arena.growth_cost(key, capacity))

    resident = ({a: held_pages(keys[a], len(prefixes[a]))
                 for a in range(len(keys)) if arena.is_resident(keys[a])}
                if paged else {})
    # every resident document stays available for the whole run while
    # fresh admissions evict unrelated retained KV
    for a in resident:
        arena.pin(keys[a])

    asking = any(stage.requests is not None for stage in stages)

    def requests_of(key):
        """The document's per-stage requests, asked as it reaches each stage."""
        if not asking:
            return None
        return lambda j: (None if stages[j].requests is None
                          else stages[j].requests(key))

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
        [(s.lengths + rows).tolist() for s, rows in zip(suffixes, canvas_rows)],
        budget,
        arena.n_pages if paged else 1 << 62,
        arena.page_tokens,
        frame_tokens=[len(f) for f in frames], resident=resident,
        anchor_partners={a: requests_of(keys[a]) for a in range(len(keys))},
        # a windowed model may pack any chunk unified, so it reserves
        # the unified path's temporary pages throughout
        temporary_suffix_pages=paged and mode == "unified",
        answer_dtype=[stage.readout.dtype for stage in stages],
        frame_canvas_tokens=len(canvas),
        page_cost=arena.page_cost if paged else (lambda *args: 0),
        tree=prefix_tree,
        advance=advance,
        frame_writes=writes,
        limit=limit,
        extra_tokens=capacity_extra,
    )
    borrowing = sched.borrowing
    borrowing.can_borrow = borrow_check(arena, keys, borrowing)
    # counted after resident documents detach: they pack no prefix
    borrowers = borrowing.borrowers()
    lowest_borrow = lowest_borrows(borrowing)
    held = [keys[a] for a, n in enumerate(borrowers) if n]
    spans = []
    tokens = 0
    pack_s = 0.0
    outstanding = []     # (groups, entries, handles) in launch order
    if label is None:
        label = " > ".join(stage.label or f"stage {j}"
                           for j, stage in enumerate(stages))
    counting_answers = unit != "anchors"
    total = (sum(sched._count(a, j) for a in range(len(prefixes))
                 for j in range(k))
             if counting_answers else len(prefixes))
    progress = Progress(label, total=total, unit=unit)
    finished = [0]

    def merged(j, start, end):
        """Whether a group's frame rides its suffix as a single entry."""
        return stages[j].single and end - start == 1

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
                    key, f, capacity_tokens=f + capacity_extra,
                    base_tokens=f,
                    borrow=(keys[parent], shared) if shared else None)
                assert got is not None, \
                    "the admission placed a document the arena cannot hold"
                if fresh and a < len(borrowers) and borrowers[a]:
                    arena.hold(key, borrowers[a])
                    arena.keep_window(key, lowest_borrow[a])
            prefix = None
            if carried:
                prefix = prefixes[a][shared:] if shared else prefixes[a]
            indices = sched.partner_indices(a, j, start, end)
            sufs = suffixes[j].take(indices)
            read_all = stages[j].read_all_rows
            rows = int(sufs.lengths.sum()) if read_all else len(sufs)
            own = {}
            if read_rows[j] is not None:
                own["read_rows"] = read_rows[j][np.asarray(list(indices))]
                rows = int(own["read_rows"].sum())
            if stages[j].canvas is not None:
                ids, row = stages[j].canvas(a)
                own = dict(canvas=ids, conditioning=row)
                if read_all:
                    rows = stages[j].canvas_rows * len(sufs)
            # under tree attention a borrowing document with one
            # suffix reads its parent's pages stacked with its siblings
            read_key = keys[parent] if shared and stages[j].single else None
            if writes[j] and start == 0 and merged(j, start, end):
                # the frame and the one suffix are one entry; the
                # frame's rows are scattered into KV after the document
                specs.append(dict(
                    key=key, prefix=prefix, start=shared, read_key=read_key,
                    f=f, suffixes=Suffixes(
                        np.concatenate([frame_ids[j], sufs.ids]),
                        [len(frame) + int(sufs.lengths[0])]),
                    write_suffix_tokens=len(frame), single=True,
                    read_all_rows=read_all, **own))
            elif writes[j] and start == 0:
                # frame entry: scatter the frame into KV after the
                # document rows; its own answer row means nothing
                specs.append(dict(
                    key=key, prefix=prefix, start=shared, read_key=read_key,
                    f=f, suffixes=Suffixes(frame_ids[j], [len(frame)]),
                    write_suffix_tokens=len(frame)))
                entries.append((j, 1))
                specs.append(dict(
                    key=key, prefix=None, f=f + len(frame),
                    suffixes=sufs, read_all_rows=read_all,
                    chains=stages[j].chains, **own))
            else:
                specs.append(dict(
                    key=key, prefix=prefix, start=shared, read_key=read_key,
                    f=f + len(frame),
                    suffixes=sufs, read_all_rows=read_all,
                    single=stages[j].single and end - start == 1,
                    chains=stages[j].chains, **own))
            entries.append((j, rows))
        chunk = pack_chunk(
            torch, arena, specs, attention_mode=mode, staging=staging,
            canvas=canvas, answer_row=answer_row,
            conditioning=None if conditioning is None else conditioning.rows)
        return chunk, entries

    def settle(anchor):
        survived = sched._true[anchor][k - 1]
        if on_settled is None:
            if arena.is_resident(keys[anchor]):
                arena.free_key(keys[anchor])
        else:
            on_settled(anchor, survived, sched.answers[k - 1].get(anchor, []))

    def event(kind, anchor):
        # progress counts settled documents: a dropped one is done too
        finished[0] += 1
        if kind == "finished":
            settle(anchor)
        elif arena.is_resident(keys[anchor]):
            arena.free_key(keys[anchor])

    shared_readout = all(stage.readout is stages[0].readout
                         for stage in stages)
    # stages sharing a readout submit together: a chunk holds documents
    # at many stages (every denoising step is one), and one submit per
    # stage costs a row selection each
    groups_of = {}
    group_key = {}
    for j, stage in enumerate(stages):
        group_key[j] = (None if shared_readout else groups_of.setdefault(
            (id(stage.readout), stage.read_all_rows), len(groups_of)))
    readout_of = {group_key[j]: stages[j].readout for j in range(len(stages))}
    read_all_of = {group_key[j]: stages[j].read_all_rows
                   for j in range(len(stages))}

    def select_rows(normed, spans_j):
        index = np.concatenate([np.arange(start, end, dtype=np.int64)
                                for start, end in spans_j])
        if hasattr(normed, "index_select"):
            return normed.index_select(
                0, torch.from_numpy(index).to(normed.device, non_blocking=True))
        return [normed[r] for r in index]

    def answer_stages(part):
        """The stage of every answer of a chunk's groups, in order."""
        out = []
        for a, j, start, end, _ in part:
            if writes[j] and start == 0 and not merged(j, start, end):
                out.append(j)        # the frame entry's answer
            out.extend([j] * (end - start))
        return out

    def canvas_conditioning(chunk, spans):
        """The conditioning rows of the canvases at the chunk's canvas spans."""
        meta = chunk.meta["canvas"]
        cu_q = meta["cu_q_host"]
        index = np.concatenate([np.arange(cu_q[k0], cu_q[k1], dtype=np.int64)
                                for k0, k1 in spans])
        rows = meta["conditioning_rows"]
        return rows.index_select(
            0, torch.from_numpy(index).to(rows.device, non_blocking=True))

    def submit(normed, entries, chunk, part):
        """Hand each readout its answer rows; returns handles by readout group.

        With one readout for every stage the whole chunk goes in one
        call, keyed None; otherwise each readout's stages' rows go to
        it in one call, selected with one index. A readout with
        ``reads_chunk`` (a denoising step's) takes its canvases' rows
        with each one's step and conditioning rows.
        """
        rows_per_answer = chunk.rows_per_answer
        if shared_readout:
            readout = stages[0].readout
            if getattr(readout, "reads_chunk", False):
                return {None: readout.submit(
                    normed, steps=[stages[j].step for j in answer_stages(part)],
                    conditioning_rows=chunk.meta["canvas"]["conditioning_rows"])}
            return {None: (readout.submit(normed, rows_per_answer=rows_per_answer)
                           if rows_per_answer else readout.submit(normed))}
        spans = {}          # group -> its entries' row spans, in chunk order
        canvases = {}       # group -> its entries' spans of chunk canvases
        steps = {}          # group -> each canvas's step
        row = 0
        seen = 0
        for j, rows in entries:
            key = group_key[j]
            spans.setdefault(key, []).append((row, row + rows))
            row += rows
            # every suffix on a canvas model packs one canvas
            count = (rows // canvas_rows[j]
                     if stages[j].read_all_rows and canvas_rows[j] else rows)
            canvases.setdefault(key, []).append((seen, seen + count))
            steps.setdefault(key, []).extend([stages[j].step] * count)
            seen += count
        handles = {}
        per_answer = list(rows_per_answer) if rows_per_answer else None
        for key, spans_key in spans.items():
            readout = readout_of[key]
            selected = select_rows(normed, spans_key)
            if getattr(readout, "reads_chunk", False):
                handles[key] = readout.submit(
                    selected, steps=steps[key],
                    conditioning_rows=canvas_conditioning(chunk, canvases[key]))
            # a stage reading one row per answer takes no row counts
            elif per_answer is None or not read_all_of[key]:
                handles[key] = readout.submit(selected)
            else:
                # answers of this readout's stages, in entry order
                counts = []
                answer = 0
                for jj, rows in entries:
                    taken = 0
                    while taken < rows:
                        taken += per_answer[answer]
                        if group_key[jj] == key:
                            counts.append(per_answer[answer])
                        answer += 1
                handles[key] = readout.submit(selected, rows_per_answer=counts)
        return handles

    def report(entry):
        groups, entries, handles = entry
        values = {key: readout_of[key].result(handle)
                  for key, handle in handles.items()}
        pos = {key: 0 for key in handles}
        transitions = []
        for a, j, start, end, _ in groups:
            key = group_key[j]
            if writes[j] and start == 0 and not merged(j, start, end):
                pos[key] += 1        # the frame entry's answer means nothing
            cnt = end - start
            stage_before = sched._stage[a]
            events = sched.report(
                a, j, start, end, values[key][pos[key]:pos[key] + cnt])
            for kind, anchor in events:
                event(kind, anchor)
                # the stage's own decision: a document the next stage's
                # requests drop still passed this one
                transitions.append((anchor, j, bool(sched._true[anchor][j])))
            if not events and sched._stage[a] != stage_before:
                transitions.append((a, j, True))
            pos[key] += cnt
        progress.update(
            progress.done + sum(end - start for _, _, start, end, _ in groups)
            if counting_answers else finished[0])
        if transitions and on_chunk is not None:
            on_chunk(transitions)

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
        nonlocal tokens, pack_s
        started = time.perf_counter()
        chunk, entries = build(part)
        pack_s += time.perf_counter() - started
        tokens += chunk.tokens
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normed = _forward(pipeline, arena, chunk)
        e1.record()
        for key in chunk.fresh_keys:
            arena.trim_window(key)
        spans.append((part[0][1], e0, e1))
        handles = submit(normed, entries, chunk, part)
        outstanding.append((part, entries, handles))
        # read the previous chunk's answers while this one runs
        while len(outstanding) > 1:
            report(outstanding.pop(0))

    while True:
        # documents with nothing to send at their first stage never run
        for kind, anchor in sched.take_settled():
            event(kind, anchor)
        if sched.done():
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
        stats["pack_s"] = pack_s
    return sched.answers, spans, tokens

