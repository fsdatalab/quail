"""Schedule LLM operator stages while retaining each document's KV.

Filters, joins, and classifications describe their work as Stage objects.
The shared admission scheduler combines documents at different stages in
each forward pass and releases KV when a document finishes or is removed.
"""

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import numpy as np

from quail.backends.quail.executor.pack import DROP, SKIP, JoinAdmission
from quail.progress import Progress, logger


@dataclass
class Stage:
    """Token requests, answer readout, and advancement condition for one step.

    Attributes:
        suffixes: Token sequences processed after each document and frame.
        readout: Converts selected hidden states to Boolean answers, scores,
            or category token values. Its dtype determines the answer array.
        frame: Tokens retained after the document. Consecutive identical
            frames are written only once.
        requests: Optional callback from document key to suffix indices.
            None selects all suffixes; DROP removes the document; SKIP
            advances it without a request.
        decide: Optional callback from document index and answer row to a
            Boolean. True advances the document or retains it at the final
            stage. Without a callback, any true answer is sufficient.
        read_all_rows: Whether to read every suffix row rather than its last.
        read_rows: Optional count of trailing rows to read per suffix,
            overriding read_all_rows. None takes the readout's
            trailing_rows for every suffix when it has more than one.
        single: Whether each document has one request that can be packed
            with its prefix and frame as one entry.
        append: Whether the request's tokens join the document's KV after
            the frame and any tokens earlier append stages added, so the
            next append stage reads them instead of feeding them again.
        label: Name displayed in progress messages.
        chains: Optional trie chains and ancestor references from trie_chains().
        canvas: Optional callback from document index to diffusion canvas
            tokens, shared by its suffixes or supplied per suffix. None uses
            the pipeline's default canvas.
        canvas_rows: Number of rows in each canvas returned by canvas.
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
    append: bool = False
    label: str = ""
    chains: list | None = None
    canvas: Callable | None = None
    canvas_rows: int = 0

    def __post_init__(self):
        trailing = getattr(self.readout, "trailing_rows", 1)
        if self.read_rows is None and trailing > 1:
            self.read_rows = [trailing] * len(self.suffixes)
            self.read_all_rows = True


def shared_preamble_tokens(question_ids) -> int:
    """Return the length of the token prefix shared by multiple questions."""
    if len(question_ids) < 2:
        return 0
    p = 0
    while all(len(q) > p and q[p] == question_ids[0][p]
              for q in question_ids):
        p += 1
    return p


def filter_stages(question_ids, readout) -> list:
    """Build one filter stage per question with a shared prompt frame.

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
    """Return which stages need to write a frame not retained by the prior stage."""
    writes = []
    previous = None
    for stage in stages:
        frame = list(stage.frame)
        writes.append(bool(frame) and frame != previous)
        previous = frame
    return writes


def _advance_stage(stages, a, j, row):
    """Apply the stage's decision after its complete row has returned."""
    stage = stages[j]
    if stage.decide is None:
        return bool(any(row) if stage.readout.dtype is None else np.any(row))
    return bool(stage.decide(a, row))


def run_stages(implementation, arena, pipeline, stages, anchor_prefixes, budget, *,
               anchor_keys=None, on_settled=None, staging=None,
               attention_mode=None, prefix_tree=None, stats=None,
               limit=None, paged=True, unit="documents", count_answers=False,
               label=None, default_attention="tree", on_chunk=None,
               on_answers=None):
    """Run every stage over the documents with one admission.

    Args:
        implementation: The device implementation the chunks run on.
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
        count_answers: Whether progress counts answers, one per partner
            of each document at each stage, instead of finished documents.
        label: The progress label; None names the stages.
        default_attention: The attention path when the plan chose none
            and the model has tree attention.
        on_chunk: Optional callable run after each chunk's answers are
            read with the documents that completed a stage in it, as
            (document index, stage, passed) tuples: passed says the
            document went on to the next stage, or survived the last.
        on_answers: Optional callable(document index, stage, start, end,
            rows) run as each chunk's answers are read, with the
            document's answer rows for its requests start to end at
            the stage.

    Returns:
        A tuple of answers, timing spans, and fresh token count. Answers is
        one document-indexed mapping per stage. Each span contains a mapping
        of stage indices to packed row counts, the chunk token count, and
        the device implementation's start and end events.
    """
    if not stages:
        return [], [], 0
    return _StageExecutor(
        implementation, arena, pipeline, stages, anchor_prefixes, budget,
        anchor_keys=anchor_keys, on_settled=on_settled, staging=staging,
        attention_mode=attention_mode, prefix_tree=prefix_tree, stats=stats,
        limit=limit, paged=paged, unit=unit, count_answers=count_answers,
        label=label, default_attention=default_attention, on_chunk=on_chunk,
        on_answers=on_answers,
    ).run()


class _StageExecutor:
    """Admission state, packed inputs, pending answers, and metrics for one run."""

    def __init__(self, implementation, arena, pipeline, stages, anchor_prefixes,
                 budget, *,
                 anchor_keys, on_settled, staging, attention_mode, prefix_tree,
                 stats, limit, paged, unit, count_answers, label,
                 default_attention, on_chunk, on_answers):
        from quail.backends.quail.executor.chunk import Suffixes
        from quail.backends.quail.executor.loop import (
            attention_path,
            borrow_check,
            lowest_borrows,
        )

        self.implementation = implementation
        self.arena = arena
        self.pipeline = pipeline
        self.stages = stages
        self.on_settled = on_settled
        self.staging = staging
        self.stats = stats
        self.paged = paged
        self.on_chunk = on_chunk
        self.on_answers = on_answers
        self.label = label

        k = len(self.stages)
        if not self.paged and (
                k > 1 or prefix_tree is not None
                or any(len(stage.suffixes) != 1 for stage in self.stages)):
            raise ValueError("the unpaged path runs one stage of one suffix")
        if any(stage.append and not stage.single for stage in self.stages):
            raise ValueError("an append stage sends one request per document")
        self.prefixes = anchor_prefixes
        # a prefix is built on each access, so its length is read once
        self.prefix_lengths = [len(prefix) for prefix in self.prefixes]
        self.keys = (list(range(len(self.prefixes))) if anchor_keys is None
                     else anchor_keys)
        if len(self.keys) != len(self.prefixes):
            raise ValueError("anchor_keys must match anchor_prefixes")
        self.frames = [list(stage.frame) for stage in self.stages]
        self.frame_ids = [np.asarray(frame, dtype=np.int64) for frame in self.frames]
        self.writes = frame_writes(self.stages)
        self.suffixes = [Suffixes.of(stage.suffixes) for stage in self.stages]
        self.read_rows = [None if stage.read_rows is None
                          else np.asarray(stage.read_rows, dtype=np.int64)
                          for stage in self.stages]
        # Stages packing label chains always need tree attention.
        if any(stage.chains is not None for stage in self.stages):
            attention_mode = "tree"
        self.mode = attention_path(
            self.pipeline, attention_mode, default=default_attention)
        self.canvas = tuple(self.pipeline.canvas_ids)
        self.answer_row = self.pipeline.canvas_answer_row
        self.canvas_rows = [len(self.canvas) if stage.canvas is None
                            else stage.canvas_rows for stage in self.stages]

        # A frame's canvas occupies document pages after the frame.
        frame_max = max(len(f) + (len(self.canvas) if f else 0) for f in self.frames)
        self.capacity_extra = frame_max
        # A single suffix also occupies document pages on the unified path.
        if self.paged and self.mode == "unified":
            self.capacity_extra = max(
                [frame_max] + [len(frame) + len(stage.suffixes[0]) + rows
                               for frame, stage, rows in zip(
                                   self.frames, self.stages, self.canvas_rows)
                               if stage.single])
        # an append stage's tokens stay in document pages after the
        # frame, so admission reserves the frame and every append round
        for frame, stage in zip(self.frames, self.stages):
            if stage.append:
                kept = sum(int(other.lengths.max())
                           for f, other, s in zip(self.frames, self.suffixes,
                                                  self.stages)
                           if s.append and f == frame)
                self.capacity_extra = max(self.capacity_extra,
                                          len(frame) + kept)
        # append-stage tokens each document holds after its frame
        self.appended = [0] * len(self.prefixes)

        resident = ({a: self._held_pages(self.keys[a], self.prefix_lengths[a])
                     for a, key in enumerate(self.keys)
                     if self.arena.is_resident(key)}
                    if self.paged else {})
        # Pin resident documents while fresh admissions evict retained KV.
        for a in resident:
            self.arena.pin(self.keys[a])

        self.asking = any(stage.requests is not None for stage in self.stages)
        if prefix_tree is not None and not prefix_tree.shared_tokens:
            prefix_tree = None
        self.sched = JoinAdmission(
            self.prefix_lengths,
            [(s.lengths + rows).tolist()
             for s, rows in zip(self.suffixes, self.canvas_rows)],
            budget,
            self.arena.n_pages if self.paged else 1 << 62,
            self.arena.page_tokens,
            frame_tokens=[len(f) for f in self.frames], resident=resident,
            anchor_partners={a: self._requests_of(key)
                             for a, key in enumerate(self.keys)},
            # a windowed model may pack any chunk unified, so it reserves
            # the unified path's temporary pages throughout
            temporary_suffix_pages=self.paged and self.mode == "unified",
            answer_dtype=[stage.readout.dtype for stage in self.stages],
            frame_canvas_tokens=len(self.canvas),
            page_cost=self.arena.page_cost if self.paged else (lambda *args: 0),
            tree=prefix_tree,
            advance=partial(_advance_stage, stages),
            frame_writes=self.writes,
            limit=limit,
            extra_tokens=self.capacity_extra,
        )
        self.borrowing = self.sched.borrowing
        self.borrowing.can_borrow = borrow_check(self.arena, self.keys, self.borrowing)
        # The admission detaches residents from borrowing: they pack no prefix.
        self.borrowers = self.borrowing.borrowers()
        self.lowest_borrow = lowest_borrows(self.borrowing)
        self.held = [self.keys[a] for a, n in enumerate(self.borrowers) if n]

        self.spans = []
        self.tokens = 0
        self.pack_s = 0.0
        self.outstanding = []     # (groups, handles) in launch order

        if self.label is None:
            self.label = " > ".join(stage.label or f"stage {j}"
                                   for j, stage in enumerate(self.stages))
        self.counting_answers = count_answers
        total = (sum(self.sched.partner_count(a, j) for a in range(len(self.prefixes))
                     for j in range(k))
                 if self.counting_answers else len(self.prefixes))
        self.progress = Progress(self.label, total=total, unit=unit)
        self.finished = 0

        # Combine stages sharing a readout in one submission per chunk.
        self.shared_readout = all(stage.readout is self.stages[0].readout
                                  for stage in self.stages)
        groups_of = {}
        self.group_key = {}
        for j, stage in enumerate(self.stages):
            self.group_key[j] = (None if self.shared_readout else groups_of.setdefault(
                (id(stage.readout), stage.read_all_rows), len(groups_of)))
        self.readout_of = {self.group_key[j]: stage.readout
                           for j, stage in enumerate(self.stages)}
        self.read_all_of = {self.group_key[j]: stage.read_all_rows
                            for j, stage in enumerate(self.stages)}

    def _entry_rows(self, a, j, start, end, carried):
        f = self.prefix_lengths[a]
        indices = self.sched.partner_indices(a, j, start, end)
        lengths = self.suffixes[j].lengths_at(indices)
        rows = ((f if carried else 0) + int(lengths.sum())
                + self.canvas_rows[j] * len(lengths))
        if self.writes[j] and start == 0:
            rows += len(self.frames[j]) + len(self.canvas)
        return rows

    def _held_pages(self, key, prefix_tokens):
        # the admission prices a document at page_cost(prefix + extra)
        # less what it holds; a trimmed window holds fewer sliding
        # pages than that price assumes, so count what growing takes
        capacity = prefix_tokens + self.capacity_extra
        return (self.arena.page_cost(capacity)
                - self.arena.growth_cost(key, capacity))

    def _requests_of(self, key):
        """Return a stage request selector, or None if every request is selected."""
        if not self.asking:
            return None
        stages = self.stages
        return lambda j: (None if stages[j].requests is None
                          else stages[j].requests(key))

    def _merged(self, j, start, end):
        """Return whether the group packs its frame and suffix as one entry."""
        return self.stages[j].single and end - start == 1

    def _build(self, chunk_groups):
        """Pack a chunk and identify each entry's stage and answer row count."""
        from quail.backends.quail.executor.chunk import Suffixes, pack_chunk

        specs = []
        entries = []
        appended = {}
        for a, j, start, end, carried in chunk_groups:
            key = self.keys[a]
            f = self.prefix_lengths[a]
            frame = self.frames[j]
            append = self.stages[j].append
            # a written frame replaces any path an earlier stage kept
            path = 0 if self.writes[j] and start == 0 else self.appended[a]
            parent, shared = ((self.borrowing.parent(a), self.borrowing.shared(a))
                              if carried else (None, 0))
            if self.paged:
                fresh = not self.arena.is_resident(key)
                got = self.arena.activate(
                    key, f, capacity_tokens=f + self.capacity_extra,
                    base_tokens=f,
                    borrow=(self.keys[parent], shared) if shared else None)
                assert got is not None, \
                    "the admission placed a document the arena cannot hold"
                if fresh and a < len(self.borrowers) and self.borrowers[a]:
                    self.arena.hold(key, self.borrowers[a])
                    self.arena.keep_window(key, self.lowest_borrow[a])
            prefix = None
            if carried:
                prefix = self.prefixes[a][shared:] if shared else self.prefixes[a]
            indices = self.sched.partner_indices(a, j, start, end)
            sufs = self.suffixes[j].take(indices)
            read_all = self.stages[j].read_all_rows
            rows = int(sufs.lengths.sum()) if read_all else len(sufs)
            own = {}
            if self.read_rows[j] is not None:
                own["read_rows"] = self.read_rows[j][np.asarray(list(indices))]
                rows = int(own["read_rows"].sum())
            if self.stages[j].canvas is not None:
                ids = np.asarray(self.stages[j].canvas(a))
                if ids.ndim == 2:
                    ids = ids[np.asarray(list(indices), dtype=np.int64)]
                own = dict(canvas=ids)
                if read_all:
                    rows = self.stages[j].canvas_rows * len(sufs)
            # under tree attention a borrowing document with one
            # suffix reads its parent's pages stacked with its siblings
            read_key = self.keys[parent] if shared and self.stages[j].single else None
            kept = int(sufs.lengths.sum()) if append else 0
            if append or (self.writes[j] and start == 0):
                appended[a] = path + kept
            if self.writes[j] and start == 0 and self._merged(j, start, end):
                # the frame and the one suffix are one entry; the
                # frame's rows are scattered into KV after the document
                specs.append(dict(
                    key=key, prefix=prefix, start=shared, read_key=read_key,
                    f=f, suffixes=Suffixes(
                        np.concatenate([self.frame_ids[j], sufs.ids]),
                        [len(frame) + int(sufs.lengths[0])]),
                    write_suffix_tokens=len(frame) + kept, single=True,
                    read_all_rows=read_all, **own))
            elif self.writes[j] and start == 0:
                # frame entry: scatter the frame into KV after the
                # document rows; its own answer row means nothing
                specs.append(dict(
                    key=key, prefix=prefix, start=shared, read_key=read_key,
                    f=f, suffixes=Suffixes(self.frame_ids[j], [len(frame)]),
                    write_suffix_tokens=len(frame)))
                entries.append((j, 1))
                specs.append(dict(
                    key=key, prefix=None, f=f + len(frame),
                    suffixes=sufs, read_all_rows=read_all,
                    write_suffix_tokens=kept,
                    chains=self.stages[j].chains, **own))
            else:
                specs.append(dict(
                    key=key, prefix=prefix, start=shared, read_key=read_key,
                    f=f + len(frame) + path,
                    suffixes=sufs, read_all_rows=read_all,
                    write_suffix_tokens=kept,
                    single=self.stages[j].single and end - start == 1,
                    chains=self.stages[j].chains, **own))
            entries.append((j, rows))
        chunk = pack_chunk(
            self.implementation, self.arena, specs, attention_mode=self.mode,
            staging=self.staging, canvas=self.canvas, answer_row=self.answer_row)
        for a, tokens in appended.items():
            self.appended[a] = tokens
        return chunk, entries

    def _settle(self, settlement):
        anchor = settlement.anchor
        if self.on_settled is None:
            if self.arena.is_resident(self.keys[anchor]):
                self.arena.free_key(self.keys[anchor])
        else:
            self.on_settled(
                anchor, settlement.survived, self.sched.answers[-1].get(anchor, []))

    def _event(self, settlement):
        # Progress counts settled documents: a dropped one is done too.
        self.finished += 1
        if settlement.kind == "finished":
            self._settle(settlement)
        elif self.arena.is_resident(self.keys[settlement.anchor]):
            self.arena.free_key(self.keys[settlement.anchor])

    def _select_rows(self, normed, spans_j):
        index = np.concatenate([np.arange(start, end, dtype=np.int64)
                                for start, end in spans_j])
        return self.implementation.select_rows(normed, index)

    def _submit(self, normed, entries, chunk):
        """Submit answer rows to readouts and return handles by readout group.

        With one readout for every stage the whole chunk goes in one
        call, keyed None; otherwise each readout's stages' rows go to
        it in one call, selected with one index.
        """
        rows_per_answer = chunk.rows_per_answer
        if self.shared_readout:
            readout = self.stages[0].readout
            return {None: (readout.submit(normed, rows_per_answer=rows_per_answer)
                           if rows_per_answer else readout.submit(normed))}
        spans = {}          # group -> its entries' row spans, in chunk order
        row = 0
        for j, rows in entries:
            spans.setdefault(self.group_key[j], []).append((row, row + rows))
            row += rows
        handles = {}
        per_answer = list(rows_per_answer) if rows_per_answer else None
        for key, spans_key in spans.items():
            readout = self.readout_of[key]
            selected = self._select_rows(normed, spans_key)
            # a stage reading one row per answer takes no row counts
            if per_answer is None or not self.read_all_of[key]:
                handles[key] = readout.submit(selected)
            else:
                # answers of this readout's stages, in entry order
                counts = []
                answer = 0
                for jj, rows in entries:
                    taken = 0
                    while taken < rows:
                        taken += per_answer[answer]
                        if self.group_key[jj] == key:
                            counts.append(per_answer[answer])
                        answer += 1
                handles[key] = readout.submit(selected, rows_per_answer=counts)
        return handles

    def _report(self, entry):
        groups, handles = entry
        values = {key: self.readout_of[key].result(handle)
                  for key, handle in handles.items()}
        pos = {key: 0 for key in handles}
        transitions = []
        for a, j, start, end, _ in groups:
            key = self.group_key[j]
            if self.writes[j] and start == 0 and not self._merged(j, start, end):
                pos[key] += 1        # the frame entry's answer means nothing
            cnt = end - start
            rows = values[key][pos[key]:pos[key] + cnt]
            result = self.sched.report(a, j, start, end, rows)
            if self.on_answers is not None:
                self.on_answers(a, j, start, end, rows)
            for settlement in result.settlements:
                self._event(settlement)
            transitions.extend(result.transitions)
            pos[key] += cnt
        self.progress.update(
            self.progress.done + sum(end - start for _, _, start, end, _ in groups)
            if self.counting_answers else self.finished)
        if transitions and self.on_chunk is not None:
            self.on_chunk(transitions)

    def _run_part(self, part):
        """Run one chunk of groups, halving it when its pages do not fit."""
        from quail.backends.quail.executor.chunk import ArenaFullError

        try:
            self._run_one(part)
        except ArenaFullError as error:
            # retained KV nothing here reads makes room first
            need = self.arena.page_cost(sum(self._entry_rows(*e) for e in part))
            if self.arena.evict_retained(need):
                logger.info("chunk of %d groups retried after evicting "
                            "retained KV", len(part))
                self._run_part(part)
                return
            if len(part) < 2:
                raise
            logger.info("chunk of %d groups split: %s", len(part), error)
            half = len(part) // 2
            self._run_part(part[:half])
            self._run_part(part[half:])

    def _run_one(self, part):
        from quail.backends.quail.executor.loop import _forward

        started = time.perf_counter()
        chunk, entries = self._build(part)
        self.pack_s += time.perf_counter() - started
        self.tokens += chunk.tokens
        e0 = self.implementation.record_event()
        normed = _forward(self.pipeline, self.arena, chunk)
        e1 = self.implementation.record_event()
        for key in chunk.fresh_keys:
            self.arena.trim_window(key)
        by_stage = {}
        for entry in part:
            by_stage[entry[1]] = by_stage.get(entry[1], 0) + self._entry_rows(*entry)
        self.spans.append((by_stage, chunk.tokens, e0, e1))
        handles = self._submit(normed, entries, chunk)
        self.outstanding.append((part, handles))
        # read the previous chunk's answers while this one runs
        while len(self.outstanding) > 1:
            self._report(self.outstanding.pop(0))

    def run(self):
        """Drive admission and drain the launched answer handles in order."""
        while True:
            # documents with nothing to send at their first stage never run
            for settlement in self.sched.take_settled():
                self._event(settlement)
            if self.sched.done():
                break
            if self.sched.blocked_pages:
                # the free list is short for the next fresh document:
                # retained KV nothing here reads makes room
                self.arena.evict_retained(self.sched.blocked_pages)
            groups = self.sched.next_chunk(
                self.arena.free_pages if self.paged else 1 << 62)
            if not groups:
                if self.outstanding:
                    self._report(self.outstanding.pop(0))
                    continue
                if (self.sched.blocked_pages
                        and self.arena.evict_retained(self.sched.blocked_pages)):
                    continue
                # parents freed but held for queued children hold the
                # pages: the children pack their whole prefixes
                if self.arena.drop_holds(self.held):
                    continue
                raise AssertionError("nothing buildable and nothing in flight")
            self._run_part(groups)
            # the packed children hold their parents' pages now
            for a, _, start, _, carried in groups:
                parent = (self.borrowing.tree_parent[a]
                          if a < len(self.borrowing.tree_parent) else None)
                if carried and start == 0 and parent is not None:
                    self.arena.release(self.keys[parent])
        while self.outstanding:
            self._report(self.outstanding.pop(0))
        # a limit ends the run with documents still queued
        for anchor in self.sched.drain():
            if self.arena.is_resident(self.keys[anchor]):
                self.arena.free_key(self.keys[anchor])
        self.progress.finish(f"{self.label} done", f"{self.tokens:,} fresh tokens")
        self.arena.drop_holds(self.held)
        if self.stats is not None:
            self.stats["borrowed_tokens"] = self.borrowing.borrowed_tokens
            self.stats["pack_s"] = self.pack_s
        return self.sched.answers, self.spans, self.tokens
