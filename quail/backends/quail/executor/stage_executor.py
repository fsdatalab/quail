"""Run one stage list: admission, packed forward chunks, and answer readout."""

import time
from functools import partial
from itertools import groupby

import numpy as np

from quail.backends.quail.executor.admission import (
    DROP,
    SKIP,
    JoinAdmission,
    OwnRequests,
    SkipTo,
)
from quail.backends.quail.executor.readout_groups import ReadoutGroups
from quail.backends.quail.executor.stages import (
    PendingAnswers,
    _advance_stage,
    frame_writes,
    state_share_points,
)
from quail.progress import Progress, logger

# the draft token fed after a readout row, where there is none
_NO_FOLLOWING = np.full(1 << 16, -1, dtype=np.int64)
_NO_FOLLOWING.setflags(write=False)


def attention_path(pipeline, requested, default_attention="unified") -> str:
    """The path a filter stream or join runs: the plan's request where the model has it.

    None asks for the default_attention. A model without tree attention, and a
    canvas model (whose rows the tree path does not pack), run unified.
    """
    if requested is None:
        requested = default_attention
    if requested == "tree" and pipeline.tree_attention and not pipeline.canvas_ids:
        return "tree"
    return "unified"


def forward_chunk(pipeline, arena, chunk):
    """Run the forward pass, then release the chunk's temporary pages.

    The loop owns every page it hands the forward pass, so it frees the
    temporaries whether the pass returns or raises.
    """
    try:
        if pipeline.decode_graphs is not None:
            out = pipeline.decode_graphs.forward(chunk)
        else:
            out = pipeline.forward_chunk(chunk)
        arena.sample_memory(chunk.tokens, len(chunk.layout))
        return out
    finally:
        keys, chunk.temporary_keys = chunk.temporary_keys, ()
        for key in keys:
            arena.free_key(key)


def borrow_check(arena, keys, borrowing):
    """can_borrow for Borrowing: whether the arena can serve a borrow.

    A fresh parent admitted in the same chunk is not allocated yet;
    its sliding pages will start at the window origin below the tokens
    it borrows itself. A parent already resident, admitted in this
    chunk or earlier, has the pages it has: it may have trimmed its
    window in an earlier query.
    """
    def can_borrow(document, parent, same_chunk):
        shared = borrowing.tree_shared[document]
        if not same_chunk or arena.is_resident(keys[parent]):
            return arena.can_borrow(keys[parent], shared)
        return arena.origin(borrowing.shared(parent)) <= arena.origin(shared)
    return can_borrow


def _replay_counts(graphs) -> tuple:
    """Return the passes a pipeline's decode graphs and piece graphs replayed."""
    return (graphs.replays,
            0 if graphs.pieces is None else graphs.pieces.replays)


def lowest_borrows(borrowing):
    """Per document, the fewest tokens any of its borrowers shares."""
    lowest = {}
    for document, parent in enumerate(borrowing.tree_parent):
        if parent is not None:
            shared = borrowing.tree_shared[document]
            lowest[parent] = min(lowest.get(parent, shared), shared)
    return lowest


class StageExecutor:
    """Admission state, packed inputs, pending answers, and metrics for one run."""

    def __init__(self, torch, arena, pipeline, stages, anchor_prefixes, budget, *,
                 anchor_keys, on_settled, staging, attention_mode, prefix_tree,
                 stats, limit, paged, unit, count_answers, label,
                 default_attention, on_chunk, on_answers, stop_groups=None,
                 appended_tokens=None, gpu_timing=False, survival=None,
                 growth=None):
        from quail.backends.quail.executor.chunk import Suffixes


        self.torch = torch
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
        self.gpu_timing = gpu_timing

        k = len(self.stages)
        if not self.paged and (
                k > 1 or prefix_tree is not None
                or any(stage.own_suffixes or len(stage.suffixes) != 1
                       for stage in self.stages)):
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
        if getattr(self.arena, "num_state_slots", 0):
            self.writes = [write or (stage.rewind and bool(stage.frame))
                           for write, stage in zip(self.writes, self.stages)]
        # a stage whose suffixes are each document's own is priced by
        # one sequence of its most tokens until a document enters it
        self.suffixes = [Suffixes.of([[0] * stage.suffix_tokens])
                         if stage.own_suffixes else Suffixes.of(stage.suffixes)
                         for stage in self.stages]
        self.own = {}      # (stage, document) -> its StageRequests
        self.read_rows = [None if stage.read_rows is None
                          else np.asarray(stage.read_rows, dtype=np.int64)
                          for stage in self.stages]
        self.read_masks = [None if stage.read_mask is None
                           else np.asarray(stage.read_mask, dtype=bool)
                           for stage in self.stages]
        # Stages packing label paths always need tree attention.
        if any(stage.paths is not None for stage in self.stages):
            attention_mode = "tree"
        self.mode = attention_path(
            self.pipeline, attention_mode, default_attention=default_attention)
        # rounds and prefix slices are laid out as plain rows over the
        # document's pages: on the tree path, or on the unified path
        # of a model with one KV pool and no recurrent state
        plain = bool(paged and (
            self.mode == "tree"
            or not (getattr(self.arena, "has_sliding", False)
                    or getattr(self.arena, "num_state_slots", 0))))
        # a prefix longer than a pass's room is packed over several
        # passes, so a document's length never bounds the budget
        self.slice_prefill = plain
        # a stage that runs ahead of its answers feeds its rounds from
        # the GPU
        self.ahead = [bool(stage.ahead) and plain for stage in stages]
        self.feeding = any(self.ahead)
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
                [frame_max] + [len(frame) + int(other.lengths.max()) + rows
                               for frame, stage, other, rows in zip(
                                   self.frames, self.stages, self.suffixes,
                                   self.canvas_rows)
                               if stage.single])
        # what a document's fixed requests need: all but the stages
        # that append every round, whose KV can grow as they write
        fixed_extra = self.capacity_extra
        appended = sorted(
            (tuple(frame), int(suffixes.lengths.max()), stage.repeats)
            for frame, suffixes, stage in zip(self.frames, self.suffixes, self.stages)
            if stage.append)
        for frame, group in groupby(appended, key=lambda entry: entry[0]):
            entries = list(group)
            kept = sum(tokens for _, tokens, _ in entries)
            if any(repeats for _, _, repeats in entries):
                # a repeating stage appends its suffix every round, so
                # only the caller's bound says what it leaves in KV
                if appended_tokens is None:
                    raise ValueError("a repeating append stage needs "
                                     "appended_tokens")
                kept = max(kept, int(appended_tokens))
            else:
                if appended_tokens is not None:
                    kept = min(kept, int(appended_tokens))
                fixed_extra = max(fixed_extra, len(frame) + kept)
            self.capacity_extra = max(self.capacity_extra, len(frame) + kept)
        # append-stage tokens each document holds after its frame
        self.appended = [0] * len(self.prefixes)
        # rows past the prefix a document's pages cover at admission:
        # all it can ever need, or, when its KV grows as it writes,
        # its first request and the rounds in flight before it grows
        self.growth = None
        self.reserve_extra = self.capacity_extra
        if growth is not None and self.slice_prefill:
            from quail.backends.quail.executor.growth import KVGrowth

            self.reserve_extra = min(self.capacity_extra,
                                     max(fixed_extra, int(growth.initial_tokens)))
            self.growth = KVGrowth(
                self.arena, self.keys, growth, self.prefix_lengths,
                [length + self.capacity_extra for length in self.prefix_lengths],
                initial_tokens=self.reserve_extra)

        resident = {}
        if self.paged:
            for a, key in enumerate(self.keys):
                if not self.arena.is_resident(key):
                    continue
                if self.arena.prefix_tokens(key) != self.prefix_lengths[a]:
                    # KV an earlier operator left under this key was
                    # built for another prefix; it is fed afresh
                    self.arena.free_key(key)
                    continue
                resident[a] = self._held_pages(key, self.prefix_lengths[a])
        # Pin resident documents while fresh admissions evict retained KV.
        for a in resident:
            self.arena.pin(self.keys[a])

        self.asking = any(stage.requests is not None or stage.own_suffixes
                          for stage in self.stages)
        if prefix_tree is not None and not prefix_tree.shared_tokens:
            prefix_tree = None
        self.sched = JoinAdmission(
            self.prefix_lengths,
            [s.lengths + rows
             for s, rows in zip(self.suffixes, self.canvas_rows)],
            budget,
            self.arena.num_pages if self.paged else 1 << 62,
            self.arena.page_tokens,
            frame_tokens=[len(f) for f in self.frames], resident=resident,
            anchor_partners={a: self._requests_of(a, key)
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
            limit=limit, survival=survival,
            stop_groups=stop_groups,
            stop_width=self._stop_width(stop_groups, budget),
            extra_tokens=self.reserve_extra,
            reserve=self._reserve if self.paged else None,
            repeats=[stage.repeats for stage in self.stages],
            slice_prefill=self.slice_prefill, ahead=self.ahead,
            costs=[stage.costs if ahead else None
                   for stage, ahead in zip(self.stages, self.ahead)],
            admits=None if self.growth is None else self.growth.admits,
        )
        # the stages whose rounds are packed many at once, from arrays
        self.bulk = [bool(ahead and stage.rounds is not None)
                     for stage, ahead in zip(self.stages, self.ahead)]
        self.borrowing = self.sched.borrowing
        self.borrowing.can_borrow = borrow_check(self.arena, self.keys, self.borrowing)
        # a Gated DeltaNet layer starts a borrower from the parent's state
        # at the shared point, so a parent writes one slot per point
        self.share_points = [()] * len(self.keys)
        self._extends = False
        spec = getattr(self.pipeline, "spec", None)
        if (self.paged and getattr(spec, "state_layers", ())
                and not getattr(self.arena, "num_state_slots", 0)):
            raise ValueError(
                "the arena has no checkpoint slots: the pool cannot hold "
                "one document's state beside a chunk's KV, so a document "
                "this long needs a smaller chunk or a larger pool")
        if self.paged and getattr(self.arena, "num_state_slots", 0):
            self._extends = (any(self.writes)
                             or any(s.append for s in self.stages))
            # a parent holds one slot per share length beside its own and
            # its extension, and its borrowers need slots of their own, so
            # a small pool keeps only the lengths that save the most
            base = 1 + int(self._extends)
            room = max(0, (self.arena.num_state_slots - base) // 3)
            self.share_points = state_share_points(
                self.borrowing.tree_parent, self.borrowing.tree_shared,
                limit=room)
            need = base + max(map(len, self.share_points), default=0)
            if self.arena.num_state_slots < need:
                raise ValueError(
                    f"the arena has {self.arena.num_state_slots} checkpoint "
                    f"slots and this query holds {need} per document in "
                    f"flight; a document this long needs a smaller chunk "
                    f"or a larger pool")
            self.borrowing.can_borrow = self._state_borrow_check(
                self.borrowing.can_borrow)
        # The admission detaches residents from borrowing: they pack no prefix.
        self.borrowers = self.borrowing.borrowers()
        self.lowest_borrow = lowest_borrows(self.borrowing)
        self.held = [self.keys[a] for a, n in enumerate(self.borrowers) if n]

        self._held_parents = set()
        self._reserved_parents = {}
        self.spans = []
        self.tokens = 0
        self.pack_s = 0.0
        # host seconds per phase of the loop, and the readouts' GPU
        # event pairs, for telling host-bound passes from GPU-bound
        self.host_s = {"admit": 0.0, "advance": 0.0, "launch": 0.0,
                       "launch_graph": 0.0, "launch_pieces": 0.0,
                       "launch_eager": 0.0, "wait": 0.0,
                       "report": 0.0, "decide": 0.0, "submit": 0.0}
        self.readout_spans = []
        self.outstanding = []     # PendingAnswers in launch order
        self.passes = 0
        # per anchor, the pass and answer index of its latest launched
        # request: the source of its next device-fed round
        self.latest = {}
        self.split_answers = any(s.append and s.own_suffixes for s in stages)

        if self.label is None:
            self.label = " > ".join(stage.label or f"stage {j}"
                                   for j, stage in enumerate(self.stages))
        self.counting_answers = count_answers
        total = (sum(self.sched.num_partners(a, j) for a in range(len(self.prefixes))
                     for j in range(k))
                 if self.counting_answers else len(self.prefixes))
        self.progress = Progress(self.label, total=total, unit=unit)
        self.finished = 0

        self.readouts = ReadoutGroups(self.torch, self.stages)

    def _stop_width(self, stop_groups, budget) -> int:
        """Anchors of one stop key group in flight at once.

        One per group is the most a group can save, but with fewer
        groups than a chunk holds it would leave the chunk mostly
        empty, so the width grows to keep about one chunk in flight.
        """
        if stop_groups is None or not self.prefixes:
            return 1
        groups = len(set(stop_groups)) or 1
        mean = (sum(self.prefix_lengths) / len(self.prefix_lengths)
                + float(np.mean(self.suffixes[0].lengths)))
        per_chunk = max(1, int(budget // max(1.0, mean)))
        return max(1, -(-per_chunk // groups))

    def _suffixes_of(self, j, a):
        """Return the suffixes document a requests at stage j."""
        return (self.own[(j, a)].suffixes if self.stages[j].own_suffixes
                else self.suffixes[j])

    def _entry_rows(self, a, j, start, end, carried):
        if self.bulk[j] and not carried:
            return max(0, int(self.stages[j].costs(np.array([a]))[0]))
        f = self.prefix_lengths[a]
        sliced = self.sched.slices.get(a) if carried else None
        if sliced is not None:
            first, last = sliced
            if end == start:
                return last - first
            f = last - first
        indices = self.sched.partner_indices(a, j, start, end)
        lengths = self._suffixes_of(j, a).lengths_at(indices)
        rows = ((f if carried else 0) + int(lengths.sum())
                + self.canvas_rows[j] * len(lengths))
        if self.writes[j] and start == 0:
            rows += len(self.frames[j]) + len(self.canvas)
        return rows

    def _rows_by_stage(self, part):
        """Return the rows each stage packs into a chunk, pricing rounds together."""
        by_stage = {}
        rounds = {}
        for entry in part:
            j = entry[1]
            if self.bulk[j] and not entry[4]:
                rounds.setdefault(j, []).append(entry[0])
            else:
                by_stage[j] = by_stage.get(j, 0) + self._entry_rows(*entry)
        for j, anchors in rounds.items():
            costs = self.stages[j].costs(np.asarray(anchors, dtype=np.int64))
            by_stage[j] = by_stage.get(j, 0) + int(np.maximum(costs, 0).sum())
        return by_stage

    def _held_pages(self, key, prefix_tokens):
        # the admission prices a document at page_cost(prefix + extra)
        # less what it holds; a trimmed window holds fewer sliding
        # pages than that price assumes, so count what growing takes
        capacity = prefix_tokens + self.reserve_extra
        return (self.arena.page_cost(capacity)
                - self.arena.growth_cost(key, capacity))

    def _requests_of(self, a, key):
        """Return a stage request selector, or None if every request is selected.

        At a stage whose suffixes are each document's own, the selector
        asks the stage for the document's sequences, keeps them for the
        chunk builder, and hands the admission their token counts.
        """
        if not self.asking:
            return None
        stages = self.stages

        def ask(j):
            stage = stages[j]
            if not stage.own_suffixes:
                return None if stage.requests is None else stage.requests(key)
            got = stage.suffixes(key)
            if got is DROP or got is SKIP or isinstance(got, SkipTo):
                return got
            own = got.suffixes
            longest = (int(own.lengths[0]) if len(own) == 1
                       else int(own.lengths.max()) if len(own) else 0)
            if longest > stage.suffix_tokens:
                raise ValueError(
                    f"{stage.label or f'stage {j}'}: a document's suffix has "
                    f"{int(own.lengths.max())} tokens, over the stage's "
                    f"{stage.suffix_tokens}")
            if len(own):
                self.own[(j, a)] = got
            return OwnRequests(own.lengths)

        return ask

    def _state_borrow_check(self, can_borrow):
        """Narrow can_borrow to borrows a parent's saved state can serve.

        The borrower must start after the parent's own start and before
        its own end, at a point the parent saves a state for or has
        saved already.
        """
        def check(doc, parent, same_chunk):
            shared = self.borrowing.tree_shared[doc]
            if not shared < self.prefix_lengths[doc]:
                return False
            if not shared > self.borrowing.shared(parent):
                return False
            # the parent saves a state only at the lengths it kept
            if (shared != self.prefix_lengths[parent]
                    and shared not in self.share_points[parent]):
                return False
            if not same_chunk and self.arena.shared_state(
                    self.keys[parent], shared) is None:
                return False
            return can_borrow(doc, parent, same_chunk)
        return check

    def _reserve(self, anchor, borrow):
        """Reserve a document's KV and checkpoints before admitting its work."""
        key = self.keys[anchor]
        prefix = self.prefix_lengths[anchor]
        parent, shared = (None, 0) if borrow is None else borrow
        points = tuple(t for t in self.share_points[anchor]
                       if shared < t < prefix)
        if not self.arena.reserve(
                key, prefix, capacity_tokens=prefix + self.reserve_extra,
                borrow=(self.keys[parent], shared) if shared else None,
                extension=self._extends, share_points=points):
            return False
        if parent is not None:
            self.arena.read_state((self.keys[parent],))
            self._reserved_parents[anchor] = self.keys[parent]
        if (anchor not in self._held_parents and anchor < len(self.borrowers)
                and self.borrowers[anchor]):
            self._held_parents.add(anchor)
            self.arena.hold(key, self.borrowers[anchor])
            self.arena.keep_window(key, self.lowest_borrow[anchor])
        if self.growth is not None:
            lent = (self.lowest_borrow.get(anchor, 0) // self.arena.page_tokens
                    if anchor < len(self.borrowers) and self.borrowers[anchor] else 0)
            self.growth.admitted(anchor, lent)
        return True

    def _merged(self, j, start, end):
        """Return whether the group packs its frame and suffix as one entry."""
        return self.stages[j].single and end - start == 1

    def _build(self, chunk_groups):
        """Pack a chunk and identify each entry's stage and answer row count.

        Returns:
            The chunk; its entries, (stage, answer rows) per readout
            answer in order; per answer row, the draft token fed after
            it, or -1; per group, the index of its answer, or None for
            a prefix slice; and per device-fed round, in chunk order,
            the pass and answer index of the answer it follows.
        """
        from quail.backends.quail.executor.chunk import Suffixes, pack_chunk

        specs = []
        entries = []
        following = []
        group_entries = []
        sources = []
        appended = {}

        def entry(j, rows, fed=None):
            entries.append((j, rows))
            if fed is None:
                fed = (_NO_FOLLOWING[:rows] if rows <= len(_NO_FOLLOWING)
                       else np.full(rows, -1, dtype=np.int64))
            following.append(fed)

        bulk = {}
        for a, j, start, end, carried in chunk_groups:
            key = self.keys[a]
            f = self.prefix_lengths[a]
            frame = self.frames[j]
            stage = self.stages[j]
            append = stage.append
            if self.bulk[j] and not carried:
                # the rounds of this stage are packed together, after
                # every other group; _run_one orders the groups so
                bulk.setdefault(j, []).append(a)
                group_entries.append(("bulk", j, len(bulk[j]) - 1))
                continue
            sliced = self.sched.slices.get(a) if carried else None
            if sliced is not None and end == start:
                # a slice of the prefix alone: its rows read the slices
                # before it from the key's pages and are written after
                # them; nothing is asked of it yet
                first, last = sliced
                # the tree path's f is the resident rows the slice
                # reads; the unified path's is the sequence's length
                specs.append(dict(
                    key=key, prefix=self.prefixes[a][first:last], start=first,
                    f=first if self.mode == "tree" else last,
                    suffixes=Suffixes.of([])))
                group_entries.append(None)
                continue
            request = self.own.get((j, a))
            if (request is not None and not carried and append
                    and stage.single and end - start == 1
                    and not self.writes[j] and not stage.rewind
                    and stage.read_all_rows and stage.paths is None
                    and stage.canvas is None and request.read_rows is None
                    and self.read_rows[j] is None
                    and self.read_masks[j] is None):
                # a decode round: the document's own one suffix, kept
                # after the path earlier rounds appended; the general
                # bookkeeping below reduces to this
                base = f + len(frame)
                position = request.position
                if position is None:
                    position = base + max(0, self.appended[a] - request.discard)
                sufs = request.suffixes
                total = int(sufs.lengths[0])
                kept = (total if request.write_suffix_tokens is None
                        else request.write_suffix_tokens)
                appended[a] = position - base + kept
                spec = dict(
                    key=key, prefix=None, f=position,
                    suffixes=sufs, read_all_rows=True,
                    write_suffix_tokens=kept, single=True)
                specs.append(spec)
                entry(j, total, np.concatenate((sufs.ids[1:], _NO_FOLLOWING[:1])))
                group_entries.append(len(entries) - 1)
                continue
            # a written frame, or a rewinding stage, replaces any path
            # an earlier stage kept
            rewinds = self.writes[j] or self.stages[j].rewind
            path = 0 if rewinds and start == 0 else self.appended[a]
            if request is not None:
                path = max(0, path - request.discard)
                if request.position is not None:
                    path = request.position - f - len(frame)
            parent, shared = ((self.borrowing.parent(a), self.borrowing.shared(a))
                              if carried else (None, 0))
            prefix = None
            if sliced is not None:
                # the rest of a prefix whose first slices are resident
                shared = sliced[0]
                prefix = self.prefixes[a][shared:]
            elif carried:
                prefix = self.prefixes[a][shared:] if shared else self.prefixes[a]
            indices = self.sched.partner_indices(a, j, start, end)
            sufs = self._suffixes_of(j, a).take(indices)
            read_all = self.stages[j].read_all_rows
            rows = int(sufs.lengths.sum()) if read_all else len(sufs)
            own = {}
            read_rows = (request.read_rows if request is not None
                         else self.read_rows[j])
            if read_rows is not None:
                own["read_rows"] = np.asarray(read_rows)[np.asarray(list(indices))]
                rows = int(own["read_rows"].sum())
            if self.read_masks[j] is not None:
                own["read_mask"] = self.read_masks[j]
                rows = int(own["read_mask"].sum())
            if self.stages[j].canvas is not None:
                ids = np.asarray(self.stages[j].canvas(a))
                if ids.ndim == 2:
                    ids = ids[np.asarray(list(indices), dtype=np.int64)]
                own = dict(canvas=ids)
                if read_all:
                    rows = self.stages[j].canvas_rows * len(sufs)
            # under tree attention a borrowing document with one
            # suffix reads its parent's pages stacked with its siblings
            borrowed = shared and parent is not None
            read_key = self.keys[parent] if borrowed and self.stages[j].single else None
            carry = {}
            if prefix is not None:
                carry = dict(
                    borrow_key=self.keys[parent] if borrowed else None,
                    share_points=self.share_points[a])
            kept = int(sufs.lengths.sum()) if append else 0
            if request is not None and request.write_suffix_tokens is not None:
                kept = request.write_suffix_tokens
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
                    read_all_rows=read_all, **carry, **own))
            elif self.writes[j] and start == 0:
                # frame entry: scatter the frame into KV after the
                # document rows; its own answer row means nothing
                specs.append(dict(
                    key=key, prefix=prefix, start=shared, read_key=read_key,
                    f=f, suffixes=Suffixes(self.frame_ids[j], [len(frame)]),
                    write_suffix_tokens=len(frame), **carry))
                entry(j, 1)
                specs.append(dict(
                    key=key, prefix=None, f=f + len(frame),
                    suffixes=sufs, read_all_rows=read_all,
                    write_suffix_tokens=kept,
                    paths=self.stages[j].paths, **own))
            else:
                specs.append(dict(
                    key=key, prefix=prefix, start=shared, read_key=read_key,
                    f=f + len(frame) + path,
                    suffixes=sufs, read_all_rows=read_all,
                    write_suffix_tokens=kept,
                    single=self.stages[j].single and end - start == 1,
                    paths=self.stages[j].paths, **carry, **own))
            entry(j, rows)
            group_entries.append(len(entries) - 1)
        firsts = {}
        for j, anchors in bulk.items():
            firsts[j] = len(entries)
            requests = self.stages[j].rounds(np.asarray(anchors, dtype=np.int64))
            lengths = requests.lengths
            ends = np.cumsum(lengths)
            entries.extend((j, int(length)) for length in lengths.tolist())
            fed = np.concatenate((requests.ids[1:], _NO_FOLLOWING[:1]))
            fed[ends - 1] = -1
            following.append(fed)
            # a round's position replaces the appended-token bookkeeping
            spec = dict(rounds=requests, keys=[self.keys[a] for a in anchors],
                        feed_bound=self.stages[j].suffix_tokens)
            if self.mode == "tree" and self.stages[j].document_reads:
                spec["document_reads"] = self._document_reads(anchors)
            specs.append(spec)
            sources.extend(self.latest[anchors[index]]
                           for index in np.flatnonzero(requests.device_fed))
        if bulk:
            group_entries = [firsts[e[1]] + e[2] if isinstance(e, tuple) else e
                             for e in group_entries]
        chunk = pack_chunk(
            self.torch, self.arena, specs, attention_mode=self.mode,
            staging=self.staging, canvas=self.canvas, answer_row=self.answer_row)
        for a, tokens in appended.items():
            self.appended[a] = tokens
        return chunk, entries, following, group_entries, sources

    def _document_reads(self, anchors):
        """Return each round's document group and the tokens it reads there.

        A borrower's group is its parent, and a parent is its own
        group. A group reads the fewest tokens any of its members
        borrows, in whole pages. A group with one round in the chunk
        reads nothing apart: its pages stay in its call B read.
        """
        groups = np.empty(len(anchors), dtype=np.int64)
        shared = np.zeros(len(anchors), dtype=np.int64)
        for index, a in enumerate(anchors):
            parent = self.borrowing.parent(a)
            if parent is not None:
                groups[index], shared[index] = parent, self.borrowing.shared(a)
            else:
                groups[index], shared[index] = a, self.lowest_borrow.get(a, 0)
        _, inverse, sizes = np.unique(groups, return_inverse=True,
                                      return_counts=True)
        least = np.full(len(sizes), np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(least, inverse, shared)
        shared = least[inverse]
        shared[sizes[inverse] < 2] = 0
        return groups, shared

    def _ordered(self, part):
        """Return the groups with the bulk stages' rounds last."""
        if not any(self.bulk):
            return part
        rounds = [g for g in part if self.bulk[g[1]] and not g[4]]
        if not rounds:
            return part
        return [g for g in part if not (self.bulk[g[1]] and not g[4])] + rounds

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
        if self.growth is not None:
            self.growth.settled(settlement.anchor)

    def _report(self, entry):
        if entry.values is None:
            # waits for the pass's readout, so this is GPU time, not host
            started = time.perf_counter()
            entry.values = self.readouts.results(entry.handles)
            self.host_s["wait"] += time.perf_counter() - started
            entry.positions = np.zeros(len(self.readouts), dtype=np.int64)
        groups = entry.groups[entry.at:entry.at + entry.step]
        values, position = entry.values, entry.positions
        transitions = []
        group_rows = []
        prepared = {}
        for a, j, start, end, _ in groups:
            if end == start:
                group_rows.append(None)     # a prefix slice has no answer
                continue
            key = self.readouts.group_key[j]
            if self.writes[j] and start == 0 and not self._merged(j, start, end):
                position[key] += 1        # the frame entry's answer means nothing
            count = end - start
            rows = values[key][position[key]:position[key] + count]
            position[key] += count
            group_rows.append(rows)
            if self.stages[j].prepare is not None:
                anchors, stage_rows = prepared.setdefault(j, ([], []))
                anchors.append(a)
                stage_rows.append(rows)
        started = time.perf_counter()
        for j, (anchors, stage_rows) in prepared.items():
            self.stages[j].prepare(anchors, stage_rows)
        self.host_s["decide"] += time.perf_counter() - started
        for (a, j, start, end, _), rows in zip(groups, group_rows):
            if end == start:
                continue        # a prefix slice: nothing was asked
            if (self.stages[j].own_suffixes and not self.ahead[j]
                    and end == self.sched.num_partners(a, j)):
                # the request was packed; a repeating stage asks anew,
                # and an ahead stage has asked already
                del self.own[(j, a)]
            result = self.sched.report(a, j, start, end, rows)
            if self.on_answers is not None:
                self.on_answers(a, j, start, end, rows)
            for settlement in result.settlements:
                self._event(settlement)
            transitions.extend(result.transitions)
        entry.at += len(groups)
        self.progress.update(
            self.progress.done + sum(end - start for _, _, start, end, _ in groups)
            if self.counting_answers else self.finished)
        if transitions and self.on_chunk is not None:
            self.on_chunk(transitions)
        self._release_skipped()

    def _report_next(self, partial=False):
        """Decide the oldest results, splitting append stages when no work is ready."""
        entry = self.outstanding[0]
        if not entry.step:
            entry.step = len(entry.groups)
            if partial and self.split_answers and all(
                    self.stages[j].append for _, j, _, _, _ in entry.groups):
                entry.step = (entry.step + 1) // 2
        self._report(entry)
        if entry.at == len(entry.groups):
            self.outstanding.pop(0)

    def _release_skipped(self):
        """Let go of what the documents a stop key skipped would have used.

        A skipped document never packs, so its parent's hold for it is
        released, or not counted when the parent is still queued, and
        its own borrowers pack their whole prefix.
        """
        for a in self.sched.take_skipped():
            self.borrowing.skip(a)
            parent = (self.borrowing.tree_parent[a]
                      if a < len(self.borrowing.tree_parent) else None)
            if parent is not None:
                if parent in self.borrowing.admitted:
                    self.arena.release(self.keys[parent])
                else:
                    self.borrowers[parent] -= 1
            if self.arena.is_resident(self.keys[a]):
                self.arena.free_key(self.keys[a])
            if self.growth is not None:
                self.growth.settled(a)

    def _log_memory_event(self, kind, **detail):
        log = getattr(self.arena, "log_event", None)
        if log is not None:
            reason = detail.get("reason", "")
            if reason:
                detail["reason"] = ("slots" if "checkpoint slot" in reason
                                    else "pages" if "pages" in reason
                                    else "rows")
            log(kind, **detail)

    def _run_part(self, part):
        """Run one chunk of groups, halving it when its pages do not fit."""
        from quail.backends.quail.executor.chunk import ArenaFullError

        try:
            self._run_one(part)
        except ArenaFullError as error:
            # retained KV nothing here reads makes room first
            need = self.arena.page_cost(sum(self._entry_rows(*e) for e in part))
            if self.arena.evict_retained(need):
                self._log_memory_event("evict_retry", groups=len(part))
                logger.info("chunk of %d groups retried after evicting "
                            "retained KV", len(part))
                self._run_part(part)
                return
            if len(part) < 2:
                raise
            self._log_memory_event("split", groups=len(part), reason=str(error))
            logger.info("chunk of %d groups split: %s", len(part), error)
            half = len(part) // 2
            self._run_part(part[:half])
            self._run_part(part[half:])

    def _launched(self, part):
        """Tell each stage with a launched callback which documents it launched."""
        by_stage = {}
        for a, j, start, end, _ in part:
            if end > start and self.stages[j].launched is not None:
                by_stage.setdefault(j, []).append(a)
        for j, anchors in by_stage.items():
            self.stages[j].launched(anchors)

    def _run_one(self, part):
        started = time.perf_counter()
        part = self._ordered(part)
        chunk, entries, following, group_entries, sources = self._build(part)
        if sources:
            # the rounds follow answers of the pass before, whose
            # readout is still on the stream
            previous = self.outstanding[-1] if self.outstanding else None
            if previous is None or any(
                    pass_index != previous.pass_index for pass_index, _ in sources):
                raise AssertionError(
                    "a device-fed round follows an answer already decided")
            chunk.apply_feed(self.torch, previous.feed,
                             [index for _, index in sources], self.arena.page_tokens)
        self.pack_s += time.perf_counter() - started
        self.tokens += chunk.tokens
        if self.gpu_timing:
            e0 = self.torch.cuda.Event(enable_timing=True)
            e1 = self.torch.cuda.Event(enable_timing=True)
            e0.record()
        graphs = self.pipeline.decode_graphs
        before = None if graphs is None else _replay_counts(graphs)
        started = time.perf_counter()
        normed = forward_chunk(self.pipeline, self.arena, chunk)
        launched = time.perf_counter() - started
        self.host_s["launch"] += launched
        after = None if graphs is None else _replay_counts(graphs)
        kind = ("launch_eager" if before == after
                else "launch_graph" if after[0] != before[0] else "launch_pieces")
        self.host_s[kind] += launched
        if self.gpu_timing:
            e1.record()
            self.spans.append((self._rows_by_stage(part), chunk.tokens, e0, e1))
        for key in chunk.fresh_keys:
            self.arena.trim_window(key)
        # the readout is queued right behind the pass, so the stream
        # never waits for the host; the earlier passes' readouts are
        # done by now, so their answers are decided while the GPU runs
        # this pass and its readout
        started = time.perf_counter()
        if self.gpu_timing:
            e2 = self.torch.cuda.Event(enable_timing=True)
            e2.record()
        handles = self.readouts.submit(
            normed, entries, chunk.rows_per_answer)
        feed = None
        if self.feeding and entries:
            feed = self.readouts.feed(handles, np.concatenate(following),
                                      entries, self.ahead)
        for (a, _, _, _, _), index in zip(part, group_entries):
            if index is not None:
                self.latest[a] = (self.passes, index)
        self._launched(part)
        submitted = time.perf_counter()
        self.host_s["submit"] += submitted - started
        if self.gpu_timing:
            e3 = self.torch.cuda.Event(enable_timing=True)
            e3.record()
            # e1 to e2 is the stream idle until the host submits the
            # readout; e2 to e3 is the readout
            self.readout_spans.append((e1, e2, e3))
        while self.outstanding:
            self._report_next()
        self.host_s["report"] += time.perf_counter() - submitted
        self.outstanding.append(PendingAnswers(part, handles, self.passes, feed))
        self.passes += 1

    def run(self):
        """Drive admission and drain the launched answer handles in order."""
        run_started = time.perf_counter()
        graphs = self.pipeline.decode_graphs
        replays = 0 if graphs is None else graphs.replays
        pieces = (0 if graphs is None or graphs.pieces is None
                  else graphs.pieces.replays)
        fallbacks = {} if graphs is None else dict(graphs.fallbacks)
        while True:
            # documents with nothing to send at their first stage never run
            for settlement in self.sched.take_settled():
                self._event(settlement)
            if self.sched.done():
                break
            if len(self.outstanding) > 1:
                started = time.perf_counter()
                self._report_next()
                self.host_s["report"] += time.perf_counter() - started
            if self.sched.blocked_pages:
                # the free list is short for the next fresh document:
                # retained KV nothing here reads makes room
                self.arena.evict_retained(self.sched.blocked_pages)
            started = time.perf_counter()
            if self.growth is not None:
                # the ready documents' next requests get their pages
                # now; the ones that cannot wait a pass
                self.sched.stalled = self.growth.grow(
                    list(self.sched.ready), self.arena.free_pages)
            groups = self.sched.next_chunk(
                self.arena.free_pages if self.paged else 1 << 62)
            self.host_s["admit"] += time.perf_counter() - started
            if not groups:
                if self.outstanding:
                    started = time.perf_counter()
                    self._report_next(partial=True)
                    self.host_s["report"] += time.perf_counter() - started
                    continue
                if (self.sched.blocked_pages
                        and self.arena.evict_retained(self.sched.blocked_pages)):
                    continue
                # parents freed but held for queued children hold the
                # pages: the children pack their whole prefixes
                if self.arena.drop_holds(self.held):
                    continue
                raise AssertionError(
                    f"nothing buildable and nothing in flight; "
                    f"{len(self.sched.stalled)} documents wait for KV pages")
            self._run_part(groups)
            started = time.perf_counter()
            # the chunk is built, so the anchors launched ahead of
            # their answers can be asked for their next request
            self.sched.advance_ahead()
            # the packed children hold their parents' pages now
            for a, _, start, _, carried in groups:
                parent = (self.borrowing.tree_parent[a]
                          if a < len(self.borrowing.tree_parent) else None)
                if carried and start == 0 and parent is not None:
                    self.arena.release(self.keys[parent])
                if a in self._reserved_parents:
                    self.arena.done_reading((self._reserved_parents.pop(a),))
            self.host_s["advance"] += time.perf_counter() - started
        while self.outstanding:
            self._report_next()
        self._release_skipped()
        # a limit ends the run with documents still queued
        for anchor in self.sched.drain():
            if self.arena.is_resident(self.keys[anchor]):
                self.arena.free_key(self.keys[anchor])
            if self.growth is not None:
                self.growth.settled(anchor)
        self.progress.finish(f"{self.label} done", f"{self.tokens:,} fresh tokens")
        self.arena.drop_holds(self.held)
        if self.sched.oversized:
            logger.warning(
                "%s: skipped %d documents too long for one forward pass; "
                "they get no answer", self.label, len(self.sched.oversized))
        if self.stats is not None:
            self.stats["borrowed_tokens"] = self.borrowing.borrowed_tokens
            self.stats["pack_s"] = self.pack_s
            self.stats["skipped_documents"] = len(self.sched.oversized)
            self.stats["host_s"] = {k: round(v, 3) for k, v in self.host_s.items()}
            self.stats["run_s"] = round(time.perf_counter() - run_started, 3)
            self.stats["graph_replays"] = (0 if graphs is None
                                           else graphs.replays - replays)
            self.stats["piece_replays"] = (
                0 if graphs is None or graphs.pieces is None
                else graphs.pieces.replays - pieces)
            self.stats["graph_fallbacks"] = (
                {} if graphs is None else
                {reason: count - fallbacks.get(reason, 0)
                 for reason, count in graphs.fallbacks.items()
                 if count != fallbacks.get(reason, 0)})
            if self.growth is not None:
                self.stats["kv_growth"] = self.growth.stats()
        if self.spans:
            self.torch.cuda.synchronize()
            if self.stats is not None:
                self.stats["readout_gpu_s"] = round(sum(
                    e2.elapsed_time(e3) for _, e2, e3 in self.readout_spans)
                    / 1000.0, 3)
                self.stats["idle_gpu_s"] = round(sum(
                    e1.elapsed_time(e2) for e1, e2, _ in self.readout_spans)
                    / 1000.0, 3)
        return self.sched.answers, self.spans, self.tokens
