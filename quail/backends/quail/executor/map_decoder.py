"""Map and extraction answers through the same decode-row scheduler.

A prompt's final row chooses its first answer token. Map answers then feed
one row per pass; extraction answers feed that token and document-copy drafts.
Token choice and draft verification run on the GPU. One nonblocking copy per
pass returns tokens and surviving document positions. Copy answers wait for
their previous result before laying out rows at the accepted prefix's end.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from quail.backends.quail.executor.chunk import DecodeRows

# copies of chosen tokens kept in flight before the host waits on the
# oldest; two lets the host read a pass's tokens while the next runs
IN_FLIGHT = 2
# the share of a map's answers whose whole length admission reserves; a
# longer answer grows its pages and may be evicted. 1.0 reserves the limit
ANSWER_QUANTILE = 0.9


@dataclass
class _Copy:
    """One pass's chosen tokens on their way to the host."""

    event: object
    host: object
    documents: np.ndarray
    indices: np.ndarray     # each token's position in its answer
    maps: np.ndarray        # each token's map, when it was chosen


class MapDecoder:
    """Every map answer of one pipeline run, in arrays by document.

    Args:
        torch: The torch module.
        head: The model's whole output head, (vocabulary, hidden), on the GPU.
        num_documents: Documents in the run.
        limits: Per map, the most tokens its answer may have.
        stop_ids: Token ids that end an answer.
        page_tokens: Tokens per arena page.
        expected_tokens: Prior mean answer lengths, or min(64, limit) per map.
    """

    def __init__(self, torch, head, num_documents: int, limits, stop_ids,
                 page_tokens: int, expected_tokens=None):
        self.torch = torch
        self.head = head
        self.limits = np.asarray(limits, dtype=np.int64)
        self.expected_tokens = (np.minimum(64, self.limits) if expected_tokens is None
                                else np.asarray(expected_tokens, dtype=np.float64))
        self.page_tokens = page_tokens
        n = num_documents
        self.stop_ids = np.asarray(sorted(stop_ids), dtype=np.int64)
        # per map, each document's tokens and its answer's length, -1
        # while the answer has not ended
        self.tokens = [np.zeros((n, int(limit)), dtype=np.int32)
                       for limit in self.limits]
        self.lengths = [np.full(n, -1, dtype=np.int64) for _ in self.limits]
        self.written = [np.zeros(n, dtype=np.int64) for _ in self.limits]
        self.paused = np.zeros(n, dtype=bool)
        # the answer each document is writing now
        self.map = np.full(n, -1, dtype=np.int64)      # its map, -1 for none
        self.origin = np.zeros(n, dtype=np.int64)      # its first decode position
        self.fed = np.zeros(n, dtype=np.int64)         # decode rows packed
        self.stopped = np.zeros(n, dtype=bool)         # read a stop token
        self.pages = np.zeros((n, 1), dtype=np.int32)  # its pages, padded
        self.num_pages = np.zeros(n, dtype=np.int64)
        # a slot borrows its first borrowed[slot] tokens' pages from
        # parent[slot]; lead[slot] is the most any slot borrows from it
        self.parent = np.arange(n, dtype=np.int64)
        self.borrowed = np.zeros(n, dtype=np.int64)
        self.lead = np.zeros(n, dtype=np.int64)
        self.device = head.device
        self.next_token = None
        self.copies = deque()
        # per slot, the token copies on their way to the host that hold
        # one of its tokens; an evicted answer recomputes once it has none
        self.inflight = np.zeros(n, dtype=np.int64)
        self.finished = []      # (document, map) whose answers ended
        self.rows_fed = np.zeros(len(self.limits), dtype=np.int64)  # per map
        # KV tokens the decode rows read: in all, alone, through their
        # own pages beside a shared read, and through shared reads
        self.read_tokens = dict(all=0, alone=0, own=0, shared=0)
        # Per answer stage, each branch's map index.
        self.slot_maps = {}
        self.answer_slots = {}
        # whether a document's branches read its shared pages together
        self.share = False
        self.copy_constraints = {}
        self.copy_readouts = {}
        self.copy_groups = []
        self.copy_inputs = []
        self.copy_pending = np.zeros(n, dtype=bool)
        self.copy_words = 0
        self.copy_trace_words = 0
        self.copy_trace = None
        self.constrained = np.zeros(len(self.limits), dtype=bool)
        self.copy_document = np.full(n, -1, dtype=np.int64)
        self.copy_sources = np.full(n, -1, dtype=np.int64)
        self.num_streams = np.zeros(n, dtype=np.int64)
        self.steps = [np.zeros(n, dtype=np.int64) for _ in self.limits]
        self.drafts = np.zeros(n, dtype=np.int64)

    def configure_copy(self, first, slots, documents, specs):
        """Register extraction fields for setup after the first forward launch."""
        self.constrained[first:first + len(specs)] = True
        self.copy_inputs.append((first, slots, documents, specs))

    def _initialize(self):
        if self.next_token is None:
            self.next_token = self.torch.zeros(
                len(self.map), dtype=self.torch.int64, device=self.device)
        pending, self.copy_inputs = self.copy_inputs, []
        for first, slots, documents, specs in pending:
            self._configure_copy(first, slots, documents, specs)

    def _configure_copy(self, first, slots, documents, specs):
        from quail.backends.quail.executor.extract.copy import CopyConstraint
        from quail.backends.quail.executor.readout import AsyncTopLogprobs
        from quail.cost.draft_policy import LIVE_BUCKETS

        key = specs[0].top_tokens, tuple(documents.vocabulary.stop_ids)
        if key not in self.copy_readouts:
            self.copy_readouts[key] = (AsyncTopLogprobs(
                self.torch, self.torch.nn.functional, self.head, key[0], [key[1]]), [])
        _, questions = self.copy_readouts[key]
        questions.extend(range(first, first + len(specs)))
        constraint = CopyConstraint(self.torch, self.device, documents, specs[0])
        self.copy_words = max(self.copy_words, constraint.mask_words)
        self.copy_trace_words = max(self.copy_trace_words, constraint.trace_width)
        self.copy_trace = constraint.trace or self.copy_trace
        streams = constraint.stream_ids.repeat(len(specs), 1)
        index = np.zeros(len(self.map), dtype=np.int64)
        index[slots.flatten()] = np.arange(slots.size)
        group = SimpleNamespace(
            constraint=constraint, key=key,
            questions=np.arange(first, first + len(specs)), index=index,
            streams=streams, alive=streams >= 0,
            widths=np.full(slots.size, streams.shape[1], dtype=np.int64),
            source=self.torch.zeros(slots.size, dtype=self.torch.int64,
                                    device=self.device),
            masks=np.zeros((slots.size, constraint.mask_words), dtype=np.int64),
            prepared_written=np.full(slots.size, -1, dtype=np.int64),
            prepared_drafts=np.zeros(slots.size, dtype=np.int64),
            lookahead=np.full(slots.size, -1, dtype=np.int64))
        self.copy_groups.append(group)
        for q, spec in enumerate(specs):
            self.copy_document[slots[q]] = np.arange(len(documents.documents))
            self.copy_constraints[first + q] = SimpleNamespace(
                constraint=constraint, spec=spec,
                sources=np.full(len(self.map), -1, dtype=np.int64),
                masks=group.masks[q * slots.shape[1]:(q + 1) * slots.shape[1]],
                trials=np.zeros(len(LIVE_BUCKETS)), accepts=np.zeros(len(LIVE_BUCKETS)),
                rho=np.array([document.rho for document in documents.documents]))

    def row_counts(self, running):
        """Return one chosen-token row and each answer's copy draft rows."""
        from quail.backends.quail.executor.extract.copy import draft_lengths

        counts = np.ones(len(running), dtype=np.int64)
        written = np.zeros(len(running), dtype=np.int64)
        sizes = np.zeros(len(running), dtype=np.int64)
        for q, state in self.copy_constraints.items():
            mine = self.map[running] == q
            slots = running[mine]
            if not len(slots):
                continue
            written[mine] = self.written[q][slots]
            sizes[mine] = draft_lengths(
                state.spec, self.lengths[q], written[mine], self.num_streams[slots],
                state.rho[self.copy_document[slots]], state.trials, state.accepts)
        for group in self.copy_groups:
            mine = np.isin(self.map[running], group.questions)
            slots = running[mine]
            if not len(slots):
                continue
            constraint = group.constraint
            local = group.index[slots]
            if np.any((written[mine] > 0) & (
                    (group.prepared_written[local] != written[mine])
                    | (group.prepared_drafts[local] < sizes[mine]))):
                raise AssertionError("copy inputs were not prepared before packing")
            position, option = np.divmod(self.copy_sources[slots],
                                         constraint.num_options)
            remaining = (1 + constraint.filler_lengths[option]
                         + constraint.position_lengths[position] - written[mine])
            counts[mine] += np.minimum(sizes[mine], np.maximum(0, remaining))
        self.drafts[running] = counts - 1
        return counts

    def _prepare_copy(self, group, slots, written, drafts):
        local = group.index[slots]
        prepare = ((written > 0) & ((group.prepared_written[local] != written)
                                   | (group.prepared_drafts[local] < drafts)))
        if not prepare.any():
            return
        slots, local = slots[prepare], local[prepare]
        tokens = np.empty(len(slots), dtype=np.int64)
        for q in group.questions:
            mine = self.map[slots] == q
            tokens[mine] = self.tokens[q][slots[mine], 0]
        group.constraint.prepare(
            self.copy_document[slots], tokens, group.masks[local],
            written[prepare], drafts[prepare], self.copy_sources[slots])
        group.prepared_written[local] = written[prepare]
        group.prepared_drafts[local] = drafts[prepare]

    def output_answers(self, map_index, slots):
        """Return extraction text and original character spans for a field."""
        if map_index not in self.copy_constraints:
            return [None] * len(slots), [None] * len(slots)
        state = self.copy_constraints[map_index]
        return state.constraint.output_answers(
            self.copy_document[slots], state.sources[slots],
            self.lengths[map_index][slots])

    # ---- documents entering and leaving -----------------------------------

    def start(self, document: int, map_index: int, position: int, pages,
              parent: int | None = None, shared_tokens: int = 0) -> None:
        """Begin an answer whose first token the current pass chooses.

        Args:
            document: The document's index in the run.
            map_index: The map the answer is for.
            position: The position of the answer's first decode row,
                right after its prompt.
            pages: The document's page ids in logical order.
            parent: The slot whose whole pages this one borrows, or None.
            shared_tokens: The tokens it borrows from the parent.
        """
        if self.map[document] >= 0:
            raise ValueError(f"document {document} is already writing an answer")
        if map_index in self.slot_maps:
            map_index = self.slot_maps[map_index][document]
        self.set_pages(document, pages)
        self.map[document] = map_index
        self.origin[document] = position
        self.fed[document] = 0
        self.stopped[document] = False
        self.copy_sources[document] = -1
        self.copy_pending[document] = False
        self.num_streams[document] = 0
        self.drafts[document] = 0
        self.parent[document] = document
        self.borrowed[document] = 0
        self.lead[document] = 0
        if parent is not None and shared_tokens:
            self.parent[document] = parent
            self.borrowed[document] = shared_tokens
            self.lead[parent] = max(self.lead[parent], shared_tokens)

    def set_pages(self, document, pages):
        """Update one slot's table after allocation or growth."""
        pages = np.asarray(pages, dtype=np.int32)
        if len(pages) > self.pages.shape[1]:
            wider = np.zeros((len(self.map), len(pages)), dtype=np.int32)
            wider[:, :self.pages.shape[1]] = self.pages
            self.pages = wider
        self.pages[document, :len(pages)] = pages
        self.num_pages[document] = len(pages)

    def add_pages(self, documents, pages) -> None:
        """Append one page to each listed slot's table."""
        width = int(self.num_pages[documents].max()) + 1
        if width > self.pages.shape[1]:
            wider = np.zeros((len(self.map), width), dtype=np.int32)
            wider[:, :self.pages.shape[1]] = self.pages
            self.pages = wider
        self.pages[documents, self.num_pages[documents]] = pages
        self.num_pages[documents] += 1

    def pause(self, document):
        """Pause an answer and return slots whose shared-page identity changed."""
        self.paused[document] = True
        children = np.flatnonzero(self.parent == document)
        self.parent[children] = children
        self.borrowed[children] = 0
        self.parent[document] = document
        self.borrowed[document] = 0
        self.lead.fill(0)
        np.maximum.at(self.lead, self.parent, self.borrowed)
        return np.union1d(children, [document])

    def resume(self, document, pages):
        """Continue after prefilling the prompt and every saved answer token."""
        self.set_pages(document, pages)
        self.fed[document] = self.written[self.map[document]][document]
        self.paused[document] = False
        self.drafts[document] = 0

    def eviction_order(self):
        """Running slots ordered by longest expected remaining answer first."""
        from quail.backends.quail.executor.answer_survival import remaining_tokens

        running = np.flatnonzero((self.map >= 0) & ~self.paused)
        scores = np.zeros(len(self.map))
        written = np.zeros(len(self.map), dtype=np.int64)
        for q in np.unique(self.map[running]):
            curve = remaining_tokens(self.lengths[q], self.written[q],
                                     self.expected_tokens[q], int(self.limits[q]))
            mine = running[self.map[running] == q]
            written[mine] = self.written[q][mine]
            scores[mine] = curve[written[mine]]
        return running[np.lexsort((running, written[running], -scores[running]))]

    def answer_reserve(self) -> np.ndarray:
        """Per map, the tokens ANSWER_QUANTILE of its answers stay within."""
        from quail.backends.quail.executor.answer_survival import quantile_tokens

        return np.array([
            quantile_tokens(self.lengths[q], self.written[q],
                            self.expected_tokens[q], int(self.limits[q]),
                            ANSWER_QUANTILE)
            for q in range(len(self.limits))], dtype=np.int64)

    def running(self):
        """Slots whose next decode row can choose another token."""
        running = np.flatnonzero((self.map >= 0) & ~self.paused)
        ready = (~self.constrained[self.map[running]]
                 | (self.inflight[running] == 0))
        return running[ready & ~self.stopped[running] & ~self.copy_pending[running]
                       & (self.fed[running] + 1 < self.limits[self.map[running]])]

    @property
    def active(self) -> bool:
        """Whether any answer is still being written."""
        return bool((self.map >= 0).any())

    def take_finished(self) -> list:
        """Return the (document, map) pairs whose answers ended since last asked."""
        finished, self.finished = self.finished, []
        return finished

    # ---- one pass --------------------------------------------------------

    def rows(self) -> DecodeRows | None:
        """Lay out each continuing answer's chosen token and optional copy drafts.

        An answer continues while no stop token has been read and its
        tokens, counting the one this row writes, stay within its limit.
        """
        running = self.running()
        if not len(running):
            return None
        counts = self.row_counts(running)
        offsets = (np.arange(int(counts.sum()))
                   - np.repeat(np.cumsum(counts) - counts, counts))
        running = np.repeat(running, counts)
        positions = self.origin[running] + self.fed[running] + offsets
        page = positions // self.page_tokens
        if (page >= self.num_pages[running]).any():
            raise AssertionError("an answer's row is past its pages")
        width = int(self.num_pages[running].max())
        table = self.pages[running, :width]
        index = np.arange(len(running))
        write_rows = (table[index, page].astype(np.int64)
                      * self.page_tokens + positions % self.page_tokens)
        rows = DecodeRows(positions=positions, write_rows=write_rows,
                          page_table=table, slots=running)
        if not self.share:
            return rows
        # a group is a slot and the slots that borrow the most from it,
        # whose first pages are the same; its rows read those pages once
        # together when two or more of them run. A slot both borrowing
        # and lent from takes the group with the longer shared pages
        parents = self.parent[running]
        as_child = np.where(self.borrowed[running] == self.lead[parents],
                            self.borrowed[running], 0)
        as_parent = self.lead[running]
        groups = np.where(as_parent >= as_child, running, parents)
        shared = np.maximum(as_parent, as_child)
        _, inverse, counts = np.unique(np.where(shared > 0, groups, -1 - index),
                                       return_inverse=True, return_counts=True)
        together = (shared > 0) & (counts[inverse] > 1)
        if not together.any():
            return rows
        skip = np.where(together, shared // self.page_tokens, 0)
        own_width = int((self.num_pages[running] - skip).max())
        columns = skip[:, None] + np.arange(own_width)
        rows.own_table = np.where(
            columns < self.num_pages[running][:, None],
            table[index[:, None], np.minimum(columns, width - 1)], 0
        ).astype(np.int32)
        rows.own_lengths = positions + 1 - np.where(together, shared, 0)
        member = np.flatnonzero(together)
        order = member[np.argsort(groups[member], kind="stable")]
        starts = np.flatnonzero(np.r_[True, np.diff(groups[order]) != 0])
        rows.shared_rows = order
        rows.shared_cu = np.r_[starts, len(order)].astype(np.int32)
        firsts = order[starts]
        most = int((shared[firsts] // self.page_tokens).max())
        rows.shared_table = table[firsts, :most].astype(np.int32)
        rows.shared_used = shared[firsts].astype(np.int32)
        return rows

    def feed(self, chunk) -> None:
        """Copy each decode row's input token from the token buffer."""
        rows = chunk.decode_rows
        if rows is None:
            return
        torch = self.torch
        slots = torch.from_numpy(rows.slots).to(self.device, non_blocking=True)
        start = chunk.tokens - len(rows)
        inputs = self.next_token.index_select(0, slots)
        for group in self.copy_groups:
            mine = np.flatnonzero(np.isin(self.map[rows.slots], group.questions))
            if not len(mine):
                continue
            constraint = group.constraint
            selected = rows.slots[mine]
            local = constraint.tensor(group.index[selected])
            at = constraint.tensor(rows.positions[mine] - self.origin[selected])
            tokens = constraint.tokens_at(group.source[local], at)
            inputs[constraint.tensor(mine)] = tokens
        chunk.input_ids[start:].copy_(inputs)
        plain = rows.slots[~self.constrained[self.map[rows.slots]]]
        self.fed[plain] += 1
        self.rows_fed += np.bincount(self.map[rows.slots],
                                     minlength=len(self.limits))
        self._count_reads(rows, chunk.meta.get("shared") is not None)

    def _count_reads(self, rows, shared: bool) -> None:
        reads = self.read_tokens
        lengths = rows.positions + 1
        reads["all"] += int(lengths.sum())
        if not shared:
            reads["alone"] += int(lengths.sum())
            return
        member = np.zeros(len(lengths), dtype=bool)
        member[rows.shared_rows] = True
        reads["alone"] += int(lengths[~member].sum())
        reads["own"] += int(rows.own_lengths[member].sum())
        reads["shared"] += int(rows.shared_used.sum())

    def submit(self, normed, first_rows, first_documents, rows) -> None:
        """Choose the next token of each answer row and send it to the host.

        Args:
            normed: The pass's final-normed hidden states, decode rows last.
            first_rows: Rows of normed that end a map prompt and choose
                an answer's first token.
            first_documents: The documents of those rows.
            rows: The pass's DecodeRows, or None.
        """
        torch = self.torch
        count = 0 if rows is None else len(rows)
        if not len(first_rows) and not count:
            return
        self._initialize()
        documents = np.concatenate([
            np.asarray(first_documents, dtype=np.int64),
            np.empty(0, dtype=np.int64) if rows is None else rows.slots])
        # Recomputed prompts continue after their saved tokens, not at token zero.
        indices = np.concatenate([
            self.fed[np.asarray(first_documents, dtype=np.int64)],
            np.empty(0, dtype=np.int64) if rows is None else
            rows.positions - self.origin[rows.slots] + 1])
        selected = []
        if len(first_rows):
            index = torch.from_numpy(np.asarray(first_rows, dtype=np.int64))
            selected.append(normed.index_select(0, index.to(normed.device,
                                                             non_blocking=True)))
        if count:
            selected.append(normed[normed.shape[0] - count:])
        hidden = selected[0] if len(selected) == 1 else torch.cat(selected)
        width = 4 + self.copy_words + self.copy_trace_words
        values = torch.full((len(documents), width), -1, dtype=torch.int64,
                            device=self.device)
        chosen, projections = self._project(hidden, documents)
        plain = np.flatnonzero(~self.constrained[self.map[documents]])
        if len(plain):
            index = torch.from_numpy(plain).to(self.device, non_blocking=True)
            values[index, 0] = chosen[index]
            slots = torch.from_numpy(documents[plain]).to(self.device,
                                                         non_blocking=True)
            self.next_token.index_copy_(0, slots, chosen[index])
        for group in self.copy_groups:
            mine = np.flatnonzero(np.isin(self.map[documents], group.questions))
            if not len(mine):
                continue
            index = group.constraint.tensor(mine)
            values[index] = self._constrain(
                group, projections[group.key][index], documents[mine], indices[mine])
        host = torch.empty(values.shape, dtype=torch.int64,
                           pin_memory=self.head.is_cuda)
        host.copy_(values, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.copies.append(_Copy(event, host, documents, indices,
                                 self.map[documents].copy()))
        np.add.at(self.inflight, documents, 1)
        for group in self.copy_groups:
            mine = np.isin(self.map[documents], group.questions)
            if (mine & (indices == 0)).any():
                group.constraint.prepare_documents()
            mine &= indices > 0
            slots, first = np.unique(documents[mine], return_index=True)
            if len(slots):
                group.lookahead[group.index[slots]] = group.constraint.prepare_ahead(
                    self.copy_document[slots], group.masks[group.index[slots]],
                    indices[mine][first])
        while len(self.copies) > IN_FLIGHT:
            self._read(self.copies.popleft())

    def _project(self, hidden, documents):
        from quail.backends.quail.executor.readout import AsyncTopLogprobs

        torch = self.torch
        chosen = torch.empty(len(hidden), dtype=torch.int64, device=self.device)
        projections = {
            key: torch.empty((len(hidden), readout.width), device=self.device)
            for key, (readout, _) in self.copy_readouts.items()}
        maps = self.map[documents]
        for start in range(0, len(hidden), AsyncTopLogprobs.BLOCK_ROWS):
            end = min(start + AsyncTopLogprobs.BLOCK_ROWS, len(hidden))
            logits = torch.nn.functional.linear(
                hidden[start:end].to(self.head.dtype), self.head)
            chosen[start:end] = logits.argmax(dim=1)
            for key, (readout, questions) in self.copy_readouts.items():
                mine = np.flatnonzero(np.isin(maps[start:end], questions))
                if not len(mine):
                    continue
                index = torch.from_numpy(mine).to(self.device, non_blocking=True)
                projections[key][start + index] = readout.from_logits(logits[index])
        return chosen, projections

    def _constrain(self, group, projected, documents, indices):
        torch, constraint = self.torch, group.constraint
        slots, first, inverse, counts = np.unique(
            documents, return_index=True, return_inverse=True, return_counts=True)
        local = constraint.tensor(self.copy_document[slots])
        index = constraint.tensor(group.index[slots])
        limits = constraint.tensor(self.limits[self.map[slots]])
        stream_width = int(group.widths[group.index[slots]].max())
        streams = group.streams[index, :stream_width]
        alive = group.alive[index, :stream_width]
        num_streams = int(np.where(indices[first] == 0, stream_width,
                                  self.num_streams[slots]).max())
        num_streams = max(1, num_streams)
        if num_streams < stream_width:
            columns = alive.to(torch.int8).topk(num_streams, dim=1).indices
            streams = streams.gather(1, columns)
            alive = alive.gather(1, columns)
        source = group.source[index]
        active = torch.ones(len(slots), dtype=torch.bool, device=self.device)
        width = 4 + self.copy_words + self.copy_trace_words
        out = torch.full((len(documents), width), -2,
                         dtype=torch.int64, device=self.device)
        for step in range(int(counts.max())):
            valid = step < counts
            row = np.minimum(first + step, len(documents) - 1)
            at = constraint.tensor(np.where(valid, indices[row], 1))
            selected = constraint.tensor(row)
            chosen, ended, keep = constraint.choose_readout(
                projected[selected], streams, alive, at, local,
                first=bool((valid & (indices[row] == 0)).any()))
            use = active & constraint.tensor(valid)
            writes = use & ~ended
            alive = torch.where(writes[:, None], keep, alive)
            following = constraint.source(streams, alive)
            hit = chosen == constraint.tokens_at(source, at)
            values = torch.stack((torch.where(ended, -1, chosen), following,
                                  alive.sum(dim=1), hit.to(torch.int64)), dim=1)
            # Only valid rows may scatter; padded rows can alias a real row.
            valid_index = constraint.tensor(np.flatnonzero(valid))
            out[selected[valid_index], :4] = torch.where(
                use[valid_index, None], values[valid_index], -2)
            if self.copy_trace is not None:
                bits = constraint.trace_values.view(torch.int32).to(torch.int64)
                out[selected[valid_index], 4 + self.copy_words:] = bits[valid_index]
            active &= writes & hit & (at + 1 < limits)
            source = following
        group.alive[index, :stream_width] = False
        group.alive[index, :num_streams] = alive
        group.streams[index, :num_streams] = streams
        group.widths[group.index[slots]] = num_streams
        group.source[index] = source
        packed = constraint.packed_positions(streams, alive, local)
        out[:, 4:4 + constraint.mask_words] = packed[constraint.tensor(inverse)]
        return out

    def harvest(self, wait: bool = False, prepare: bool = True) -> None:
        """Read landed copies, optionally preparing their next decode rows."""
        while self.copies and (wait or self.copies[0].event.query()):
            self._read(self.copies.popleft())
        if prepare:
            self.prepare()

    def prepare(self):
        """Prepare pending copy rows while a forward pass runs, or when none fits."""
        self._initialize()
        for group in self.copy_groups:
            slots = np.flatnonzero(self.copy_pending & (self.inflight == 0)
                                   & np.isin(self.map, group.questions))
            if not len(slots):
                continue
            written = np.empty(len(slots), dtype=np.int64)
            for q in group.questions:
                mine = self.map[slots] == q
                written[mine] = self.written[q][slots[mine]]
            drafts = np.full(len(slots), group.constraint.spec.draft_tokens)
            self._prepare_copy(group, slots, written, drafts)
            self.copy_pending[slots] = False

    def _read(self, copy: _Copy) -> None:
        copy.event.synchronize()
        values = copy.host.numpy()
        documents, indices, maps = copy.documents, copy.indices, copy.maps
        np.subtract.at(self.inflight, documents, 1)
        current = ((self.map[documents] == maps) & ~self.stopped[documents]
                   & (values[:, 0] != -2))
        documents, indices, values, maps = (
            documents[current], indices[current], values[current], maps[current])
        tokens = values[:, 0]
        constrained = self.constrained[maps]
        stop = np.where(constrained, tokens == -1, np.isin(tokens, self.stop_ids))
        ended = stop | (indices + 1 >= self.limits[maps])
        for q in np.unique(maps).tolist():
            mine = maps == q
            writes = mine & (~stop if q in self.copy_constraints else True)
            self.tokens[q][documents[writes], indices[writes]] = tokens[writes]
            np.maximum.at(self.written[q], documents[writes], indices[writes] + 1)
            done = mine & ended
            self.lengths[q][documents[done]] = indices[done] + ~stop[done]
            if q in self.copy_constraints:
                self._read_copy(q, documents[mine], indices[mine], values[mine])
        self.stopped[documents[stop]] = True
        self.map[documents[ended]] = -1
        self.copy_pending[documents[ended]] = False
        self.finished.extend(zip(documents[ended].tolist(), maps[ended].tolist()))
        for group in self.copy_groups:
            slots = np.unique(documents[np.isin(maps, group.questions)])
            slots = slots[(self.map[slots] >= 0) & (self.inflight[slots] == 0)]
            written = np.empty(len(slots), dtype=np.int64)
            for q in group.questions:
                mine = self.map[slots] == q
                written[mine] = self.written[q][slots[mine]]
            local = group.index[slots]
            spec = group.constraint.spec
            needed = np.minimum(written + spec.draft_tokens,
                                spec.max_answer_tokens + spec.draft_tokens - 1)
            ready = ((group.prepared_written[local] > 0)
                     & (group.lookahead[local] >= needed))
            group.prepared_written[local[ready]] = written[ready]
            group.prepared_drafts[local[ready]] = spec.draft_tokens
            self.copy_pending[slots] = ~ready

    def _read_copy(self, q, documents, indices, values):
        from quail.cost.draft_policy import LIVE_BUCKETS

        slots, first, inverse = np.unique(
            documents, return_index=True, return_inverse=True)
        last = np.zeros(len(slots), dtype=np.int64)
        np.maximum.at(last, inverse, np.arange(len(documents)))
        state = self.copy_constraints[q]
        if self.copy_trace is not None:
            from quail.backends.quail.executor.extract.trace import write_choices

            traces = values[:, 4 + self.copy_words:].astype(np.int32).view(np.float32)
            selected = [state.constraint.documents.documents[d]
                        for d in self.copy_document[documents]]
            write_choices(self.copy_trace, selected, q, indices, values[:, 0], traces)
        state.masks[self.copy_document[slots]] = values[
            last, 4:4 + state.constraint.mask_words]
        observed = ((self.num_streams[slots] > 1) & (self.drafts[slots] > 0)
                    & (values[first, 0] >= 0) & (indices[first] + 1 < self.limits[q]))
        buckets = np.abs(np.log1p(self.num_streams[slots, None])
                         - np.log1p(LIVE_BUCKETS)).argmin(1)
        np.add.at(state.trials, buckets[observed], 1)
        np.add.at(state.accepts, buckets[observed], values[first[observed], 3] > 0)
        self.copy_sources[slots] = state.sources[slots] = values[last, 1]
        self.num_streams[slots] = values[last, 2]
        self.fed[slots] = np.maximum(0, self.written[q][slots] - 1)
        self.steps[q][slots] += 1

    def answers(self, map_index: int, documents) -> list[np.ndarray]:
        """Return each document's answer tokens for one map."""
        lengths = self.lengths[map_index]
        return [self.tokens[map_index][document, :max(0, lengths[document])]
                for document in documents]


def map_decoder(state, limits, num_documents: int, expected_tokens=None) -> MapDecoder:
    """Return a decoder for a pipeline's maps on the loaded model."""
    from quail.backends.quail.executor.extract import load_vocabulary
    from quail.backends.quail.executor.model_loading import full_output_head

    return MapDecoder(
        state.torch, full_output_head(state.loaded_model.model), num_documents,
        limits, load_vocabulary(state).stop_ids,
        state.loaded_model.arena.page_tokens, expected_tokens=expected_tokens)
