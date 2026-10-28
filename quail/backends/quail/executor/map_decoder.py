"""Greedy decoding of AI.MAP answers, one row per running answer per pass.

A document's map prompt runs as a stage of its pipeline. The stage's
last row chooses the answer's first token; from the next pass on, the
document has one decode row per pass until its answer ends. Each pass:

* the host lays out every running answer's row with array operations:
  its position, the arena row its KV is written to, and its pages;
* the GPU copies each answer's last chosen token into the row's input,
  so a pass launches before the host has read the pass before it;
* after the forward pass, the GPU takes the argmax of each answer row
  over the whole vocabulary, writes it to the token buffer, and copies
  the tokens to pinned host memory without blocking;
* the host reads the copies that have landed, appends the tokens to
  the answers, and ends the answers that wrote a stop token or reached
  their limit.

The host learns of an answer's end a pass or two after the GPU writes
it, so an ended answer may feed one or two more rows. Their tokens are
dropped, and their KV lands in rows the answer's reservation covers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from quail.backends.quail.executor.chunk import DecodeRows

# copies of chosen tokens kept in flight before the host waits on the
# oldest; two lets the host read a pass's tokens while the next runs
IN_FLIGHT = 2


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
    """

    def __init__(self, torch, head, num_documents: int, limits, stop_ids,
                 page_tokens: int):
        self.torch = torch
        self.head = head
        self.limits = np.asarray(limits, dtype=np.int64)
        self.page_tokens = page_tokens
        n = num_documents
        self.stop_ids = np.asarray(sorted(stop_ids), dtype=np.int64)
        # per map, each document's tokens and its answer's length, -1
        # while the answer has not ended
        self.tokens = [np.zeros((n, int(limit)), dtype=np.int32)
                       for limit in self.limits]
        self.lengths = [np.full(n, -1, dtype=np.int64) for _ in self.limits]
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
        self.next_token = torch.zeros(n, dtype=torch.int64, device=self.device)
        self.copies = deque()
        self.finished = []      # (document, map) whose answers ended
        self.rows_fed = np.zeros(len(self.limits), dtype=np.int64)  # per map
        # KV tokens the decode rows read: in all, alone, through their
        # own pages beside a shared read, and through shared reads
        self.read_tokens = dict(all=0, alone=0, own=0, shared=0)
        # per slot, the map its answers are for, or None when each
        # stage names its map
        self.slot_maps = None
        # whether a document's branches read its shared pages together
        self.share = False

    # ---- documents entering and leaving -----------------------------------

    def start(self, document: int, map_index: int, position: int, pages,
              parent: int | None = None, shared_tokens: int = 0) -> None:
        """Begin an answer whose first token the current pass chooses.

        Args:
            document: The document's index in the run.
            map_index: The map the answer is for.
            position: The position of the answer's first decode row,
                right after its prompt.
            pages: The document's page ids in logical order; they must
                cover the prompt and the answer's limit.
            parent: The slot whose whole pages this one borrows, or None.
            shared_tokens: The tokens it borrows from the parent.
        """
        if self.map[document] >= 0:
            raise ValueError(f"document {document} is already writing an answer")
        if self.slot_maps is not None:
            map_index = int(self.slot_maps[document])
        pages = np.asarray(pages, dtype=np.int32)
        if len(pages) > self.pages.shape[1]:
            wider = np.zeros((len(self.map), len(pages)), dtype=np.int32)
            wider[:, :self.pages.shape[1]] = self.pages
            self.pages = wider
        self.pages[document, :len(pages)] = pages
        self.num_pages[document] = len(pages)
        self.map[document] = map_index
        self.origin[document] = position
        self.fed[document] = 0
        self.stopped[document] = False
        self.parent[document] = document
        self.borrowed[document] = 0
        self.lead[document] = 0
        if parent is not None and shared_tokens:
            self.parent[document] = parent
            self.borrowed[document] = shared_tokens
            self.lead[parent] = max(self.lead[parent], shared_tokens)

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
        """Lay out this pass's decode rows, one per answer that may continue.

        An answer continues while no stop token has been read and its
        tokens, counting the one this row writes, stay within its limit.
        """
        running = np.flatnonzero(self.map >= 0)
        limits = self.limits[self.map[running]]
        running = running[~self.stopped[running]
                          & (self.fed[running] + 1 < limits)]
        if not len(running):
            return None
        positions = self.origin[running] + self.fed[running]
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
        columns = skip[:, None] + np.arange(width)
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
        chunk.input_ids[start:].copy_(self.next_token.index_select(0, slots))
        self.fed[rows.slots] += 1
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
        documents = np.concatenate([
            np.asarray(first_documents, dtype=np.int64),
            np.empty(0, dtype=np.int64) if rows is None else rows.slots])
        # a first row writes token 0; a decode row fed f - 1 rows writes token f
        indices = np.concatenate([
            np.zeros(len(first_rows), dtype=np.int64),
            np.empty(0, dtype=np.int64) if rows is None else self.fed[rows.slots]])
        selected = []
        if len(first_rows):
            index = torch.from_numpy(np.asarray(first_rows, dtype=np.int64))
            selected.append(normed.index_select(0, index.to(normed.device,
                                                             non_blocking=True)))
        if count:
            selected.append(normed[normed.shape[0] - count:])
        hidden = selected[0] if len(selected) == 1 else torch.cat(selected)
        chosen = torch.argmax(
            torch.nn.functional.linear(hidden.to(self.head.dtype), self.head), dim=1)
        slots = torch.from_numpy(documents).to(self.device, non_blocking=True)
        self.next_token.index_copy_(0, slots, chosen)
        host = torch.empty(len(documents), dtype=torch.int64,
                           pin_memory=self.head.is_cuda)
        host.copy_(chosen, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.copies.append(_Copy(event, host, documents, indices,
                                 self.map[documents].copy()))
        while len(self.copies) > IN_FLIGHT:
            self._read(self.copies.popleft())

    def harvest(self, wait: bool = False) -> None:
        """Read every copy that has landed, or every copy when wait is set."""
        while self.copies and (wait or self.copies[0].event.query()):
            self._read(self.copies.popleft())

    def _read(self, copy: _Copy) -> None:
        copy.event.synchronize()
        tokens = copy.host.numpy()
        documents, indices, maps = copy.documents, copy.indices, copy.maps
        # rows of an answer that already ended carry nothing new, even
        # when the document has started its next map's answer since
        current = (self.map[documents] == maps) & ~self.stopped[documents]
        documents, indices, tokens, maps = (
            documents[current], indices[current], tokens[current], maps[current])
        for q in np.unique(maps).tolist():
            mine = maps == q
            self.tokens[q][documents[mine], indices[mine]] = tokens[mine]
        stop = np.isin(tokens, self.stop_ids)
        self.stopped[documents[stop]] = True
        ended = stop | (indices + 1 >= self.limits[maps])
        for document, index, q, at_stop in zip(
                documents[ended].tolist(), indices[ended].tolist(),
                maps[ended].tolist(), stop[ended].tolist()):
            self.lengths[q][document] = index if at_stop else index + 1
            self.map[document] = -1
            self.finished.append((document, q))

    def answers(self, map_index: int, documents) -> list[np.ndarray]:
        """Return each document's answer tokens for one map."""
        lengths = self.lengths[map_index]
        return [self.tokens[map_index][document, :max(0, lengths[document])]
                for document in documents]


def map_decoder(state, limits, num_documents: int) -> MapDecoder:
    """Return a decoder for a pipeline's maps on the loaded model."""
    from quail.backends.quail.executor.extract import load_vocabulary
    from quail.backends.quail.executor.model_loading import full_output_head

    return MapDecoder(
        state.torch, full_output_head(state.loaded_model.model), num_documents,
        limits, load_vocabulary(state).stop_ids,
        state.loaded_model.arena.page_tokens)
