"""Group a stage list's readouts so each gets one submission per chunk."""

import numpy as np

from quail.backends.quail.executor.readout import RoundFeed


class ReadoutGroups:
    """Stages grouped by readout, each group submitted once per chunk.

    Attributes:
        shared_readout: Whether every stage reads through one readout.
        group_key: Per stage, the index of its group.
        readout_of: Per group, its readout.
        read_all_of: Per group, whether its stages read every suffix row.
    """

    def __init__(self, torch, stages):
        self.torch = torch
        # Combine stages sharing a readout in one submission per chunk.
        self.shared_readout = all(stage.readout is stages[0].readout
                                  for stage in stages)
        groups = []
        self.group_key = []
        self.readout_of = []
        self.read_all_of = []
        for stage in stages:
            key = (id(stage.readout),
                   False if self.shared_readout else stage.read_all_rows)
            if key not in groups:
                groups.append(key)
                self.readout_of.append(stage.readout)
                self.read_all_of.append(stage.read_all_rows)
            self.group_key.append(groups.index(key))

    def __len__(self) -> int:
        return len(self.readout_of)

    def results(self, handles) -> list:
        """Wait on each group's handle; returns the answer values by group."""
        values = [None] * len(self.readout_of)
        for key, handle in handles:
            values[key] = self.readout_of[key].result(handle)
        return values

    def feed(self, handles, following, entries, ahead):
        """Return the chunk's RoundFeed, indexed by the chunk's answers.

        Args:
            handles: The chunk's submit() handles.
            following: Per answer row, the draft token fed after it, or -1.
            entries: Per answer, its stage and rows.
            ahead: Per stage, whether rounds are fed from its answers.
        """
        counts = np.fromiter((rows for _, rows in entries), dtype=np.int64,
                             count=len(entries))
        if self.shared_readout:
            return self.readout_of[0].feed(handles[0][1], following, counts)
        # each readout read only its own stages' rows: a group with a
        # stage rounds are fed from computes its answers' feed, which is
        # placed at those answers' places in the chunk
        stage = np.fromiter((j for j, _ in entries), dtype=np.int64,
                            count=len(entries))
        group = np.asarray(self.group_key, dtype=np.int64)[stage]
        fed = np.asarray(ahead, dtype=bool)[stage]
        accepted = chosen = None
        for key, handle in handles:
            mine = group == key
            if not fed[mine].any():
                continue
            rows = np.repeat(mine, counts)
            part = self.readout_of[key].feed(handle, following[rows], counts[mine])
            if accepted is None:
                accepted = part.accepted.new_zeros(len(entries))
                chosen = part.chosen.new_zeros(len(entries))
            index = part.accepted.new_tensor(np.flatnonzero(mine))
            accepted.index_copy_(0, index, part.accepted)
            chosen.index_copy_(0, index, part.chosen)
        if accepted is None:
            return None
        return RoundFeed(accepted=accepted, chosen=chosen)

    def _select_rows(self, normed, spans_j):
        index = np.concatenate([np.arange(start, end, dtype=np.int64)
                                for start, end in spans_j])
        if hasattr(normed, "index_select"):
            return normed.index_select(
                0, self.torch.from_numpy(index).to(normed.device, non_blocking=True))
        return [normed[r] for r in index]

    def submit(self, normed, entries, rows_per_answer):
        """Submit answer rows to readouts and return handles by readout group.

        With one readout for every stage the whole chunk goes in one
        call, in group zero; otherwise each readout's stages' rows go to
        it in one call, selected with one index.
        """
        if self.shared_readout:
            readout = self.readout_of[0]
            if rows_per_answer is None:
                return [(0, readout.submit(normed))]
            return [(0, readout.submit(normed, rows_per_answer=rows_per_answer))]
        spans = [[] for _ in self.readout_of]
        counts = [[] for _ in self.readout_of]
        boundaries = (None if rows_per_answer is None else np.cumsum(rows_per_answer))
        row = answer = 0
        for j, rows in entries:
            key = self.group_key[j]
            spans[key].append((row, row + rows))
            row += rows
            if boundaries is not None:
                end = int(np.searchsorted(boundaries, row)) + 1
                if self.read_all_of[key]:
                    counts[key].append(rows_per_answer[answer:end])
                answer = end
        handles = []
        for key, spans_key in enumerate(spans):
            if not spans_key:
                continue
            readout = self.readout_of[key]
            selected = self._select_rows(normed, spans_key)
            if not counts[key]:
                handles.append((key, readout.submit(selected)))
            else:
                lengths = counts[key]
                lengths = lengths[0] if len(lengths) == 1 else np.concatenate(lengths)
                handles.append((key, readout.submit(selected, rows_per_answer=lengths)))
        return handles
