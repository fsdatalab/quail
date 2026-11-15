"""Each map's answer survival curves, counted as its answers are written."""

import numpy as np

from quail.common.answers import prior_survival, survival_rows


def input_bucket(tokens):
    """Return the input-length bucket: one per doubling of the prefix length."""
    return np.floor(np.log2(np.maximum(np.asarray(tokens, dtype=float), 1.0))
                    ).astype(np.int64)


class AnswerCurves:
    """Kaplan-Meier curves of a decoder's answers, per map and per bucket.

    Maps and extraction fields share the curves: headroom admission reads
    a map's bucket curves, and the draft policy and the eviction order
    read its whole curve. An update recounts only the answers that moved.

    Args:
        decoder: The MapDecoder whose answers the curves count.
        prefix_lengths: Each slot's prompt length, which sets its
            input-length bucket; None puts every slot in one bucket.
        bucket_answers: The ended answers a bucket needs before its
            curve replaces its map's.

    Attributes:
        whole: Per map, S(t) over all its answers, 0 from its limit on.
        buckets: Per map and bucket, S(t); row map * num_buckets + bucket.
        slot_map: Per slot, its map, or -1 for a slot with no answer.
        slot_row: Per slot, its row of buckets, or -1.
        slot_bucket: Per slot, its input-length bucket.
        rows: The rows of buckets that have answers.
    """

    def __init__(self, decoder, prefix_lengths=None, bucket_answers=30):
        self.decoder = decoder
        num_slots = len(decoder.map)
        if prefix_lengths is None:
            prefix_lengths = np.ones(num_slots, dtype=np.int64)
        self.slot_bucket = input_bucket(prefix_lengths)
        self.num_buckets = int(self.slot_bucket.max()) + 1 if num_slots else 1
        self.bucket_answers = bucket_answers
        self.width = int(decoder.limits.max()) + 1
        self.slot_map = np.full(num_slots, -1, dtype=np.int64)
        self.slot_row = np.full(num_slots, -1, dtype=np.int64)
        self.slots = None
        self.buckets = None
        self.whole = None
        # per slot, the tokens it was last counted at and whether it had
        # ended; per map and per curve row, those counts by token
        self._observed = np.zeros(num_slots, dtype=np.int64)
        self._ended = np.zeros(num_slots, dtype=bool)
        self._watched = np.zeros(0, dtype=np.int64)

    def _bind(self):
        decoder = self.decoder
        limits = decoder.limits
        # per map, the planner's geometric prior, 0 past the map's limit
        self.prior = np.zeros((len(limits), self.width))
        for q, limit in enumerate(limits):
            self.prior[q, :limit + 1] = prior_survival(
                decoder.expected_tokens[q], int(limit))
        self.slots = np.concatenate(
            [np.asarray(slots, dtype=np.int64)
             for slots in decoder.answer_slots.values()])
        self.slot_map[self.slots] = np.concatenate(
            [np.full(len(slots), q) for q, slots in decoder.answer_slots.items()])
        maps = self.slot_map[self.slots]
        self.slot_row[self.slots] = (maps * self.num_buckets
                                     + self.slot_bucket[self.slots])
        self.rows = np.unique(self.slot_row[self.slots])
        self.row_limit = np.repeat(limits, self.num_buckets)
        num_maps, num_rows = len(limits), len(self.row_limit)
        self.buckets = np.zeros((num_rows, self.width))
        self.whole = np.zeros((num_maps, self.width))
        self.map_counts = np.zeros((num_maps, self.width), dtype=np.int64)
        self.map_events = np.zeros((num_maps, self.width), dtype=np.int64)
        self.row_counts = np.zeros((num_rows, self.width), dtype=np.int64)
        self.row_events = np.zeros((num_rows, self.width), dtype=np.int64)
        self.num_ended = np.zeros(num_rows, dtype=np.int64)
        decoder.ended_log = []
        slots = self.slots
        self._count(slots, *self._observation(slots), sign=1)

    @property
    def bucket_curves(self):
        """The curve of each (map, bucket) with answers, cut at its map's limit."""
        if self.buckets is None:
            return {}
        return {(int(row // self.num_buckets), int(row % self.num_buckets)):
                self.buckets[row, :self.row_limit[row] + 1] for row in self.rows}

    def _observation(self, slots):
        """Each slot's last token count and whether its answer ended."""
        decoder = self.decoder
        maps = self.slot_map[slots]
        lengths = decoder.lengths[maps, slots]
        ended = lengths >= 0
        return np.where(ended, lengths, decoder.written[maps, slots]), ended

    def _count(self, slots, observed, ended, sign):
        """Add (sign 1) or remove (sign -1) slots' observations from the counts."""
        maps, rows = self.slot_map[slots], self.slot_row[slots]
        np.add.at(self.map_counts, (maps, observed), sign)
        np.add.at(self.map_events, (maps[ended], observed[ended]), sign)
        np.add.at(self.row_counts, (rows, observed), sign)
        np.add.at(self.row_events, (rows[ended], observed[ended]), sign)
        np.add.at(self.num_ended, rows[ended], sign)
        if sign > 0:
            self._observed[slots] = observed
            self._ended[slots] = ended

    def update(self, writing=None):
        """Rebuild the curves of maps whose answers moved since the last update.

        Only answers written now, at the last update, or ended since
        can have moved.

        Args:
            writing: Slots whose answers are being written, or None to find them.
        """
        decoder = self.decoder
        first = self.slots is None
        if first:
            self._bind()
        if writing is None:
            writing = np.flatnonzero(decoder.map >= 0)
        watch = np.unique(np.concatenate([writing, self._watched, *decoder.ended_log]))
        decoder.ended_log.clear()
        self._watched = writing
        watch = watch[self.slot_map[watch] >= 0]
        observed, ended = self._observation(watch)
        moved = (observed != self._observed[watch]) | (ended != self._ended[watch])
        changed = np.zeros(len(decoder.limits), dtype=bool)
        if moved.any():
            slots = watch[moved]
            self._count(slots, self._observed[slots], self._ended[slots], sign=-1)
            self._count(slots, observed[moved], ended[moved], sign=1)
            changed[self.slot_map[slots]] = True
        if first:
            changed[:] = True
        if changed.any():
            self._rebuild(changed)

    def _rebuild(self, changed):
        """Rebuild the whole-map and bucket curves of the changed maps."""
        mine = np.flatnonzero(changed)
        self.whole[mine] = survival_rows(self.map_counts[mine], self.map_events[mine],
                                         self.prior[mine],
                                         self.decoder.limits[mine])
        rows = self.rows[changed[self.rows // self.num_buckets]]
        self.buckets[rows] = self.whole[rows // self.num_buckets]
        # a bucket with few ended answers keeps its map's curve
        rows = rows[self.num_ended[rows] >= self.bucket_answers]
        if len(rows):
            self.buckets[rows] = survival_rows(self.row_counts[rows],
                                              self.row_events[rows],
                                              self.whole[rows // self.num_buckets],
                                              self.row_limit[rows])
