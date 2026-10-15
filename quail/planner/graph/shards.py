"""Split a table's documents across GPU workers by token count."""

import numpy as np


def balanced_shards(doc_tokens, workers: int):
    """Greedily partition documents into shards balanced by token count."""
    lengths = np.asarray(getattr(doc_tokens, "arrow_array", doc_tokens), dtype=np.int64)
    if workers == 1:
        return (tuple(range(len(lengths))),), [int(lengths.sum())]
    order = np.argsort(-lengths, kind="stable")
    loads = [0] * workers
    shards = [[] for _ in range(workers)]
    for i in order:
        w = loads.index(min(loads))
        shards[w].append(int(i))
        loads[w] += int(lengths[i])
    return tuple(tuple(sorted(s)) for s in shards), loads


def contiguous_shards(doc_tokens, workers: int, *, total=None):
    """Split ordered documents into compact token balanced ranges."""
    n_docs = len(doc_tokens)
    if workers == 1 and total is not None:
        return ((0, n_docs),), [total]
    lengths = np.asarray(getattr(doc_tokens, "arrow_array", doc_tokens), dtype=np.int64)
    cumulative = np.concatenate(([0], np.cumsum(lengths)))
    total = int(cumulative[-1])
    zeros = np.flatnonzero(lengths == 0)
    ranges = []
    loads = []
    start = 0
    consumed = 0
    for worker in range(workers):
        if worker == workers - 1:
            stop = n_docs
            load = total - consumed
        else:
            remaining_workers = workers - worker
            target = (total - consumed) / remaining_workers
            stop = min(n_docs, max(start + 1, int(np.searchsorted(
                cumulative, consumed + target))))
            before = int(cumulative[max(start, stop - 1)]) - consumed
            after = int(cumulative[stop]) - consumed
            if before > 0 and abs(target - before) <= abs(target - after):
                stop -= 1
            # The greedy rule stops before a zero-length document once load is positive.
            positive = np.searchsorted(cumulative, consumed, side="right")
            zero = np.searchsorted(zeros, positive)
            if zero < len(zeros):
                stop = min(stop, int(zeros[zero]))
            load = int(cumulative[stop]) - consumed
        ranges.append((start, stop))
        loads.append(load)
        start = stop
        consumed += load
    return tuple(ranges), loads
