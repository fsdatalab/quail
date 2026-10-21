"""Split a table's documents across GPU workers by token count."""

import numpy as np


def balanced_shards(document_tokens, gpus: int):
    """Greedily partition documents into shards balanced by token count."""
    lengths = np.asarray(document_tokens, dtype=np.int64)
    if gpus == 1:
        return (tuple(range(len(lengths))),), [int(lengths.sum())]
    order = np.argsort(-lengths, kind="stable")
    loads = [0] * gpus
    shards = [[] for _ in range(gpus)]
    for i in order:
        w = loads.index(min(loads))
        shards[w].append(int(i))
        loads[w] += int(lengths[i])
    return tuple(tuple(sorted(s)) for s in shards), loads


def contiguous_shards(document_tokens, gpus: int, *, total=None):
    """Split ordered documents into compact token balanced ranges."""
    num_documents = len(document_tokens)
    if gpus == 1 and total is not None:
        return ((0, num_documents),), [total]
    lengths = np.asarray(document_tokens, dtype=np.int64)
    cumulative = np.concatenate(([0], np.cumsum(lengths)))
    total = int(cumulative[-1])
    zeros = np.flatnonzero(lengths == 0)
    ranges = []
    loads = []
    start = 0
    consumed = 0
    for worker in range(gpus):
        if worker == gpus - 1:
            stop = num_documents
            load = total - consumed
        else:
            remaining_workers = gpus - worker
            target = (total - consumed) / remaining_workers
            stop = min(num_documents, max(start + 1, int(np.searchsorted(
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
