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

