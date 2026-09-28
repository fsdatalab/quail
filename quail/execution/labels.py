"""AI.CLASSIFY label scoring from per-position log probabilities.

A label's score is the sum of its tokens' log probabilities after the
prompt, each read at the position before the token. Labels that share
leading tokens share those positions, so the positions to read are the
proper prefixes of the labels' token sequences: a trie whose internal
nodes are read once each.
"""

import numpy as np


def label_trie(label_ids) -> dict[tuple[int, ...], list[int]]:
    """Return each proper label prefix and the tokens that can follow it.

    Args:
        label_ids: One token id sequence per label.

    Raises:
        ValueError: A label has no tokens.
    """
    children = {}
    for ids in label_ids:
        if not ids:
            raise ValueError("a label has no tokens")
        for depth in range(len(ids)):
            children.setdefault(tuple(ids[:depth]), set()).add(ids[depth])
    return {prefix: sorted(tokens) for prefix, tokens in children.items()}


def trie_targets(trie) -> list[int]:
    """Return every token id any trie node reads, sorted."""
    return sorted({token for tokens in trie.values() for token in tokens})


def label_scores(label_ids, prefixes, targets, logprobs) -> np.ndarray:
    """Return every label's summed log probability.

    Args:
        label_ids: One token id sequence per label.
        prefixes: The trie prefixes, one per row of ``logprobs``.
        targets: The token ids, one per column of ``logprobs``.
        logprobs: Log probabilities of each target token read after
            each prefix, shape (len(prefixes), len(targets)).
    """
    row = {prefix: index for index, prefix in enumerate(prefixes)}
    column = {token: index for index, token in enumerate(targets)}
    scores = np.zeros(len(label_ids), dtype=np.float64)
    for label, ids in enumerate(label_ids):
        for depth, token in enumerate(ids):
            scores[label] += logprobs[row[tuple(ids[:depth])], column[token]]
    return scores


def best_label(scores) -> int:
    """Return the index of the highest score; a tie goes to the earlier label."""
    return int(np.argmax(scores))
