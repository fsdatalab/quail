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


def trie_paths(label_ids) -> list[tuple[int, ...]]:
    """Return the label trie's deepest proper prefixes, shortest first.

    A chain read row by row after one of them returns the log
    probabilities after every shorter prefix too, so chains over these
    prefixes together read every trie node. With one-token labels only,
    the one path is the empty prefix: the answer cue's own row.
    """
    nodes = set(label_trie(label_ids))
    deepest = [node for node in nodes
               if not any(node + (token,) in nodes
                          for token in label_trie(label_ids)[node])]
    return sorted(deepest, key=lambda prefix: (len(prefix), prefix))


def label_path_scores(label_ids, paths, targets, logprobs) -> np.ndarray:
    """Return every label's summed log probability from chains over trie paths.

    Args:
        label_ids: One token id sequence per label.
        paths: The proper prefix each chain holds after the answer cue;
            every proper prefix of every label starts one of them.
        targets: The token ids, one per column of ``logprobs``.
        logprobs: Shape (paths, rows, targets). Row r of chain p holds
            the log probabilities read after the answer cue and the
            first r tokens of ``paths[p]``.
    """
    where = {}      # proper prefix -> (chain, row) that read after it
    for chain, path in enumerate(paths):
        for depth in range(len(path) + 1):
            where.setdefault(tuple(path[:depth]), (chain, depth))
    column = {token: index for index, token in enumerate(targets)}
    scores = np.zeros(len(label_ids), dtype=np.float64)
    for label, ids in enumerate(label_ids):
        for depth, token in enumerate(ids):
            chain, row = where[tuple(ids[:depth])]
            scores[label] += logprobs[chain, row, column[token]]
    return scores


def label_chain_scores(label_ids, targets, logprobs) -> np.ndarray:
    """Return every label's summed log probability from one chain per label.

    Args:
        label_ids: One token id sequence per label.
        targets: The token ids, one per column of ``logprobs``.
        logprobs: Shape (labels, rows, targets). Row r of chain i holds
            the log probabilities read after the answer cue and the
            first r tokens of label i, so it scores the label's token r.
    """
    column = {token: index for index, token in enumerate(targets)}
    scores = np.zeros(len(label_ids), dtype=np.float64)
    for label, ids in enumerate(label_ids):
        scores[label] = sum(logprobs[label, row, column[token]]
                            for row, token in enumerate(ids))
    return scores


def best_label(scores) -> int:
    """Return the index of the highest score; a tie goes to the earlier label."""
    return int(np.argmax(scores))
