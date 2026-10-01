"""Score category tokens and match decoded text to categories.

Letter scoring reads one token per category. Tree scoring sums token
log probabilities for each complete category. GreedyDecoder selects
one allowed token per round without comparing complete label scores.
"""

import numpy as np

from quail.labels import label_trie


def trie_targets(trie) -> list[int]:
    """Return every token id any trie node reads, sorted."""
    return sorted({token for tokens in trie.values() for token in tokens})


def match_label(text: str, labels) -> str | None:
    """Match the first answer line to a supplied category.

    A leading "thought" line and "ANSWER:" prefix are removed before matching.
    Matching ignores case and accepts text starting with a category. When
    several categories match, the longest wins, then the first in the list.

    Args:
        text: Decoded model answer.
        labels: Category strings in query order.

    Returns:
        The matching category string, or None if no category matches.
    """
    answer = text.strip()
    first, _, rest = answer.partition("\n")
    if first.strip().casefold() == "thought":
        answer = rest.strip()
    if answer.casefold().startswith("answer:"):
        answer = answer[len("answer:"):]
    answer = answer.strip().split("\n", 1)[0].strip().casefold()
    best = None
    for index, label in enumerate(labels):
        candidate = label.casefold()
        if answer.startswith(candidate) and (
                best is None or len(candidate) > len(labels[best].casefold())):
            best = index
    return None if best is None else labels[best]


def letter_scores(label_ids, targets, logprobs) -> np.ndarray:
    """Return each category letter's log probability at the answer position.

    Args:
        label_ids: One single-token sequence per category.
        targets: Token IDs in readout-column order.
        logprobs: One-dimensional array of values for the target tokens.

    Returns:
        Scores in category order, using the same scale as logprobs.
    """
    column = {token: index for index, token in enumerate(targets)}
    return np.asarray([float(logprobs[column[ids[0]]]) for ids in label_ids],
                      dtype=np.float64)


def best_label(scores) -> int:
    """Return the first index with the highest score."""
    return int(np.argmax(scores))


class GreedyDecoder:
    """Per-document state for choosing one allowed label token per round.

    Each round selects the highest-scoring child of the current trie node.
    Decoding finishes at a complete category. Nodes are ordered by depth.

    Args:
        label_ids: Token sequence for each category. No complete sequence may
            be the prefix of another category's sequence.
        targets: Token IDs corresponding to the readout columns.
        documents: Number of documents to track.

    Raises:
        ValueError: One category's token sequence is a prefix of another.
    """

    def __init__(self, label_ids, targets, documents):
        self.label_ids = [tuple(ids) for ids in label_ids]
        self.trie = label_trie(label_ids)
        self.nodes = sorted(self.trie, key=lambda node: (len(node), node))
        self.index = {node: i for i, node in enumerate(self.nodes)}
        self.leaf = {}
        for label, ids in enumerate(self.label_ids):
            self.leaf.setdefault(ids, label)
        if any(node in self.leaf for node in self.trie):
            raise ValueError("a label is a proper prefix of another label")
        # a tie between children goes to the earlier label's token
        self.order = {}
        for label, ids in enumerate(self.label_ids):
            for token in ids:
                self.order.setdefault(token, label)
        self.column = {token: i for i, token in enumerate(targets)}
        self.rounds = max(len(ids) for ids in self.label_ids)
        self.node = [()] * documents
        self.label = np.full(documents, -1, dtype=np.int64)
        self.tokens = 0     # chain tokens requested so far

    def requests(self, doc):
        """Return the current trie-node request and count its tokens.

        Args:
            doc: Document index.

        Returns:
            A one-element list containing the current node index, or None if
            the document already has a label.
        """
        if self.label[doc] >= 0:
            return None
        node = self.node[doc]
        self.tokens += len(node) + 1
        return [self.index[node]]

    def update(self, doc, row):
        """Advance a document to its highest-scoring allowed token.

        Args:
            doc: Document index.
            row: Token log probabilities in target-column order.
        """
        node = self.node[doc]
        best = min(self.trie[node],
                   key=lambda token: (-float(row[self.column[token]]),
                                      self.order[token]))
        child = node + (best,)
        self.node[doc] = child
        if child in self.leaf:
            self.label[doc] = self.leaf[child]


def trie_chains(label_ids) -> list:
    """Split the label trie into chains that each compute every node once.

    The first chain starts at the root, the answer cue's row, and
    follows each node's first child; every other child starts a new
    chain. A chain is one causal segment of the forward pass, so a
    row sees the chain's earlier rows; the rows above the chain's
    first node lie in earlier chains and are gathered separately.

    Returns:
        One (nodes, start, gathers) per chain: the trie nodes whose
        rows the chain holds, in order; the depth of its first node,
        which is the position offset past the answer cue; and the
        ancestor rows outside it as (chain, leading rows) pairs.
    """
    trie = label_trie(label_ids)
    chains = []
    placed = {}       # node -> (chain, row)

    def extend(chain, node):
        placed[node] = (chain, len(chains[chain][0]))
        chains[chain][0].append(node)
        # a full label's row is never read: only nodes with children
        internal = [node + (token,) for token in trie[node]
                    if node + (token,) in trie]
        for index, child in enumerate(internal):
            if index == 0:
                extend(chain, child)
            else:
                chains.append(([], len(child), []))
                extend(len(chains) - 1, child)

    chains.append(([], 0, []))
    extend(0, ())
    for index, (nodes, _, gathers) in enumerate(chains):
        if index == 0:
            continue
        parent = nodes[0][:-1]
        counts = {}
        for depth in range(len(parent) + 1):
            chain, row = placed[tuple(parent[:depth])]
            counts[chain] = max(counts.get(chain, 0), row + 1)
        gathers.extend(sorted(counts.items()))
    return chains


def tree_scores(label_ids, chains, targets, logprobs) -> np.ndarray:
    """Return every label's summed log probability from packed trie chains.

    Args:
        label_ids: One token id sequence per label.
        chains: The chains ``trie_chains`` built.
        targets: The token ids, one per column of ``logprobs``.
        logprobs: Shape (rows, targets): the chains' rows back to back,
            each read after the trie node it holds.
    """
    row_of = {}
    row = 0
    for nodes, _, _ in chains:
        for node in nodes:
            row_of[node] = row
            row += 1
    column = {token: index for index, token in enumerate(targets)}
    scores = np.zeros(len(label_ids), dtype=np.float64)
    for label, ids in enumerate(label_ids):
        for depth, token in enumerate(ids):
            scores[label] += logprobs[row_of[tuple(ids[:depth])], column[token]]
    return scores
