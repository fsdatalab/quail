"""Score label tokens and match decoded text to labels.

The letters rule reads one token per label. The trie_tree rule scores
each label through the end of the word that tells it apart, and gives
a label that starts another the answers whose longest matching label
it is. The trie_decode rule (GreedyDecoder) picks one allowed token at
each node where the labels split and never compares complete label
scores.
"""

import math

import numpy as np

from quail.labels import decode_runs, fed_nodes, label_trie


def trie_targets(trie) -> list[int]:
    """Return every token id any trie node reads, sorted."""
    return sorted({token for tokens in trie.values() for token in tokens})


def match_label(text: str, labels) -> str | None:
    """Match the first answer line to a label.

    A leading "thought" line and "ANSWER:" prefix are removed before matching.
    Matching ignores case and accepts text that starts with a label. When
    several labels match, the longest wins, then the first in the list.

    Args:
        text: Decoded model answer.
        labels: Label strings in query order.

    Returns:
        The matching label, or None if no label matches.
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
    """Return each label letter's log probability at the answer position.

    Args:
        label_ids: One single-token sequence per label.
        targets: Token IDs in readout-column order.
        logprobs: One value per target token.

    Returns:
        Scores in label order, on the same scale as logprobs.
    """
    column = {token: index for index, token in enumerate(targets)}
    return np.asarray([float(logprobs[column[ids[0]]]) for ids in label_ids],
                      dtype=np.float64)


def best_label(scores) -> int:
    """Return the first index with the highest score."""
    return int(np.argmax(scores))


class GreedyDecoder:
    """Per-document state for choosing one allowed label token per read.

    Each read selects the highest-scoring child of a choice node and
    feeds the run to the next choice node. Decoding finishes when a
    choice leads to a label.

    Args:
        label_ids: Token sequence for each label. No label's sequence may
            be a proper prefix of another label's sequence.
        targets: Token IDs corresponding to the readout columns.
        documents: Number of documents to track.
        cue: The answer cue token.

    Raises:
        ValueError: One label's token sequence is a proper prefix of another.
    """

    def __init__(self, label_ids, targets, documents, cue):
        self.label_ids = [tuple(ids) for ids in label_ids]
        self.trie = label_trie(label_ids)
        self.leaf = {}
        for label, ids in enumerate(self.label_ids):
            self.leaf.setdefault(ids, label)
        if any(node in self.leaf for node in self.trie):
            raise ValueError("a label is a proper prefix of another label")
        self.runs = decode_runs(label_ids)
        self.suffixes = [[cue, *self.runs.runs[0]], *self.runs.runs[1:]]
        self.rounds = self.runs.rounds
        # a tie between children goes to the earlier label's token
        self.order = {}
        for label, ids in enumerate(self.label_ids):
            for token in ids:
                self.order.setdefault(token, label)
        self.column = {token: i for i, token in enumerate(targets)}
        self.node = [self.runs.first] * documents
        self.pending = [0] * documents
        self.label = np.full(documents, -1, dtype=np.int64)
        self.tokens = 0     # tokens requested so far

    def requests(self, doc):
        """Return the request that feeds the document's next run.

        The first request feeds the answer cue and the forced tokens
        to the first choice node. Each later request feeds the run the
        last choice picked; the earlier path is already in KV.

        Args:
            doc: Document index.

        Returns:
            A one-element list holding the run's suffix index, or None
            if the document already has a label.
        """
        if self.label[doc] >= 0:
            return None
        index = self.pending[doc]
        self.tokens += len(self.suffixes[index])
        return [index]

    def update(self, doc, row):
        """Advance a document to its highest-scoring allowed token.

        Args:
            doc: Document index.
            row: Token log probabilities in target-column order, read at
                the document's current choice node.
        """
        node = self.node[doc]
        if node in self.leaf:
            self.label[doc] = self.leaf[node]
            return
        best = min(self.trie[node],
                   key=lambda token: (-float(row[self.column[token]]),
                                      self.order[token]))
        target, run = self.runs.after[node, best]
        if run is None:
            self.label[doc] = self.leaf[target]
        else:
            self.node[doc] = target
            self.pending[doc] = run


def trie_chains(label_ids, read=None) -> list:
    """Split the fed trie nodes into chains that together hold each once.

    The fed nodes are the read nodes and their ancestors. The first
    chain starts at the root, the answer cue's row, and follows each
    node's first fed child; every other fed child starts a new chain.
    A chain is one causal segment of the forward pass, so a row sees
    the chain's earlier rows; the rows above the chain's first node
    lie in earlier chains and are gathered separately.

    Args:
        label_ids: One token id sequence per label.
        read: The nodes whose rows are read, from ``read_nodes``.
            None reads every node with children.

    Returns:
        One (nodes, start, gathers) per chain: the trie nodes whose
        rows the chain holds, in order; the depth of its first node,
        which is the position offset past the answer cue; and the
        ancestor rows outside it as (chain, leading rows) pairs.
    """
    trie = label_trie(label_ids)
    if read is None:
        read = set(trie)
    fed = fed_nodes(read)
    chains = []
    placed = {}       # node -> (chain, row)

    def extend(chain, node):
        placed[node] = (chain, len(chains[chain][0]))
        chains[chain][0].append(node)
        children = [node + (token,) for token in trie[node]
                    if node + (token,) in fed]
        for index, child in enumerate(children):
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


def chain_reads(chains, read=None) -> list[bool]:
    """Return which chain rows, in chain order, are read.

    Args:
        chains: The chains ``trie_chains`` built.
        read: The read nodes, from ``read_nodes``; None reads every row.
    """
    return [read is None or node in read
            for nodes, _, _ in chains for node in nodes]


def tree_scores(label_ids, chains, targets, logprobs, read=None) -> np.ndarray:
    """Return every label's log probability from packed trie chains.

    A label's probability is the product of its tokens' probabilities
    at the read nodes on its path. A label that starts another gets
    that minus the probability of each label that extends it with no
    label in between, as ``match_label`` gives decoded text to its
    longest matching label.

    Args:
        label_ids: One token id sequence per label.
        chains: The chains ``trie_chains`` built.
        targets: The token ids, one per column of ``logprobs``.
        logprobs: Shape (read rows, targets): the chains' read rows
            back to back, each read after the trie node it holds.
        read: The read nodes, from ``read_nodes``; None reads every row.

    Returns:
        One log probability per label, in label order. A label whose
        extensions hold all of its probability scores minus infinity.
    """
    row_of = {}
    for node, is_read in zip((node for nodes, _, _ in chains for node in nodes),
                             chain_reads(chains, read)):
        if is_read:
            row_of[node] = len(row_of)
    column = {token: index for index, token in enumerate(targets)}

    def logprob(ids, start=0):
        return sum(float(logprobs[row_of[ids[:depth]], column[ids[depth]]])
                   for depth in range(start, len(ids)) if ids[:depth] in row_of)

    paths = [tuple(ids) for ids in label_ids]
    taken = dict.fromkeys(paths, 0.0)   # share held by the nearest extensions
    for ids in taken:
        for depth in range(len(ids) - 1, 0, -1):
            if ids[:depth] in taken:
                taken[ids[:depth]] += math.exp(logprob(ids, depth))
                break
    scores = np.empty(len(paths), dtype=np.float64)
    for label, ids in enumerate(paths):
        # rounding can push the extensions' share to 1 or past it
        scores[label] = (logprob(ids) + math.log1p(-taken[ids])
                         if taken[ids] < 1.0 else -math.inf)
    return scores
