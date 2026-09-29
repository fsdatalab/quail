"""AI.CLASSIFY label scoring from per-position log probabilities.

Under the letters rule every label is one token and its score is that
token's log probability at the answer's row. Under the trie rule a
label's score is the sum of its tokens' log probabilities after the
prompt, each read at the position before the token. Labels that share
leading tokens share those positions, so the positions to read are the
proper prefixes of the labels' token sequences: a trie whose internal
nodes are read once each; a greedy decode over the trie follows the
likeliest token one round at a time (GreedyDecoder). A baseline's
decoded answer is matched to a label by its text instead
(match_label).
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


def match_label(text: str, labels) -> str | None:
    """Return the label a decoded answer names, or None.

    A leading "thought" line and then a leading "ANSWER:" are skipped:
    DiffusionGemma may open an empty thinking channel, which reads
    "thought" once its special tokens are removed, or repeat the answer
    cue. The rest's first line, trimmed, must start with a label,
    ignoring case; the longest such label wins, then the earlier one.
    Anything else names no label.
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
    """Return every one-token label's log probability at the answer's row.

    Args:
        label_ids: One token id per label, each a one-element sequence.
        targets: The token ids, one per column of ``logprobs``.
        logprobs: Shape (targets,): the log probabilities read at the
            answer's row.
    """
    column = {token: index for index, token in enumerate(targets)}
    return np.asarray([float(logprobs[column[ids[0]]]) for ids in label_ids],
                      dtype=np.float64)


def best_label(scores) -> int:
    """Return the index of the highest score; a tie goes to the earlier label."""
    return int(np.argmax(scores))


class GreedyDecoder:
    """Labels decoded one token per round along the label trie.

    Each round reads, for a document, the log probabilities after the
    trie node it has decoded so far, and appends the likeliest token
    among the node's children; the document is resolved once its node
    is a whole label. Nodes are indexed in ``nodes``, shortest first.
    The decoded label is the greedy path, not the label with the
    highest summed log probability.

    Args:
        label_ids: One token id sequence per label; no label may be a
            proper prefix of another, since a decoded label ends only
            at a leaf.
        targets: The token ids, one per column of the log probabilities.
        documents: How many documents the decoder tracks.

    Raises:
        ValueError: A label is a proper prefix of another.
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
        """The index into nodes the document reads this round; None once resolved."""
        if self.label[doc] >= 0:
            return None
        node = self.node[doc]
        self.tokens += len(node) + 1
        return [self.index[node]]

    def update(self, doc, row):
        """Append the likeliest child of the document's node.

        Args:
            doc: The document.
            row: The log probabilities read after the node, one per
                target.
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
