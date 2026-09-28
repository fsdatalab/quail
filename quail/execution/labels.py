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


class RoundScorer:
    """Label scores read in rounds over the trie, pruned by bounds.

    Each round reads, for a document, the log probabilities after some
    trie nodes: a chain of the answer cue and the node's tokens, at
    its last row. A label's score is the sum of its tokens' log
    probabilities, and a token not read yet adds at most 0, so a
    label's partial sum bounds its score from above. A label is
    resolved once every token is read. After each round a label whose
    bound is below a resolved label's score is pruned (a tie goes to
    the earlier label), and a document is resolved once no unresolved
    label is left: the best resolved label is its answer.

    Round r reads every alive node of depth r (``trie_rounds``), or,
    with ``search``, the one node the top-bounded unresolved label
    reads next (``trie_search``), so that label resolves first and
    prunes the rest. Nodes are indexed in ``nodes``, shortest first.

    With ``demand``, only whether the label is one of those labels
    matters: a document is also resolved once every alive label is in
    the demanded set, or none is, and its label is then the alive
    label with the highest bound.

    Args:
        label_ids: One token id sequence per label.
        targets: The token ids, one per column of the log probabilities.
        documents: How many documents the scorer tracks.
        search: Read one node per round, best first.
        demand: Indices of the labels a filter accepts, when the
            label is only tested for membership; None scores fully.
    """

    def __init__(self, label_ids, targets, documents, search=False,
                 demand=None):
        self.label_ids = [tuple(ids) for ids in label_ids]
        self.lengths = np.array([len(ids) for ids in self.label_ids])
        self.nodes = sorted(label_trie(label_ids), key=lambda n: (len(n), n))
        self.index = {node: i for i, node in enumerate(self.nodes)}
        self.search = search
        self.demand = None
        if demand is not None:
            self.demand = np.zeros(len(self.label_ids), dtype=bool)
            self.demand[list(demand)] = True
        # a node is read at most once per document
        self.rounds = len(self.nodes) if search else int(self.lengths.max())
        self.column = {token: i for i, token in enumerate(targets)}
        count = len(self.label_ids)
        self.partial = np.zeros((documents, count))
        self.read = np.zeros((documents, count), dtype=np.int64)
        self.alive = np.ones((documents, count), dtype=bool)
        self.label = np.full(documents, -1, dtype=np.int64)
        self.tokens = 0     # chain tokens requested so far

    def requests(self, doc, round):
        """Indices into nodes the document reads this round; None once resolved."""
        if self.label[doc] >= 0:
            return None
        alive = self.alive[doc] & (self.read[doc] < self.lengths)
        if self.search:
            bound = np.where(alive, self.partial[doc], -np.inf)
            best = int(np.argmax(bound))
            wanted = {self.label_ids[best][:self.read[doc, best]]}
        else:
            wanted = {ids[:round] for label, ids in enumerate(self.label_ids)
                      if alive[label] and len(ids) > round}
        indices = sorted(self.index[node] for node in wanted)
        self.tokens += sum(len(self.nodes[i]) + 1 for i in indices)
        return indices

    def update(self, doc, indices, logprobs):
        """Add one round's reads and prune.

        Args:
            doc: The document.
            indices: The nodes requested, as requests() returned them.
            logprobs: Shape (len(indices), rows, targets); the row at a
                node's depth in block i was read after node indices[i].
        """
        for i, node_index in enumerate(indices):
            node = self.nodes[node_index]
            depth = len(node)
            row = logprobs[i, depth]
            for label, ids in enumerate(self.label_ids):
                # a label that is another's prefix is fully read at
                # the node that spells it and has no token to add there
                if (self.alive[doc, label] and self.read[doc, label] == depth
                        and len(ids) > depth and ids[:depth] == node):
                    self.partial[doc, label] += row[self.column[ids[depth]]]
                    self.read[doc, label] += 1
        alive = self.alive[doc]
        resolved = alive & (self.read[doc] == self.lengths)
        if resolved.any():
            scores = np.where(resolved, self.partial[doc], -np.inf)
            best = int(np.argmax(scores))
            bound = self.partial[doc]
            earlier = np.arange(len(self.label_ids)) < best
            keep = alive & ~resolved & (
                (bound > scores[best]) | ((bound == scores[best]) & earlier))
            self.alive[doc] = keep
            self.alive[doc, best] = True
            if not keep.any():
                self.label[doc] = best
                return
        if self.demand is not None and self.label[doc] < 0:
            alive = self.alive[doc]
            inside = self.demand[alive]
            if inside.all() or not inside.any():
                bound = np.where(alive, self.partial[doc], -np.inf)
                self.label[doc] = int(np.argmax(bound))


def trace_key(model_name: str, template: str, labels, label_token_ids) -> str:
    """Return the name of one classification's trace file.

    A trace holds a model's scores for one prompt and label list, so
    the key covers the model, the prompt template, and the labels with
    their token ids.
    """
    import hashlib
    import json

    text = json.dumps([model_name, template, list(labels),
                       [list(ids) for ids in label_token_ids]])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def trace_scores(label_ids, paths, targets, logprobs) -> dict:
    """Return one document's score at every label-trie node.

    The score of a node is the summed log probability of its tokens
    after the answer cue, the partial sum an adaptive rule bounds a
    label by; the empty node scores 0 and a full label's node scores
    the label.

    Args:
        label_ids: One token id sequence per label.
        paths: The proper prefix each chain holds after the answer cue.
        targets: The token ids, one per column of ``logprobs``.
        logprobs: Shape (paths, rows, targets), as ``label_path_scores``
            reads it.
    """
    where = {}
    for chain, path in enumerate(paths):
        for depth in range(len(path) + 1):
            where.setdefault(tuple(path[:depth]), (chain, depth))
    column = {token: index for index, token in enumerate(targets)}
    scores = {(): 0.0}
    for ids in label_ids:
        for depth, token in enumerate(ids):
            node = tuple(ids[:depth + 1])
            if node in scores:
                continue
            chain, row = where[tuple(ids[:depth])]
            scores[node] = (scores[tuple(ids[:depth])]
                            + float(logprobs[chain, row, column[token]]))
    return scores


def write_label_traces(path, traces) -> None:
    """Save documents' node scores as one Parquet file.

    Args:
        path: The file to write.
        traces: One dict per document mapping each trie node, a tuple
            of token ids, to its score.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    documents, nodes, scores = [], [], []
    for document, trace in enumerate(traces):
        for node, score in trace.items():
            documents.append(document)
            nodes.append(list(node))
            scores.append(score)
    pq.write_table(pa.table({
        "document": pa.array(documents, pa.int32()),
        "node": pa.array(nodes, pa.list_(pa.int32())),
        "score": pa.array(scores, pa.float32()),
    }), path)


def read_label_traces(path) -> list[dict]:
    """Load the documents' node scores ``write_label_traces`` saved."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    traces = {}
    for document, node, score in zip(table.column("document").to_pylist(),
                                     table.column("node").to_pylist(),
                                     table.column("score").to_pylist()):
        traces.setdefault(document, {})[tuple(node)] = score
    return [traces[document] for document in sorted(traces)]


def replay_rounds(label_ids, traces, search=False, demand=None) -> list:
    """Replay an adaptive rule on saved traces; returns each document's rounds.

    The rule runs on the CPU with every read answered from the trace:
    the log probability of a token after a node is the difference of
    the two nodes' scores. A round is the list of chain lengths the
    document sent, the answer cue and each requested node's tokens.

    Args:
        label_ids: One token id sequence per label.
        traces: One dict per document mapping each trie node to its
            score, as ``trace_scores`` builds it.
        search: Replay ``trie_search``; else ``trie_rounds``.
        demand: Indices of the labels a filter accepts, or None.

    Returns:
        A list with one entry per document: (rounds, label index).

    Raises:
        ValueError: A trace lacks a node the rule read.
    """
    trie = label_trie(label_ids)
    targets = trie_targets(trie)
    column = {token: i for i, token in enumerate(targets)}
    scorer = RoundScorer(label_ids, targets, len(traces), search=search,
                         demand=demand)
    depth = max(len(node) for node in scorer.nodes) + 1
    replayed = []
    for doc, trace in enumerate(traces):
        rounds = []
        for round in range(scorer.rounds):
            indices = scorer.requests(doc, round)
            if indices is None:
                break
            if not indices:
                continue
            block = np.full((len(indices), depth, len(targets)), -np.inf)
            for i, index in enumerate(indices):
                node = scorer.nodes[index]
                if node not in trace:
                    raise ValueError(f"the trace lacks node {node}")
                for token in trie[node]:
                    child = node + (token,)
                    if child not in trace:
                        raise ValueError(f"the trace lacks node {child}")
                    block[i, len(node), column[token]] = (
                        trace[child] - trace[node])
            rounds.append([len(scorer.nodes[index]) + 1 for index in indices])
            scorer.update(doc, indices, block)
        replayed.append((rounds, int(scorer.label[doc])))
    return replayed
