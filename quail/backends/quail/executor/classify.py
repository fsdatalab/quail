"""AI.CLASSIFY labels through Quail's join loop.

Each document is an anchor: its prompt head and document stay in KV,
and the classification tail, all but its last token, is written once
after them as the anchor's frame. Every label-trie node is a partner
suffix: the tail's last token followed by the node's label tokens. The
suffix's last row reads the log probabilities of the tokens that can
follow the node, so one pass over the trie gives every label's score.
"""

import numpy as np

from quail.backends.quail.executor.loop import InputStaging, run_join
from quail.backends.quail.executor.model import full_output_head
from quail.backends.quail.executor.readout import AsyncLabelLogprobs
from quail.execution.labels import (
    best_label,
    label_scores,
    label_trie,
    trie_targets,
)
from quail.execution.reranker import RerankerBatch
from quail.execution.tokens import chain_tokens


def classify_inputs(spec, documents, rows):
    """Return the join loop's inputs for one batch of documents.

    Returns:
        (prefixes, frame, suffixes, trie prefixes, target token ids):
        one prefix per row, the frame written after each, one suffix
        per trie node, the node each suffix reads after, and the
        token ids every read returns.
    """
    head, tail = spec.prompt_token_parts
    if len(tail) < 1:
        raise ValueError("AI.CLASSIFY needs a question after the document")
    (alias,) = spec.aliases
    trie = label_trie(spec.label_token_ids)
    nodes = sorted(trie, key=lambda prefix: (len(prefix), prefix))
    prefixes = [chain_tokens(head, documents[alias][doc]) for doc in rows]
    suffixes = [[tail[-1], *node] for node in nodes]
    return prefixes, list(tail[:-1]), suffixes, nodes, trie_targets(trie)


class QuailClassifier:
    """Classify document rows with Quail's token admission and KV arena."""

    def __init__(self, state):
        self.state = state

    def classify(self, spec, rows, documents) -> RerankerBatch:
        """Return each row's label and the batch's fresh and cached tokens."""
        state = self.state
        rows = np.asarray(rows, dtype=np.int32).reshape(-1)
        prefixes, frame, suffixes, nodes, targets = classify_inputs(
            spec, documents, rows)
        readout = state.get("label_readout")
        if readout is None or list(readout.targets.tolist()) != targets:
            torch = state["torch"]
            readout = AsyncLabelLogprobs(
                torch, torch.nn.functional, full_output_head(state["model"]),
                targets)
            state["label_readout"] = readout
        if "input_staging" not in state:
            state["input_staging"] = InputStaging(state["torch"])
        state["input_staging"].fixed_tokens.clear()
        keys = [("classify", spec.name, index) for index in range(len(rows))]
        answers, _, fresh = run_join(
            state["torch"], state["arena"], state["pipeline"], readout,
            prefixes, [suffixes], state["chunk_tokens"],
            stage_frames=[frame], anchor_keys=keys,
            staging=state["input_staging"],
        )
        labels = np.empty(len(rows), dtype=object)
        for anchor, logprobs in answers[0].items():
            scores = label_scores(spec.label_token_ids, nodes, targets,
                                  logprobs)
            labels[anchor] = spec.labels[best_label(scores)]
        total = (sum(map(len, prefixes)) + len(rows) * len(frame)
                 + len(rows) * sum(map(len, suffixes)))
        return RerankerBatch(labels, fresh_tokens=fresh,
                             cached_tokens=total - fresh)
