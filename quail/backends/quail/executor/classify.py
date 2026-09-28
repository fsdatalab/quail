"""AI.CLASSIFY labels through Quail's join loop.

Each document is an anchor: its prompt head and document stay in KV,
and the classification tail, all but its last token, is written once
after them as the anchor's frame. The label scoring rule decides what
the partner suffixes are:

- ``trie_nodes``: one suffix per label-trie node, the tail's last token
  followed by the node's label tokens. Only its last row is read; it
  gives the log probabilities of the tokens that can follow the node.
- ``label_chains``: one suffix per label, the tail's last token
  followed by all but the label's last token. Every row is read, so
  one suffix scores the whole label.
- ``trie_paths``: one suffix per deepest proper prefix of the label
  trie, the tail's last token followed by the prefix. Every row is
  read and returns every label token, so the cue's row scores all
  one-token labels at once and labels sharing leading tokens share
  the rows after them.
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from quail.backends.quail.executor.loop import InputStaging, run_join
from quail.backends.quail.executor.model import full_output_head
from quail.backends.quail.executor.readout import AsyncLabelLogprobs
from quail.execution.labels import (
    best_label,
    label_chain_scores,
    label_path_scores,
    label_scores,
    label_trie,
    trie_paths,
    trie_targets,
)
from quail.execution.reranker import RerankerBatch
from quail.execution.tokens import chain_tokens, prefix_tree

logger = logging.getLogger("quail")


@dataclass(frozen=True)
class LabelRequests:
    """What one scoring rule streams after each document, and how it scores.

    Attributes:
        frame: The classification tail but its last token, written once
            after each document.
        suffixes: The partner suffixes, each starting with the tail's
            last token.
        targets: The token ids every read returns, sorted.
        read_all_rows: Whether every row of a suffix is read, or only
            its last.
        score: Maps one document's log probabilities, shape
            (suffixes, rows, targets) with all rows read and
            (suffixes, targets) otherwise, to one score per label.
    """

    frame: list
    suffixes: list
    targets: list
    read_all_rows: bool
    score: Callable[[np.ndarray], np.ndarray]


def label_requests(spec) -> LabelRequests:
    """Return the requests of the spec's scoring rule."""
    head, tail = spec.prompt_token_parts
    if len(tail) < 1:
        raise ValueError("AI.CLASSIFY needs a question after the document")
    cue, frame = tail[-1], list(tail[:-1])
    ids = spec.label_token_ids
    if spec.scoring == "label_chains":
        targets = sorted({token for label in ids for token in label})
        return LabelRequests(
            frame, [[cue, *label[:-1]] for label in ids], targets, True,
            lambda logprobs: label_chain_scores(ids, targets, logprobs))
    if spec.scoring == "trie_paths":
        paths = trie_paths(ids)
        targets = sorted({token for label in ids for token in label})
        return LabelRequests(
            frame, [[cue, *path] for path in paths], targets, True,
            lambda logprobs: label_path_scores(ids, paths, targets, logprobs))
    trie = label_trie(ids)
    nodes = sorted(trie, key=lambda prefix: (len(prefix), prefix))
    targets = trie_targets(trie)
    return LabelRequests(
        frame, [[cue, *node] for node in nodes], targets, False,
        lambda logprobs: label_scores(ids, nodes, targets, logprobs))


def document_prefixes(spec, documents, rows) -> list:
    """Return each row's prompt head and document as one token sequence."""
    head, _ = spec.prompt_token_parts
    (alias,) = spec.aliases
    return [chain_tokens(head, documents[alias][doc]) for doc in rows]


class QuailClassifier:
    """Classify document rows with Quail's token admission and KV arena."""

    def __init__(self, state):
        self.state = state

    def classify(self, spec, rows, documents) -> RerankerBatch:
        """Return each row's label and the batch's fresh and cached tokens."""
        state = self.state
        rows = np.asarray(rows, dtype=np.int32).reshape(-1)
        requests = label_requests(spec)
        prefixes = document_prefixes(spec, documents, rows)
        frame, suffixes, targets = (
            requests.frame, requests.suffixes, requests.targets)
        readout_rows = (max(map(len, suffixes)) if requests.read_all_rows
                        else 1)
        readout = state.get("label_readout")
        if (readout is None or list(readout.targets.tolist()) != targets
                or readout.rows != readout_rows):
            torch = state["torch"]
            readout = AsyncLabelLogprobs(
                torch, torch.nn.functional, full_output_head(state["model"]),
                targets, rows=readout_rows)
            state["label_readout"] = readout
        if "input_staging" not in state:
            state["input_staging"] = InputStaging(state["torch"])
        state["input_staging"].fixed_tokens.clear()
        keys = [("classify", spec.name, index) for index in range(len(rows))]
        tree = None
        if spec.share_prefixes:
            started = time.perf_counter()
            tree = prefix_tree(prefixes, state["arena"].page_tokens)
            logger.info(
                "prefix sharing on %s: %s documents borrow %s tokens "
                "(tree built in %.2f s)", spec.name, len(prefixes),
                tree.shared_tokens, time.perf_counter() - started)
        stats = {}
        answers, spans, fresh = run_join(
            state["torch"], state["arena"], state["pipeline"], readout,
            prefixes, [suffixes], state["chunk_tokens"],
            stage_frames=[frame], anchor_keys=keys,
            staging=state["input_staging"],
            read_all_rows=requests.read_all_rows,
            prefix_tree=tree, stats=stats,
        )
        labels = np.empty(len(rows), dtype=object)
        for anchor, logprobs in answers[0].items():
            if requests.read_all_rows:
                # a one-row readout returns (suffixes, targets)
                logprobs = logprobs.reshape(len(suffixes), readout_rows, -1)
            labels[anchor] = spec.labels[best_label(requests.score(logprobs))]
        label_tokens = len(rows) * sum(map(len, suffixes))
        total = (sum(map(len, prefixes)) + len(rows) * len(frame)
                 + label_tokens)
        gpu_s = 0.0
        if state.get("gpu_timing"):
            # every chunk's answers were read, so its end event completed
            state["torch"].cuda.synchronize()
            gpu_s = sum(start.elapsed_time(end)
                        for _, start, end in spans) / 1000.0
        return RerankerBatch(labels, fresh_tokens=fresh,
                             cached_tokens=total - fresh,
                             label_tokens=label_tokens,
                             borrowed_tokens=stats.get("borrowed_tokens", 0),
                             gpu_s=gpu_s,
                             chunks=len(spans) if state.get("gpu_timing") else 0)
