"""AI.CLASSIFY labels through Quail's join loop.

Each document is an anchor: its prompt head and document stay in KV,
and the classification tail, all but its last token, is written once
after them as the anchor's frame. The label scoring rule decides what
the partner suffixes are:

- ``trie_paths``: one suffix per deepest proper prefix of the label
  trie, the tail's last token followed by the prefix. Every row is
  read and returns every label token, so the cue's row scores all
  one-token labels at once and labels sharing leading tokens share
  the rows after them.
- ``trie_tree``: the whole trie as one suffix: the cue and every
  node's token once, split into chains that each follow first
  children. Each chain is a causal segment and reads the ancestors
  above it from earlier chains, so every node is computed once and
  every row is read.
- ``canvas``: a diffusion model's rule. Every label ends with the
  answer end token, and the canvas holds as many rows as the longest.
  One suffix per label-trie node, the tail's last token followed by
  the node's tokens, with the canvas rows left after them; the first
  canvas row is read. Every suffix fills the same canvas, so no label
  is favored for its length, and a row is conditioned on the node's
  tokens written before it.
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from quail.backends.quail.executor.loop import InputStaging
from quail.backends.quail.executor.model import full_output_head
from quail.backends.quail.executor.readout import AsyncLabelLogprobs
from quail.backends.quail.executor.stages import Stage, run_stages
from quail.execution.labels import (
    best_label,
    label_path_scores,
    label_scores,
    label_trie,
    tree_scores,
    trie_chains,
    trie_paths,
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
        chains: Under ``trie_tree``, the chains the one suffix packs.
        canvas_rows: Under ``canvas``, the rows of the canvas, and per
            suffix the rows left after its node.
    """

    frame: list
    suffixes: list
    targets: list
    read_all_rows: bool
    score: Callable[[np.ndarray], np.ndarray]
    chains: list | None = None
    canvas_rows: int = 0
    canvas_widths: list | None = None


def label_requests(spec, targets=None) -> LabelRequests:
    """Return the requests of the spec's scoring rule.

    Args:
        spec: The classification.
        targets: The token ids the readout returns, sorted; None reads
            the spec's own label tokens. A chain's stages share one
            readout over every stage's label tokens.
    """
    head, tail = spec.prompt_token_parts
    if len(tail) < 1:
        raise ValueError("AI.CLASSIFY needs a question after the document")
    cue, frame = tail[-1], list(tail[:-1])
    ids = spec.label_token_ids
    if targets is None:
        targets = sorted({token for label in ids for token in label})
    if spec.scoring == "canvas":
        trie = label_trie(ids)
        nodes = sorted(trie, key=lambda prefix: (len(prefix), prefix))
        rows = max(len(label) for label in ids)
        return LabelRequests(
            frame, [[cue, *node] for node in nodes], targets, False,
            lambda logprobs: label_scores(ids, nodes, targets, logprobs),
            canvas_rows=rows, canvas_widths=[rows - len(node) for node in nodes])
    if spec.scoring == "trie_tree":
        chains = trie_chains(ids)
        tokens = [cue if node == () else node[-1]
                  for nodes, _, _ in chains for node in nodes]
        return LabelRequests(
            frame, [tokens], targets, True,
            lambda logprobs: tree_scores(ids, chains, targets, logprobs[0]),
            chains=chains)
    if spec.scoring == "trie_paths":
        paths = trie_paths(ids)
        return LabelRequests(
            frame, [[cue, *path] for path in paths], targets, True,
            lambda logprobs: label_path_scores(ids, paths, targets, logprobs))
    raise ValueError(f"unknown label scoring rule {spec.scoring!r}")


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
        """Return each row's labels and the batch's fresh and cached tokens.

        A chain's stages run as the stages of one join call: a document
        goes on to the next stage while its KV is resident if the gate
        accepts its label, and every stage's requests share one readout.
        """
        state = self.state
        rows = np.asarray(rows, dtype=np.int32).reshape(-1)
        specs = spec.chain
        targets = sorted({token for stage in specs
                          for ids in stage.label_token_ids for token in ids})
        requests = [label_requests(stage, targets) for stage in specs]
        read_all = requests[0].read_all_rows
        if any(request.read_all_rows != read_all for request in requests):
            raise ValueError("a classify chain's stages read rows one way")
        prefixes = document_prefixes(spec, documents, rows)
        canvas = None
        if spec.scoring == "canvas":
            # every suffix's canvas is a leading part of the run's; a
            # canvas classification never chains, so one canvas length
            if len(specs) > 1:
                raise ValueError("a canvas classification runs alone")
            canvas = state["pipeline"].canvas_rows(requests[0].canvas_rows)
        readout_rows = (max(len(suffix) for request in requests
                            for suffix in request.suffixes)
                        if read_all else 1)
        # every label read at the same row needs no normalizer: one-token
        # labels at the cue row
        same_rows = all(len(ids) == 1 for stage in specs
                        for ids in stage.label_token_ids)
        normalize = not same_rows
        readout = state.get("label_readout")
        if (readout is None or list(readout.targets.tolist()) != targets
                or readout.rows != readout_rows
                or getattr(readout, "normalize", normalize) != normalize):
            torch = state["torch"]
            head = full_output_head(state["model"])
            readout = AsyncLabelLogprobs(
                torch, torch.nn.functional, head, targets, rows=readout_rows,
                normalize=normalize)
            state["label_readout"] = readout
            logger.info("label readout: head %s x %s in %s, %s targets, "
                        "%s rows per request, %s", *head.shape,
                        str(head.dtype).replace("torch.", ""), len(targets),
                        readout_rows,
                        "normalized" if normalize else "targets' logits")
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

        labels = [np.full(len(rows), None, dtype=object) for _ in specs]

        def label_of(index, logprobs):
            if read_all:
                # a one-row readout returns (suffixes, targets)
                logprobs = logprobs.reshape(
                    len(requests[index].suffixes), readout_rows, -1)
            return specs[index].labels[
                best_label(requests[index].score(logprobs))]

        def gate(index, key):
            """DROP when the filter before classification index rejects."""
            accepted = spec.stages[index - 1].accepted
            if accepted is not None and labels[index - 1][key[2]] not in accepted:
                return Stage.DROP
            return None

        # one stage per classification; a later classification's stage
        # gates on the label before it
        stages = []
        for index, (stage_spec, request) in enumerate(zip(specs, requests)):
            def whole(anchor, row, index=index):
                labels[index][anchor] = label_of(index, row)
                return True

            stages.append(Stage(
                suffixes=request.suffixes, readout=readout,
                frame=request.frame, decide=whole,
                requests=((lambda key, index=index: gate(index, key))
                          if index else None),
                read_all_rows=read_all, label=stage_spec.name,
                chains=request.chains, canvas_widths=request.canvas_widths))

        stats = {}
        answers, spans, fresh = run_stages(
            state["torch"], state["arena"], state["pipeline"], stages,
            prefixes, state["chunk_tokens"], anchor_keys=keys,
            staging=state["input_staging"], prefix_tree=tree, stats=stats,
            label=f"classify {spec.name}", canvas=canvas)
        label_tokens = 0
        streamed = 0     # frame and suffix tokens packed after documents
        for index, request in enumerate(requests):
            first = answers[index]
            for anchor, logprobs in first.items():
                if labels[index][anchor] is None:
                    labels[index][anchor] = label_of(index, logprobs)
            suffix_tokens = len(first) * (
                sum(map(len, request.suffixes))
                + sum(request.canvas_widths or ()))
            label_tokens += suffix_tokens
            # a frame equal to the stage before's is already in KV
            written = index == 0 or request.frame != requests[index - 1].frame
            streamed += suffix_tokens + (len(first) * len(request.frame)
                                         if written else 0)
        total = sum(map(len, prefixes)) + streamed
        gpu_s = 0.0
        if state.get("gpu_timing"):
            # every chunk's answers were read, so its end event completed
            state["torch"].cuda.synchronize()
            gpu_s = sum(start.elapsed_time(end)
                        for _, start, end in spans) / 1000.0
        return RerankerBatch(
            labels[0], fresh_tokens=fresh, cached_tokens=total - fresh,
            label_tokens=label_tokens,
            borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0),
            gpu_s=gpu_s,
            chunks=len(spans) if state.get("gpu_timing") else 0,
            later={stage.spec.name: labels[index + 1]
                   for index, stage in enumerate(spec.stages)})
