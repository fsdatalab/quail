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
- ``trie_decode``: one stage per trie depth. Each round a document
  sends the chain of the node it has decoded so far and appends the
  likeliest child token read after it, until the node is a whole
  label (GreedyDecoder); the label is the greedy path.
- ``trie_tree``: the whole trie as one suffix: the cue and every
  node's token once, split into chains that each follow first
  children. Each chain is a causal segment and reads the ancestors
  above it from earlier chains, so every node is computed once and
  every row is read.
- ``canvas``: a diffusion model's rule. The model decodes its answer
  with its own denoising steps (denoise.py), and the text is matched
  to a label (match_label). Each step is a stage whose one suffix is
  the tail's last token followed by the document's canvas; a document
  goes on to the next step until its canvas is done. A document whose
  answer names no label gets None.
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from quail.backends.quail.executor.denoise import (
    CANVAS_SEED,
    AsyncCanvasReadout,
    ConditioningRows,
    DocumentCanvas,
    answer_text,
)
from quail.backends.quail.executor.loop import InputStaging
from quail.backends.quail.executor.model import full_output_head
from quail.backends.quail.executor.readout import AsyncLabelLogprobs
from quail.backends.quail.executor.stages import Stage, run_stages
from quail.execution.labels import (
    GreedyDecoder,
    best_label,
    label_path_scores,
    label_trie,
    match_label,
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
            (suffixes, targets) otherwise, to one score per label;
            None under ``trie_decode``, which decodes as it reads.
        chains: Under ``trie_tree``, the chains the one suffix packs.
        rounds: Under ``trie_decode``, the rounds a document may need;
            each round offers every node's chain and a document sends
            the chain of the node it has decoded so far.
    """

    frame: list
    suffixes: list
    targets: list
    read_all_rows: bool
    score: Callable[[np.ndarray], np.ndarray] | None
    chains: list | None = None
    rounds: int = 0


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
    if spec.scoring == "trie_decode":
        nodes = sorted(label_trie(ids), key=lambda prefix: (len(prefix), prefix))
        return LabelRequests(
            frame, [[cue, *node] for node in nodes], targets, False, None,
            rounds=max(len(label) for label in ids))
    if spec.scoring == "canvas":
        raise ValueError("the canvas rule decodes answers; it scores no labels")
    raise ValueError(f"unknown label scoring rule {spec.scoring!r}")


def document_prefixes(spec, documents, rows) -> list:
    """Return each row's prompt head and document as one token sequence."""
    head, _ = spec.prompt_token_parts
    (alias,) = spec.aliases
    return [chain_tokens(head, documents[alias][doc]) for doc in rows]


def _tokenizer(state):
    """The checkpoint's tokenizer, which decodes answers; kept on the model."""
    if "tokenizer" in state:
        return state["tokenizer"]
    model = state["model"]
    tokenizer = getattr(model, "quail_tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer

        path = model.quail_vllm_config.model_config.tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model.quail_tokenizer = tokenizer
    return tokenizer


class ClassifyStages:
    """The stages one classification asks of each document, and its labels.

    One stage per classification of the chain, or one per trie depth
    under ``trie_decode``; a later classification's first stage gates
    on the label before it. The stages read the label readout kept on
    the state, rebuilt when the targets or rows change.

    Args:
        state: The executor state.
        spec: The classification and its chained stages.
        count: How many documents the stages run over.
        index_of: Callable(admission key) -> the document's index.
        on_label: Optional callable(index, label) run when a document's
            first classification labels it.
    """

    def __init__(self, state, spec, count, index_of, on_label=None):
        self.spec = spec
        self.specs = specs = spec.chain
        self.index_of = index_of
        targets = sorted({token for stage in specs
                          for ids in stage.label_token_ids for token in ids})
        self.requests = requests = [label_requests(stage, targets)
                                    for stage in specs]
        read_all = requests[0].read_all_rows
        if any(request.read_all_rows != read_all for request in requests):
            raise ValueError("a classify chain's stages read rows one way")
        self.readout_rows = readout_rows = (
            max(len(suffix) for request in requests
                for suffix in request.suffixes) if read_all else 1)
        self.read_all = read_all
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
        self.readout = readout
        self.labels = [np.full(count, None, dtype=object) for _ in specs]
        self.decoders = {}
        self.stages = []
        for index, (stage_spec, request) in enumerate(zip(specs, requests)):
            if not request.rounds:
                def whole(anchor, row, index=index):
                    self._set(index, anchor, self.label_of(index, row),
                              on_label)
                    return True

                self.stages.append(Stage(
                    suffixes=request.suffixes, readout=readout,
                    frame=request.frame, decide=whole,
                    requests=((lambda key, index=index: self.gate(index, key))
                              if index else None),
                    read_all_rows=read_all, label=stage_spec.name,
                    chains=request.chains))
                continue
            decoder = GreedyDecoder(stage_spec.label_token_ids, targets, count)
            self.decoders[index] = decoder
            for round in range(request.rounds):
                def ask(key, index=index, round=round, decoder=decoder):
                    if round == 0 and index and self.gate(index, key) is Stage.DROP:
                        return Stage.DROP
                    nodes = decoder.requests(self.index_of(key))
                    # a resolved document has nothing left to read
                    return Stage.DROP if nodes is None else nodes

                def round_read(anchor, row, index=index, decoder=decoder,
                               names=stage_spec.labels):
                    decoder.update(anchor, np.asarray(row).reshape(-1))
                    if decoder.label[anchor] >= 0:
                        self._set(index, anchor,
                                  names[decoder.label[anchor]], on_label)
                    return True

                self.stages.append(Stage(
                    suffixes=request.suffixes, readout=readout,
                    frame=request.frame, requests=ask, decide=round_read,
                    read_all_rows=False,
                    label=f"{stage_spec.name} round {round}"))

    def _set(self, index, anchor, label, on_label):
        self.labels[index][anchor] = label
        if index == 0 and on_label is not None:
            on_label(anchor, label)

    def label_of(self, index, logprobs):
        """The label classification ``index`` gives from its read rows."""
        if self.read_all:
            # a one-row readout returns (suffixes, targets)
            logprobs = logprobs.reshape(
                len(self.requests[index].suffixes), self.readout_rows, -1)
        return self.specs[index].labels[
            best_label(self.requests[index].score(logprobs))]

    def gate(self, index, key):
        """DROP when the filter before classification ``index`` rejects."""
        accepted = self.spec.stages[index - 1].accepted
        label = self.labels[index - 1][self.index_of(key)]
        if accepted is not None and label not in accepted:
            return Stage.DROP
        return None

    def finish(self, answers) -> tuple[int, int]:
        """Label every document from its answers; returns the token counts.

        Returns:
            (label tokens, streamed tokens): the suffix tokens read, and
            those plus the frames written after the documents.
        """
        label_tokens = 0
        streamed = 0
        position = 0
        for index, request in enumerate(self.requests):
            first = answers[position]
            if index in self.decoders:
                suffix_tokens = self.decoders[index].tokens
            else:
                for anchor, logprobs in first.items():
                    if self.labels[index][anchor] is None:
                        self.labels[index][anchor] = self.label_of(
                            index, logprobs)
                suffix_tokens = len(first) * sum(map(len, request.suffixes))
            label_tokens += suffix_tokens
            # a frame equal to the stage before's is already in KV
            written = (index == 0
                       or request.frame != self.requests[index - 1].frame)
            streamed += suffix_tokens + (len(first) * len(request.frame)
                                         if written else 0)
            position += max(1, request.rounds)
        return label_tokens, streamed

    def later(self) -> dict:
        """The chained classifications' labels by output name."""
        return {stage.spec.name: self.labels[index + 1]
                for index, stage in enumerate(self.spec.stages)}


class QuailClassifier:
    """Classify document rows with Quail's token admission and KV arena."""

    def __init__(self, state):
        self.state = state

    def classify(self, spec, rows, documents) -> RerankerBatch:
        """Return each row's labels and the batch's fresh and cached tokens.

        A chain's stages run as the stages of one join call: a document
        goes on to the next stage while its KV is resident if the gate
        accepts its label, and every stage's requests share one readout.
        The ``canvas`` rule decodes instead (decode).
        """
        state = self.state
        rows = np.asarray(rows, dtype=np.int32).reshape(-1)
        prefixes = document_prefixes(spec, documents, rows)
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
        if spec.scoring == "canvas":
            return self.decode(spec, rows, prefixes, keys, tree)
        plan = ClassifyStages(state, spec, len(rows), lambda key: key[2])
        stats = {}
        answers, spans, fresh = run_stages(
            state["torch"], state["arena"], state["pipeline"], plan.stages,
            prefixes, state["chunk_tokens"], anchor_keys=keys,
            staging=state["input_staging"], prefix_tree=tree, stats=stats,
            label=f"classify {spec.name}")
        label_tokens, streamed = plan.finish(answers)
        total = sum(map(len, prefixes)) + streamed
        return self._batch(plan.labels[0], fresh, total - fresh, label_tokens,
                           stats, spans, plan.later())

    def decode(self, spec, rows, prefixes, keys, tree) -> RerankerBatch:
        """Label each row by decoding its answer with the model's denoising steps.

        Each denoising step is a stage: the tail's last token and the
        document's canvas, after the head, the document, and the rest
        of the tail, which are computed once and stay in KV. A document
        goes on to the next step until its canvas is done; its answer
        text is then matched to a label, or None.
        """
        state = self.state
        if spec.stages:
            raise ValueError("a canvas classification runs alone")
        model_spec = state["model_spec"]
        settings = model_spec.denoising
        if settings is None:
            raise ValueError(f"{model_spec.name!r} has no denoising sampler")
        _, tail = spec.prompt_token_parts
        if len(tail) < 1:
            raise ValueError("AI.CLASSIFY needs a question after the document")
        cue, frame = tail[-1], list(tail[:-1])
        torch = state["torch"]
        pipeline = state["pipeline"]
        head = full_output_head(state["model"])
        conditioning = ConditioningRows(
            torch, head.shape[1], settings.canvas_rows, head.dtype, head.device)
        readout = AsyncCanvasReadout(torch, torch.nn.functional, head,
                                     pipeline.normalizer, settings, conditioning)
        tokenizer = _tokenizer(state)
        canvases = {}
        labels = np.full(len(rows), None, dtype=object)

        def canvas(anchor):
            document = canvases.get(anchor)
            if document is None:
                document = canvases[anchor] = DocumentCanvas(
                    settings, model_spec.vocab, (CANVAS_SEED, int(rows[anchor])),
                    conditioning.take())
            return document.canvas, document.conditioning_row

        def step(anchor, record):
            document = canvases[anchor]
            if document.update(record[0]["tokens"], record[0]["entropy"]):
                return True
            text = answer_text(tokenizer, document.tokens,
                               settings.stop_token_ids)
            labels[anchor] = match_label(text, spec.labels)
            conditioning.release(document.conditioning_row)
            return False

        stages = [Stage(suffixes=[[cue]], readout=readout, frame=frame,
                        decide=step, read_all_rows=True, single=True,
                        label=spec.name, canvas=canvas,
                        canvas_rows=settings.canvas_rows)
                  for _ in range(settings.max_steps)]
        stats = {}
        _, spans, fresh = run_stages(
            torch, state["arena"], pipeline, stages, prefixes,
            state["chunk_tokens"], anchor_keys=keys,
            staging=state["input_staging"], prefix_tree=tree, stats=stats,
            label=f"classify {spec.name}", conditioning=conditioning)
        steps = sum(document.steps for document in canvases.values())
        unmatched = sum(label is None for label in labels)
        logger.info("classify %s: %s denoising steps over %s documents; "
                    "%s answers name no label", spec.name, steps,
                    len(canvases), unmatched)
        label_tokens = steps * (1 + settings.canvas_rows)
        total = (sum(map(len, prefixes)) + len(canvases) * len(frame)
                 + label_tokens)
        return self._batch(labels, fresh, total - fresh, label_tokens, stats,
                           spans, {})

    def _batch(self, labels, fresh, cached, label_tokens, stats, spans,
               later) -> RerankerBatch:
        """The labels and token counts of one run, with its GPU time when timed."""
        state = self.state
        gpu_s = 0.0
        if state.get("gpu_timing"):
            # every chunk's answers were read, so its end event completed
            state["torch"].cuda.synchronize()
            gpu_s = sum(start.elapsed_time(end)
                        for _, start, end in spans) / 1000.0
        return RerankerBatch(
            labels, fresh_tokens=fresh, cached_tokens=cached,
            label_tokens=label_tokens,
            borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0),
            gpu_s=gpu_s,
            chunks=len(spans) if state.get("gpu_timing") else 0,
            later=later)
