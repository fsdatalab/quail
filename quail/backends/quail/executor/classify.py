"""Execute classification with the shared stage scheduler.

Letters scores category letters in one pass. Tree scoring evaluates all
label token sequences. Greedy decoding chooses one allowed token per
round, keeping the chosen path in the document's KV after the frame.
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from quail.backends.quail.executor.model import full_output_head
from quail.backends.quail.executor.parts import input_staging
from quail.backends.quail.executor.readout import AsyncLabelLogprobs
from quail.backends.quail.executor.stages import Stage, run_stages
from quail.backends.quail.executor.state import QueryExecutionState
from quail.execution.labels import (
    GreedyDecoder,
    best_label,
    letter_scores,
    tree_scores,
    trie_chains,
)
from quail.execution.reranker import RerankerBatch
from quail.execution.tokens import chain_tokens, prefix_tree
from quail.specs.base import CANVAS_ENTROPY_NATS, CANVAS_SEED

logger = logging.getLogger("quail")


@dataclass(frozen=True)
class LabelRequests:
    """Token requests and scoring callbacks for one classification method.

    Attributes:
        frame: Prompt tail tokens before the final answer cue token, written
            once after each document.
        suffixes: Request token sequences. Scoring rules begin each with the
            answer cue; greedy decoding has the cue, then one single-token
            request per target.
        targets: Sorted token IDs returned by the readout.
        read_all_rows: Whether every suffix row is read, rather than its last.
        score: Callback converting one document's readout values to category
            scores. Values have shape (suffixes, rows, targets) when all rows
            are read, otherwise (suffixes, targets). None uses greedy decoding.
        chains: Optional trie chains packed into the sole suffix for tree scoring.
        rounds: Maximum decoder rounds. Each round feeds one token, kept in
            the document's KV after the frame. Zero scores all labels in one
            stage.
    """

    frame: list
    suffixes: list
    targets: list
    read_all_rows: bool
    score: Callable[[np.ndarray], np.ndarray] | None
    chains: list | None = None
    rounds: int = 0


def label_requests(spec, targets=None) -> LabelRequests:
    """Build token requests and a score function for a classification.

    Args:
        spec: ClassifySpec containing the prompt and category token sequences.
        targets: Sorted token IDs returned by the readout. None uses all token
            IDs present in the category sequences.

    Returns:
        LabelRequests describing the frame, suffixes, and scoring procedure.

    Raises:
        ValueError: The prompt has no tail tokens or scoring is unknown.
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
    if spec.scoring == "letters":
        return LabelRequests(
            frame, [[cue]], targets, True,
            lambda logprobs: letter_scores(ids, targets, logprobs[0, 0]))
    if spec.scoring == "trie_decode":
        return LabelRequests(
            frame, [[cue], *([token] for token in targets)], targets, False,
            None, rounds=max(len(label) for label in ids))
    raise ValueError(f"unknown label scoring rule {spec.scoring!r}")


def document_prefixes(spec, documents, rows) -> list:
    """Combine the prompt head and document tokens for each input row.

    Args:
        spec: Single-document classification specification.
        documents: Tokenized documents indexed by table alias and document ID.
        rows: Document IDs to classify, in input order.

    Returns:
        One token sequence per input row.
    """
    head, _ = spec.prompt_token_parts
    (alias,) = spec.aliases
    return [chain_tokens(head, documents[alias][doc]) for doc in rows]


@dataclass
class JoinPartners:
    """Partner documents and the pairs retained by a join.

    Attributes:
        ids: The partner documents' ids, in the order the suffixes take.
        documents: Each partner document's token ids, in that order.
        kept: Callable(anchor index) -> the indices into ``ids`` of the
            partners the anchor is classified with; asked when the
            anchor reaches the stage, after the join that pairs them.
    """

    ids: list
    documents: list
    kept: Callable[[int], list]


def _label_readout(state: QueryExecutionState, targets, rows: int, normalize: bool):
    """Return the model's label readout, matching targets and shape."""
    readout = state.loaded_model.label_readout
    if (readout is None or list(readout.targets.tolist()) != targets
            or readout.rows != rows
            or getattr(readout, "normalize", normalize) != normalize):
        torch = state.torch
        head = full_output_head(state.loaded_model.model)
        readout = AsyncLabelLogprobs(
            torch, torch.nn.functional, head, targets, rows=rows,
            normalize=normalize)
        state.loaded_model.label_readout = readout
        logger.info("label readout: head %s x %s in %s, %s targets, "
                    "%s rows per request, %s", *head.shape,
                    str(head.dtype).replace("torch.", ""), len(targets),
                    rows, "normalized" if normalize else "targets' logits")
    return readout


def _softmax(scores) -> np.ndarray:
    """Normalize category scores into probabilities along the last axis."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return scores / scores.sum(axis=-1, keepdims=True)


class ClassifyStages:
    """Classification stages and the labels they produce.

    Tree and letter scoring use one stage. Greedy decoding uses one stage
    per token depth and skips remaining stages after selecting a label.
    Diffusion models can add a stage for repeated draws. Pair classification
    uses one stage over the partner documents retained by the join.

    Args:
        state: Current query settings and loaded model resources.
        spec: Classification specification.
        count: Number of anchor documents.
        index_of: Callable mapping an admission key to a document index.
        on_label: Optional callback receiving a document index and its label.
        partners: JoinPartners for pair classification, or None for individual
            documents.
    """

    canvas_rows = 0         # rows of a letters stage's canvas on a canvas model
    draws = 1               # noise draws a canvas letters read may average
    probabilities = None    # (documents, labels) label probabilities, when asked
    seeds = None            # document index -> its canvas seed; the index itself
    partners = None         # JoinPartners, for joined rows
    pair_labels = None      # (anchor index, partner index) -> label, for joined rows

    def __init__(self, state: QueryExecutionState, spec, count, index_of, on_label=None,
                 partners=None):
        self.spec = spec
        self.index_of = index_of
        if spec.partner is not None:
            self._joined_stages(state, spec, count, partners)
            return
        targets = sorted({token for ids in spec.label_token_ids for token in ids})
        self.request = request = label_requests(spec, targets)
        read_all = request.read_all_rows
        self.read_all = read_all
        self.readout_rows = readout_rows = (
            max(map(len, request.suffixes)) if read_all else 1)
        # a letters stage on a canvas model reads its canvas's rows
        answer_canvas = getattr(state.loaded_model.model_spec, "answer_canvas", None)
        if spec.scoring == "letters" and answer_canvas is not None:
            self.canvas_rows = answer_canvas.rows
            self.readout_rows = readout_rows = self.canvas_rows
            self.draws = max(1, spec.draws)
        # every label read at the same row needs no normalizer: one-token
        # labels at the cue row
        same_rows = all(len(ids) == 1 for ids in spec.label_token_ids)
        self.readout = readout = _label_readout(state, targets, readout_rows,
                                                not same_rows)
        self.labels = np.full(count, None, dtype=object)
        if spec.probabilities:
            self.probabilities = np.full((count, len(spec.labels)), np.nan)
        self.decoder = None
        self.stages = []
        if not request.rounds:
            def whole(anchor, row):
                if self.draws > 1:
                    self._first_draw(anchor, row, on_label)
                else:
                    self._choose(anchor, _softmax(self._scores(row)), on_label)
                return True

            canvas = self._letter_canvas(state) if self.canvas_rows else None
            self.stages.append(Stage(
                suffixes=request.suffixes, readout=readout,
                frame=request.frame, decide=whole,
                read_all_rows=read_all, label=spec.name,
                chains=request.chains,
                **({} if canvas is None else dict(
                    single=True, canvas=lambda anchor: canvas(anchor, 0),
                    canvas_rows=self.canvas_rows))))
            if self.draws > 1:
                self.stages.append(self._more_draws(canvas, on_label))
            return
        self.decoder = decoder = GreedyDecoder(spec.label_token_ids, targets, count)

        def ask(key):
            nodes = decoder.requests(self.index_of(key))
            return Stage.SKIP if nodes is None else nodes

        def round_read(anchor, row):
            decoder.update(anchor, np.asarray(row).reshape(-1))
            if decoder.label[anchor] >= 0:
                self._set(anchor, spec.labels[decoder.label[anchor]], on_label)
            return True

        for round in range(request.rounds):
            self.stages.append(Stage(
                suffixes=request.suffixes, readout=readout,
                frame=request.frame, requests=ask, decide=round_read,
                read_all_rows=False, single=True, append=True,
                label=f"{spec.name} round {round}"))

    def _joined_stages(self, state: QueryExecutionState, spec, count, partners):
        """Initialize one classification stage over the pairs retained by a join."""
        if spec.scoring != "letters":
            raise ValueError("a classification of joined rows reads letters, "
                             f"not {spec.scoring!r}")
        if partners is None:
            raise ValueError(
                "a classification of joined rows needs the join's partners")
        note, partner_label = spec.join_layout
        targets = sorted({token for ids in spec.label_token_ids for token in ids})
        request = label_requests(spec, targets)
        self.request = request
        self.readout_rows = 1
        self.readout = readout = _label_readout(state, targets, 1, False)
        self.read_all = True
        self.partners = partners
        self.pair_labels = {}
        self.labels = np.full(count, None, dtype=object)
        self.decoder = None
        question = list(request.frame)
        (cue,) = request.suffixes[0]
        # suffix p is partner p's block and the cue; its last row is read
        blocks = [list(partner_label) + list(document) + question
                  for document in partners.documents]
        self.block_tokens = [len(tokens) for tokens in blocks]
        suffixes = [tokens + [cue] for tokens in blocks]

        def ask(key):
            return list(partners.kept(self.index_of(key)))

        def score(anchor, row):
            kept = partners.kept(anchor)
            logprobs = np.asarray(row).reshape(len(kept), -1)
            for position, partner in enumerate(kept):
                self.pair_labels[(anchor, partner)] = spec.labels[best_label(
                    letter_scores(spec.label_token_ids, targets,
                                  logprobs[position]))]
            return True

        self.stages = [Stage(
            suffixes=suffixes, readout=readout, frame=list(note),
            requests=ask, decide=score, read_all_rows=True,
            read_rows=[1] * len(suffixes), label=spec.name)]

    def _letter_canvas(
            self, state: QueryExecutionState) -> Callable[[int, int], np.ndarray]:
        """Build a callable that creates reproducible answer canvas tokens.

        Args:
            state: Query state containing the model's answer canvas settings.

        Returns:
            A callable taking a document index and draw number and returning
            canvas token IDs. Its seed is independent of batch membership.
        """
        model_spec = state.loaded_model.model_spec
        settings = model_spec.answer_canvas
        template = np.full(settings.rows, settings.pad_id, dtype=np.int64)
        template[1] = settings.turn_close_id

        def canvas(anchor, draw):
            rng = np.random.default_rng((CANVAS_SEED, self.seed(anchor), draw))
            ids = template.copy()
            ids[0] = rng.integers(0, model_spec.vocab)
            return ids

        return canvas

    def _more_draws(self, canvas, on_label) -> Stage:
        """Build a stage that repeats uncertain diffusion classifications.

        Documents labeled by the first draw skip this stage. Other documents
        run the remaining draws and use the mean category probabilities.

        Args:
            canvas: Callable taking a document index and draw number and returning
                canvas tokens.
            on_label: Optional callback receiving the document index and label.

        Returns:
            A Stage that submits the remaining draws and records the final label.
        """
        more = self.draws - 1
        self.draw_probs = {}
        self.extended = 0

        def ask(key):
            return (Stage.SKIP if self.labels[self.index_of(key)] is not None
                    else None)

        def average(anchor, row):
            probs = self._probs(row, more)[0]
            self.extended += 1
            mean = (self.draw_probs[anchor] + probs.sum(axis=0)) / self.draws
            self._choose(anchor, mean, on_label)
            return True

        return Stage(
            suffixes=self.request.suffixes * more, readout=self.readout,
            frame=self.request.frame, requests=ask, decide=average,
            read_all_rows=True, label=f"{self.spec.name} draws",
            canvas=lambda anchor: np.stack(
                [canvas(anchor, draw) for draw in range(1, self.draws)]),
            canvas_rows=self.canvas_rows)

    def _probs(self, logits, draws) -> tuple[np.ndarray, np.ndarray]:
        """Compute category probabilities and entropy for each draw.

        Args:
            logits: Readout values for all draws, rows, and target tokens.
            draws: Number of draws represented in logits.

        Returns:
            A tuple of probabilities with shape (draws, labels) and entropy
            values with shape (draws,). Entropy is measured in nats.
        """
        logits = np.asarray(logits).reshape(draws, self.readout_rows, -1)
        probs = _softmax(np.stack([self.request.score(logits[draw:draw + 1])
                                   for draw in range(draws)]))
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-30)), axis=1)
        return probs, entropy

    def _first_draw(self, anchor, row, on_label) -> None:
        """Record the first draw and label the document if entropy is low.

        Args:
            anchor: Document index.
            row: Readout values for the first draw.
            on_label: Optional callback receiving the document index and label.
        """
        probs, entropy = self._probs(row, 1)
        self.draw_probs[anchor] = probs[0]
        if entropy[0] <= CANVAS_ENTROPY_NATS:
            self._choose(anchor, probs[0], on_label)

    def seed(self, anchor) -> int:
        """Return the configured document seed, or its index if none is set."""
        return int(anchor if self.seeds is None else self.seeds[anchor])

    def _set(self, anchor, label, on_label):
        self.labels[anchor] = label
        if on_label is not None:
            on_label(anchor, label)

    def _choose(self, anchor, probs, on_label):
        """Record the most probable category and optionally its probabilities."""
        if self.probabilities is not None:
            self.probabilities[anchor] = probs
        self._set(anchor, self.spec.labels[best_label(probs)], on_label)

    def _scores(self, logprobs):
        """Compute one score per category from the readout values."""
        if self.read_all:
            # a one-row readout returns (suffixes, targets)
            logprobs = logprobs.reshape(
                len(self.request.suffixes), self.readout_rows, -1)
        return self.request.score(logprobs)

    def finish(self, answers) -> tuple[int, int]:
        """Finalize labels and count tokens processed after the documents.

        Args:
            answers: Per-stage mappings from document index to readout values.

        Returns:
            A tuple of suffix tokens and total streamed tokens, including frames.
            For pair classification, the first value counts classified pairs
            instead of suffix tokens.
        """
        if self.partners is not None:
            streamed = sum(1 + self.block_tokens[partner]
                           for _, partner in self.pair_labels)
            anchors = {anchor for anchor, _ in self.pair_labels}
            streamed += len(anchors) * len(self.stages[0].frame)
            return len(self.pair_labels), streamed
        first = answers[0]
        if self.decoder is not None:
            suffix_tokens = self.decoder.tokens
        else:
            for anchor, logprobs in first.items():
                if self.labels[anchor] is None:
                    self._choose(anchor, _softmax(self._scores(logprobs)), None)
            read = sum(map(len, self.request.suffixes)) + self.canvas_rows
            suffix_tokens = len(first) * read
            if self.draws > 1:
                suffix_tokens += self.extended * (self.draws - 1) * read
        return suffix_tokens, suffix_tokens + len(first) * len(self.request.frame)


class QuailClassifier:
    """Classify document rows with Quail's token admission and KV arena."""

    def __init__(self, state: QueryExecutionState):
        self.state = state

    def classify(self, spec, rows, documents) -> RerankerBatch:
        """Classify document rows using the shared scheduler and KV arena.

        Args:
            spec: Classification specification.
            rows: Document IDs to classify, in input order.
            documents: Tokenized documents indexed by table alias and document ID.

        Returns:
            A RerankerBatch containing labels, optional category probabilities,
            token counts, and execution metrics.
        """
        state = self.state
        rows = np.asarray(rows, dtype=np.int32).reshape(-1)
        prefixes = document_prefixes(spec, documents, rows)
        staging = input_staging(state)
        keys = [("classify", spec.name, index) for index in range(len(rows))]
        tree = None
        if spec.share_prefixes:
            started = time.perf_counter()
            tree = prefix_tree(prefixes, state.loaded_model.arena.page_tokens)
            logger.info(
                "prefix sharing on %s: %s documents borrow %s tokens "
                "(tree built in %.2f s)", spec.name, len(prefixes),
                tree.shared_tokens, time.perf_counter() - started)
        plan = ClassifyStages(state, spec, len(rows), lambda key: key[2])
        # a canvas is drawn from the document's row, so an answer is
        # the same in any batch
        plan.seeds = rows
        stats = {}
        answers, spans, fresh = run_stages(
            state.torch, state.loaded_model.arena, state.loaded_model.pipeline,
            plan.stages,
            prefixes, state.chunk_tokens, anchor_keys=keys,
            staging=staging, prefix_tree=tree, stats=stats,
            label=f"classify {spec.name}")
        suffix_tokens, streamed = plan.finish(answers)
        total = sum(map(len, prefixes)) + streamed
        return self._batch(plan.labels, fresh, total - fresh, suffix_tokens,
                           stats, spans, plan.probabilities)

    def _batch(self, labels, fresh, cached, suffix_tokens, stats,
               spans, probabilities=None) -> RerankerBatch:
        """Build a classification batch with token counts and optional GPU timing."""
        state = self.state
        gpu_s = 0.0
        if state.gpu_timing:
            # every chunk's answers were read, so its end event completed
            state.torch.cuda.synchronize()
            gpu_s = sum(start.elapsed_time(end)
                        for _, _, start, end in spans) / 1000.0
        return RerankerBatch(
            labels, fresh_tokens=fresh, cached_tokens=cached,
            suffix_tokens=suffix_tokens,
            borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0),
            gpu_s=gpu_s,
            chunks=len(spans) if state.gpu_timing else 0,
            probabilities=probabilities)
