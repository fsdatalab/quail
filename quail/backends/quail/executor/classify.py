"""AI.CLASSIFY labels through Quail's join loop.

Each document is an anchor: its prompt head and document stay in KV,
and the classification tail, all but its last token, is written once
after them as the anchor's frame. The label scoring rule decides what
the partner suffixes are:

- ``letters``: the categories are lettered, so every label is one
  token and the one suffix is the tail's last token, whose row scores
  them all. On a diffusion model the suffix packs a seeded canvas
  after the cue: a random token at the reply's first row, the turn
  close, and padding; one read-only denoising step, and the first
  row's log probabilities score the letters.
- ``trie_tree``: the whole label trie as one suffix: the cue and every
  node's token once, split into chains that each follow first
  children. Each chain is a causal segment and reads the ancestors
  above it from earlier chains, so every node is computed once and
  every row is read; a label's score is the sum of its tokens' log
  probabilities.
- ``trie_decode``: one stage per trie depth. Each round a document
  sends the chain of the node it has decoded so far and appends the
  likeliest child token read after it, until the node is a whole
  label (GreedyDecoder); the label is the greedy path, and the
  document skips the rounds left.
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
    GreedyDecoder,
    best_label,
    label_trie,
    letter_scores,
    tree_scores,
    trie_chains,
)
from quail.execution.reranker import RerankerBatch
from quail.execution.tokens import chain_tokens, prefix_tree
from quail.specs.base import CANVAS_SEED

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
    if spec.scoring == "letters":
        return LabelRequests(
            frame, [[cue]], targets, True,
            lambda logprobs: letter_scores(ids, targets, logprobs[0, 0]))
    if spec.scoring == "trie_decode":
        nodes = sorted(label_trie(ids), key=lambda prefix: (len(prefix), prefix))
        return LabelRequests(
            frame, [[cue, *node] for node in nodes], targets, False, None,
            rounds=max(len(label) for label in ids))
    raise ValueError(f"unknown label scoring rule {spec.scoring!r}")


def document_prefixes(spec, documents, rows) -> list:
    """Return each row's prompt head and document as one token sequence."""
    head, _ = spec.prompt_token_parts
    (alias,) = spec.aliases
    return [chain_tokens(head, documents[alias][doc]) for doc in rows]


@dataclass
class JoinPartners:
    """The partner documents of a join and each anchor's kept partners.

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


def _label_readout(state, targets, rows: int, normalize: bool):
    """The label readout kept on the state, rebuilt when its shape changes."""
    readout = state.get("label_readout")
    if (readout is None or list(readout.targets.tolist()) != targets
            or readout.rows != rows
            or getattr(readout, "normalize", normalize) != normalize):
        torch = state["torch"]
        head = full_output_head(state["model"])
        readout = AsyncLabelLogprobs(
            torch, torch.nn.functional, head, targets, rows=rows,
            normalize=normalize)
        state["label_readout"] = readout
        logger.info("label readout: head %s x %s in %s, %s targets, "
                    "%s rows per request, %s", *head.shape,
                    str(head.dtype).replace("torch.", ""), len(targets),
                    rows, "normalized" if normalize else "targets' logits")
    return readout


class ClassifyStages:
    """The stages one classification asks of each document, and its labels.

    One stage per classification of the chain, or one per trie depth
    under ``trie_decode``; a later classification's first stage gates
    on the label before it. The stages
    read the label readout kept on the state, rebuilt when the targets
    or rows change. A classification of joined rows has one stage: it
    sends each anchor, for every partner the join kept with it, the
    partner's label and document, the question, and the cue; its
    labels are by (anchor, partner).

    Args:
        state: The executor state.
        spec: The classification and its chained stages.
        count: How many documents the stages run over.
        index_of: Callable(admission key) -> the document's index.
        on_label: Optional callable(index, label) run when a document's
            first classification labels it.
        partners: The join's JoinPartners, for a classification of
            joined rows; None otherwise.
    """

    canvas_rows = 0         # rows of a letters stage's canvas on a canvas model
    seeds = None            # document index -> its canvas seed; the index itself
    partners = None         # JoinPartners, for joined rows
    pair_labels = None      # (anchor index, partner index) -> label, for joined rows

    def __init__(self, state, spec, count, index_of, on_label=None,
                 partners=None):
        self.spec = spec
        self.specs = specs = spec.chain
        self.index_of = index_of
        if spec.partner is not None:
            self._joined_stages(state, spec, count, partners)
            return
        targets = sorted({token for stage in specs
                          for ids in stage.label_token_ids for token in ids})
        self.requests = requests = [label_requests(stage, targets)
                                    for stage in specs]
        read_all = requests[0].read_all_rows
        if any(request.read_all_rows != read_all for request in requests):
            raise ValueError("a classify chain's stages read rows one way")
        self.read_all = read_all
        self.readout_rows = readout_rows = (
            max(len(suffix) for request in requests
                for suffix in request.suffixes) if read_all else 1)
        # a letters stage on a canvas model reads its canvas's rows
        answer_canvas = getattr(state.get("model_spec"), "answer_canvas", None)
        if spec.scoring == "letters" and answer_canvas is not None:
            self.canvas_rows = answer_canvas.rows
            self.readout_rows = readout_rows = self.canvas_rows
        # every label read at the same row needs no normalizer: one-token
        # labels at the cue row
        same_rows = all(len(ids) == 1 for stage in specs
                        for ids in stage.label_token_ids)
        self.readout = readout = _label_readout(state, targets, readout_rows,
                                                not same_rows)
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
                    chains=request.chains,
                    **(self._letter_canvas(state, index)
                       if self.canvas_rows else {})))
                continue
            decoder = GreedyDecoder(stage_spec.label_token_ids, targets, count)
            self.decoders[index] = decoder
            for round in range(request.rounds):
                def ask(key, index=index, round=round, decoder=decoder):
                    if round == 0 and index and self.gate(index, key) is Stage.DROP:
                        return Stage.DROP
                    nodes = decoder.requests(self.index_of(key))
                    # a resolved document has nothing left to read
                    # and goes on to whatever follows the rounds
                    return Stage.SKIP if nodes is None else nodes

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

    def _joined_stages(self, state, spec, count, partners):
        """One stage over the anchors: each partner's block, the question, the cue."""
        if spec.stages:
            raise ValueError("a classification of joined rows runs alone")
        if spec.scoring != "letters":
            raise ValueError("a classification of joined rows reads letters, "
                             f"not {spec.scoring!r}")
        if partners is None:
            raise ValueError(
                "a classification of joined rows needs the join's partners")
        note, partner_label = spec.join_layout
        targets = sorted({token for ids in spec.label_token_ids for token in ids})
        request = label_requests(spec, targets)
        self.requests = [request]
        self.readout_rows = 1
        self.readout = readout = _label_readout(state, targets, 1, False)
        self.read_all = True
        self.partners = partners
        self.pair_labels = {}
        self.labels = [np.full(count, None, dtype=object)]
        self.decoders = {}
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

    def _letter_canvas(self, state, index) -> dict:
        """The Stage fields of a letters stage on a canvas model.

        The stage's one suffix, the cue, packs a canvas after it as one
        entry: a random token at the reply's first row, the turn close,
        and padding. The random token is drawn from the document's row
        and the stage's index, so an answer is the same in any batch.
        """
        model_spec = state["model_spec"]
        settings = model_spec.answer_canvas
        template = np.full(settings.rows, settings.pad_id, dtype=np.int64)
        template[1] = settings.turn_close_id

        def canvas(anchor):
            rng = np.random.default_rng((CANVAS_SEED, self.seed(anchor), index))
            ids = template.copy()
            ids[0] = rng.integers(0, model_spec.vocab)
            return ids

        return dict(single=True, canvas=canvas, canvas_rows=settings.rows)

    def seed(self, anchor) -> int:
        """The document's canvas seed: its index unless ``seeds`` says."""
        return int(anchor if self.seeds is None else self.seeds[anchor])

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

    def finish(self, answers) -> tuple[int, int, int]:
        """Label every document from its answers; returns the token counts.

        Returns:
            (suffix tokens, label tokens, streamed tokens): the suffix
            tokens read; the positions after the answer cue that each
            request needs once, which are a trie's nodes below the cue,
            a canvas's rows, or a decoded label's path without its last
            token; and the suffix tokens plus the frames written after
            the documents.
        """
        if self.partners is not None:
            streamed = sum(1 + self.block_tokens[partner]
                           for _, partner in self.pair_labels)
            anchors = {anchor for anchor, _ in self.pair_labels}
            streamed += len(anchors) * len(self.stages[0].frame)
            return len(self.pair_labels), 0, streamed
        suffix_tokens = 0
        label_tokens = 0
        streamed = 0
        position = 0
        for index, request in enumerate(self.requests):
            first = answers[position]
            if index in self.decoders:
                decoder = self.decoders[index]
                stage_tokens = decoder.tokens
                label_tokens += sum(len(decoder.label_ids[label]) - 1
                                    for label in decoder.label if label >= 0)
            else:
                for anchor, logprobs in first.items():
                    if self.labels[index][anchor] is None:
                        self.labels[index][anchor] = self.label_of(
                            index, logprobs)
                # a letters stage on a canvas model streams its canvas too
                stage_tokens = len(first) * (sum(map(len, request.suffixes))
                                             + self.canvas_rows)
                # every suffix starts with the cue
                label_tokens += stage_tokens - len(first) * len(request.suffixes)
            suffix_tokens += stage_tokens
            # a frame equal to the stage before's is already in KV
            written = (index == 0
                       or request.frame != self.requests[index - 1].frame)
            streamed += stage_tokens + (len(first) * len(request.frame)
                                        if written else 0)
            position += max(1, request.rounds)
        return suffix_tokens, label_tokens, streamed

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
        plan = ClassifyStages(state, spec, len(rows), lambda key: key[2])
        # a canvas is drawn from the document's row, so an answer is
        # the same in any batch
        plan.seeds = rows
        stats = {}
        answers, spans, fresh = run_stages(
            state["torch"], state["arena"], state["pipeline"], plan.stages,
            prefixes, state["chunk_tokens"], anchor_keys=keys,
            staging=state["input_staging"], prefix_tree=tree, stats=stats,
            label=f"classify {spec.name}")
        suffix_tokens, label_tokens, streamed = plan.finish(answers)
        total = sum(map(len, prefixes)) + streamed
        return self._batch(plan.labels[0], fresh, total - fresh,
                           (suffix_tokens, label_tokens), stats, spans,
                           plan.later())

    def _batch(self, labels, fresh, cached, counts, stats, spans,
               later) -> RerankerBatch:
        """The labels and token counts of one run, with its GPU time when timed."""
        state = self.state
        gpu_s = 0.0
        if state.get("gpu_timing"):
            # every chunk's answers were read, so its end event completed
            state["torch"].cuda.synchronize()
            gpu_s = sum(start.elapsed_time(end)
                        for _, _, start, end in spans) / 1000.0
        return RerankerBatch(
            labels, fresh_tokens=fresh, cached_tokens=cached,
            suffix_tokens=counts[0], label_tokens=counts[1],
            borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0),
            gpu_s=gpu_s,
            chunks=len(spans) if state.get("gpu_timing") else 0,
            later=later)
