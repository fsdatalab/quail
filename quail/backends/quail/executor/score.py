"""Numeric reranker scores through Quail's forward loop."""

import numpy as np

from quail.backends.quail.executor.classify import (
    AnswerStream,
    QuailClassifier,
    shared_prefix_tree,
)
from quail.backends.quail.executor.loop import run_join
from quail.backends.quail.executor.parts import input_staging
from quail.backends.quail.executor.readout import AsyncScores
from quail.backends.quail.executor.stages import Stage, run_stages
from quail.backends.quail.executor.state import QueryExecutionState
from quail.execution.reranker import RerankerBatch
from quail.execution.tokens import DocumentPrefixes, chain_tokens
from quail.physical import ClassifySpec
from quail.specs.base import CANVAS_ENTROPY_NATS, CANVAS_SEED


class QuailScorer:
    """Score document rows with Quail's token admission and KV arena."""

    # score() reports answers through on_answers while it runs
    streams_answers = True

    def __init__(self, state: QueryExecutionState):
        self.state = state

    def score(self, spec, rows, documents, on_answers=None):
        """Score or classify the rows.

        Args:
            spec: Score or classification specification.
            rows: Document indices, one column per alias.
            documents: Tokenized documents indexed by table alias and document ID.
            on_answers: Optional callable(row positions, values) run as each
                chunk's answers are read.

        Returns:
            A RerankerBatch with one value per row, in row order.
        """
        if isinstance(spec, ClassifySpec):
            return QuailClassifier(self.state).classify(
                spec, rows, documents, on_answers=on_answers)
        state = self.state
        rows = np.asarray(rows, dtype=np.int32)
        parts = spec.prompt_token_parts
        if len(parts) != len(spec.aliases) + 1:
            raise ValueError("AI.SCORE needs tokenized prompt parts")
        if (len(spec.aliases) == 1 and spec.draws > 1
                and state.loaded_model.pipeline.canvas_ids):
            return self._drawn(spec, rows, documents, on_answers)
        if len(spec.aliases) == 1 and spec.share_prefixes:
            return self._shared(spec, rows, documents, on_answers)
        if len(spec.aliases) == 1:
            prefixes = [parts[0]]
            suffixes = [chain_tokens(documents[spec.aliases[0]][doc], parts[1])
                        for doc in rows[:, 0]]
            partners = None
            order = np.arange(len(rows))
            offsets = np.array([0, len(rows)])
            total = len(rows) * len(parts[0]) + sum(map(len, suffixes))
        else:
            left, right = spec.aliases
            anchors, anchor_index = np.unique(rows[:, 0], return_inverse=True)
            candidates, candidate_index = np.unique(rows[:, 1], return_inverse=True)
            prefixes = [chain_tokens(parts[0], documents[left][doc], parts[1])
                        for doc in anchors]
            suffixes = [chain_tokens(documents[right][doc], parts[2])
                        for doc in candidates]
            order = np.argsort(anchor_index, kind="stable")
            offsets = np.concatenate(([0], np.cumsum(np.bincount(anchor_index))))
            partners = [candidate_index[order[start:end]]
                        for start, end in zip(offsets[:-1], offsets[1:])]
            prefix_lengths = np.asarray([len(prefix) for prefix in prefixes])
            suffix_lengths = np.asarray([len(suffix) for suffix in suffixes])
            total = int(prefix_lengths[anchor_index].sum()
                        + suffix_lengths[candidate_index].sum())
        keys = [("score", spec.name, index) for index in range(len(prefixes))]
        async_scores = self._scores()
        staging = state.loaded_model.input_staging

        def report(anchor, stage, start, end, values):
            first = offsets[anchor]
            on_answers(order[first + start:first + end], np.asarray(values))

        answers, _, fresh = run_join(
            state.torch, state.loaded_model.arena, state.loaded_model.pipeline,
            async_scores,
            prefixes, [suffixes], state.chunk_tokens, anchor_keys=keys,
            anchor_partners=(None if partners is None
                             else lambda key: [partners[key[2]]]),
            staging=staging,
            on_answers=None if on_answers is None else report,
        )
        scores = np.empty(len(rows), dtype=np.float32)
        for anchor, values in answers[0].items():
            scores[order[offsets[anchor]:offsets[anchor + 1]]] = values
        return RerankerBatch(scores, fresh_tokens=fresh, cached_tokens=total - fresh)

    def _scores(self) -> AsyncScores:
        """Return the cached score readout and clear its input staging cache."""
        state = self.state
        if state.score_readout is not None:
            input_staging(state)
            return state.score_readout
        answer_rows = state.answer_rows
        async_scores = state.async_scores
        if async_scores is None or async_scores.rows is not answer_rows:
            async_scores = AsyncScores(state.torch, answer_rows)
            state.async_scores = async_scores
        input_staging(state)
        return async_scores

    def _shared(self, spec, rows, documents, on_answers=None) -> RerankerBatch:
        """Score documents that borrow the KV pages of shared prefixes.

        Each document's prompt head and tokens are its prefix; a
        document whose prefix shares whole pages with another's reads
        those pages instead of computing them. The prompt tail follows
        each document, its last token read for the score.

        Args:
            spec: Single-document score specification.
            rows: Input row indices with shape (documents, 1).
            documents: Tokenized documents indexed by table alias and document ID.
            on_answers: Optional callable(row positions, values) run with each
                chunk's finished scores.

        Returns:
            A RerankerBatch of scores and execution metrics.

        Raises:
            ValueError: The prompt has no tokens after the document.
        """
        state = self.state
        (alias,) = spec.aliases
        head, tail = spec.prompt_token_parts
        if not tail:
            raise ValueError("AI.SCORE needs a question after the document")
        docs = rows[:, 0]
        scores = np.full(len(docs), np.nan, dtype=np.float32)

        def keep(anchor, row):
            scores[anchor] = float(row[0])
            return True

        prefixes = DocumentPrefixes(head, documents[alias], docs)
        stages = [Stage(suffixes=[[tail[-1]]], readout=self._scores(),
                        frame=list(tail[:-1]), decide=keep, single=True,
                        label=f"score {spec.name}")]
        stats = {}
        stream = AnswerStream(on_answers)
        _, _, fresh = run_stages(
            state.torch, state.loaded_model.arena, state.loaded_model.pipeline,
            stages, prefixes, state.chunk_tokens,
            anchor_keys=[("score", spec.name, index) for index in range(len(docs))],
            staging=state.loaded_model.input_staging,
            prefix_tree=shared_prefix_tree(
                spec.name, prefixes, state.loaded_model.arena),
            stats=stats, label=f"score {spec.name}",
            on_chunk=stream.chunk(lambda anchor: (
                None if np.isnan(scores[anchor]) else scores[anchor])))
        stream.finish(range(len(docs)), lambda anchor: scores[anchor])
        total = sum(len(head) + len(documents[alias][doc]) + len(tail)
                    for doc in docs)
        return RerankerBatch(scores, fresh_tokens=fresh,
                             cached_tokens=total - fresh,
                             borrowed_tokens=stats.get("borrowed_tokens", 0))

    def _drawn(self, spec, rows, documents, on_answers=None) -> RerankerBatch:
        """Score documents with reproducible diffusion draws.

        A low-entropy first answer uses one draw. Other documents run the remaining
        draws and use the mean TRUE probability. Draws share each document's KV.

        Args:
            spec: Single-document score specification.
            rows: Input row indices with shape (documents, 1).
            documents: Tokenized documents indexed by table alias and document ID.
            on_answers: Optional callable(row positions, values) run with each
                chunk's finished scores.

        Returns:
            A RerankerBatch of scores and execution metrics.
        """
        state = self.state
        pipeline = state.loaded_model.pipeline
        vocab = state.loaded_model.model_spec.vocab
        (alias,) = spec.aliases
        head, tail = spec.prompt_token_parts
        docs = rows[:, 0]
        width = len(pipeline.canvas_ids)
        more = spec.draws - 1
        scores = np.full(len(docs), np.nan, dtype=np.float32)
        first = np.zeros(len(docs), dtype=np.float64)
        extended = [0]

        def canvas(anchor, draw):
            rng = np.random.default_rng((CANVAS_SEED, int(docs[anchor]), draw))
            return rng.integers(0, vocab, width)

        def settle(anchor, row):
            p = float(row[0])
            first[anchor] = p
            q = min(max(p, 1e-12), 1 - 1e-12)
            if -(q * np.log(q) + (1 - q) * np.log(1 - q)) <= CANVAS_ENTROPY_NATS:
                scores[anchor] = p
            return True

        def average(anchor, row):
            extended[0] += 1
            scores[anchor] = (first[anchor] + float(np.sum(row))) / spec.draws
            return True

        cue = [tail[-1]]
        readout = self._scores()
        stages = [
            Stage(suffixes=[cue], readout=readout, frame=list(tail[:-1]),
                  decide=settle, single=True, label=f"score {spec.name}",
                  canvas=lambda anchor: canvas(anchor, 0), canvas_rows=width),
            Stage(suffixes=[cue] * more, readout=readout,
                  frame=list(tail[:-1]), decide=average,
                  requests=lambda key: (None if np.isnan(scores[key[2]])
                                        else Stage.SKIP),
                  label=f"score {spec.name} draws",
                  canvas=lambda anchor: np.stack(
                      [canvas(anchor, draw) for draw in range(1, spec.draws)]),
                  canvas_rows=width),
        ]
        prefixes = DocumentPrefixes(head, documents[alias], docs)
        tree = (shared_prefix_tree(spec.name, prefixes, state.loaded_model.arena)
                if spec.share_prefixes else None)
        stats = {}
        stream = AnswerStream(on_answers)
        _, _, fresh = run_stages(
            state.torch, state.loaded_model.arena, pipeline, stages, prefixes,
            state.chunk_tokens, prefix_tree=tree, stats=stats,
            anchor_keys=[("score", spec.name, index) for index in range(len(docs))],
            staging=state.loaded_model.input_staging, label=f"score {spec.name}",
            on_chunk=stream.chunk(lambda anchor: (
                None if np.isnan(scores[anchor]) else scores[anchor])))
        stream.finish(range(len(docs)), lambda anchor: scores[anchor])
        read = len(cue) + width
        total = (sum(len(head) + len(documents[alias][doc]) for doc in docs)
                 + len(docs) * (len(tail) - 1 + read) + extended[0] * more * read)
        return RerankerBatch(scores, fresh_tokens=fresh,
                             cached_tokens=total - fresh,
                             borrowed_tokens=stats.get("borrowed_tokens", 0))
