"""Numeric reranker scores through Quail's forward loop."""

import numpy as np

from quail.backends.quail.executor.classify import QuailClassifier
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

    def __init__(self, state: QueryExecutionState):
        self.state = state

    def score(self, spec, rows, documents):
        if isinstance(spec, ClassifySpec):
            return QuailClassifier(self.state).classify(spec, rows, documents)
        state = self.state
        rows = np.asarray(rows, dtype=np.int32)
        parts = spec.prompt_token_parts
        if len(parts) != len(spec.aliases) + 1:
            raise ValueError("AI.SCORE needs tokenized prompt parts")
        if (len(spec.aliases) == 1 and spec.draws > 1
                and state.loaded_model.pipeline.canvas_ids):
            return self._drawn(spec, rows, documents)
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
        answers, _, fresh = run_join(
            state.torch, state.loaded_model.arena, state.loaded_model.pipeline,
            async_scores,
            prefixes, [suffixes], state.chunk_tokens, anchor_keys=keys,
            anchor_partners=(None if partners is None
                             else lambda key: [partners[key[2]]]),
            staging=staging,
        )
        scores = np.empty(len(rows), dtype=np.float32)
        for anchor, values in answers[0].items():
            scores[order[offsets[anchor]:offsets[anchor + 1]]] = values
        return RerankerBatch(scores, fresh_tokens=fresh, cached_tokens=total - fresh)

    def _scores(self) -> AsyncScores:
        """The score readout kept on the state, with the input staging cleared."""
        state = self.state
        answer_rows = state.answer_rows
        async_scores = state.async_scores
        if async_scores is None or async_scores.rows is not answer_rows:
            async_scores = AsyncScores(state.torch, answer_rows)
            state.async_scores = async_scores
        input_staging(state)
        return async_scores

    def _drawn(self, spec, rows, documents) -> RerankerBatch:
        """Score one table's rows on a canvas model, averaging noise draws.

        Each document's prompt head and document stay in KV with the
        tail but its last token written after them. The first draw is
        the cue and a canvas of random tokens drawn from the document's
        row; a document whose first score has binary entropy above
        CANVAS_ENTROPY_NATS sends ``spec.draws - 1`` more, each with
        its own canvas, and its score is the mean over the draws.
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
        _, _, fresh = run_stages(
            state.torch, state.loaded_model.arena, pipeline, stages, prefixes,
            state.chunk_tokens,
            anchor_keys=[("score", spec.name, index) for index in range(len(docs))],
            staging=state.loaded_model.input_staging, label=f"score {spec.name}")
        read = len(cue) + width
        total = (sum(len(head) + len(documents[alias][doc]) for doc in docs)
                 + len(docs) * (len(tail) - 1 + read) + extended[0] * more * read)
        return RerankerBatch(scores, fresh_tokens=fresh,
                             cached_tokens=total - fresh)
