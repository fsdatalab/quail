"""Numeric reranker scores through Quail's forward loop."""

import numpy as np

from quail.backends.quail.executor.loop import InputStaging, run_join
from quail.backends.quail.executor.readout import AsyncScores
from quail.execution.reranker import RerankerBatch
from quail.execution.tokens import chain_tokens


class QuailScorer:
    """Score document rows with Quail's token admission and KV arena."""

    def __init__(self, state):
        self.state = state

    def score(self, spec, rows, documents):
        state = self.state
        rows = np.asarray(rows, dtype=np.int32)
        parts = spec.prompt_token_parts
        if len(parts) != len(spec.aliases) + 1:
            raise ValueError("AI.SCORE needs tokenized prompt parts")
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
        answer_rows = state["answer_rows"]
        async_scores = state.get("async_scores")
        if async_scores is None or async_scores.rows is not answer_rows:
            async_scores = AsyncScores(state["torch"], answer_rows)
            state["async_scores"] = async_scores
        if "input_staging" not in state:
            state["input_staging"] = InputStaging(state["torch"])
        state["input_staging"].fixed_tokens.clear()
        answers, _, fresh = run_join(
            state["torch"], state["arena"], state["pipeline"], async_scores,
            prefixes, [suffixes], state["chunk_tokens"], anchor_keys=keys,
            anchor_partners=(None if partners is None
                             else lambda key: [partners[key[2]]]),
            staging=state["input_staging"],
        )
        scores = np.empty(len(rows), dtype=np.float32)
        for anchor, values in answers[0].items():
            scores[order[offsets[anchor]:offsets[anchor + 1]]] = values
        return RerankerBatch(scores, fresh_tokens=fresh, cached_tokens=total - fresh)
