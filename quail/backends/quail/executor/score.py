"""Numeric reranker scores through Quail's forward loop."""

import numpy as np

from quail.backends.quail.executor.attention import FILTER_ATTENTION
from quail.backends.quail.executor.loop import InputStaging, run_join
from quail.execution.reranker import RerankerBatch
from quail.execution.tokens import chain_tokens


class AsyncScores:
    """Copy normalized YES scores asynchronously after each forward pass."""

    def __init__(self, torch, answerer):
        self.torch = torch
        self.ans = answerer
        self.weights = answerer.weights.float()
        self.available = []

    def submit(self, normed):
        torch, ans = self.torch, self.ans
        logits = ans.F.linear(normed.float(), self.weights)
        yes = logits.index_select(1, ans.true_cols).squeeze(1)
        no = logits.index_select(1, ans.false_cols).squeeze(1)
        scores = torch.sigmoid(yes - no)
        host = self.available.pop() if self.available else None
        if host is None or host.numel() < scores.shape[0]:
            host = torch.empty(scores.shape[0], dtype=torch.float32, pin_memory=True)
        host[:scores.shape[0]].copy_(scores, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return event, host, scores.shape[0]

    def result(self, handle):
        event, host, count = handle
        event.synchronize()
        self.available.append(host)
        return host[:count].numpy()


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
        answerer = state["async_answers"].ans
        async_scores = state.get("async_scores")
        if async_scores is None or async_scores.ans is not answerer:
            async_scores = AsyncScores(state["torch"], answerer)
            state["async_scores"] = async_scores
        if "input_staging" not in state:
            state["input_staging"] = InputStaging(state["torch"])
        state["input_staging"].fixed_tokens.clear()
        answers, _, fresh = run_join(
            state["torch"], state["arena"], state["pipeline"], async_scores,
            prefixes, [suffixes], state["chunk_tokens"], anchor_keys=keys,
            attention_mode=FILTER_ATTENTION,
            anchor_partners=(None if partners is None
                             else lambda key: [partners[key[2]]]),
            answer_dtype=np.float32, staging=state["input_staging"],
        )
        scores = np.empty(len(rows), dtype=np.float32)
        for anchor, values in answers[0].items():
            scores[order[offsets[anchor]:offsets[anchor + 1]]] = values
        return RerankerBatch(scores, fresh_tokens=fresh, cached_tokens=total - fresh)
