"""Numeric reranker scores through Quail's forward loop."""

from quail.backends.quail.executor.attention import FILTER_ATTENTION
from quail.backends.quail.executor.loop import run_join
from quail.execution.reranker import RerankerBatch
from quail.execution.tokens import chain_tokens


class AsyncScores:
    """Copy normalized YES scores asynchronously after each forward pass."""

    def __init__(self, torch, answerer):
        self.torch = torch
        self.ans = answerer
        self.weights = answerer.weights.float()

    def submit(self, normed):
        torch, ans = self.torch, self.ans
        logits = ans.F.linear(normed.float(), self.weights)
        yes = logits.index_select(1, ans.true_cols).squeeze(1)
        no = logits.index_select(1, ans.false_cols).squeeze(1)
        scores = torch.sigmoid(yes - no)
        host = torch.empty(scores.shape[0], dtype=torch.float32, pin_memory=True)
        host.copy_(scores, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return event, host

    @staticmethod
    def result(handle):
        event, host = handle
        event.synchronize()
        return host.tolist()


class QuailScorer:
    """Score document rows with Quail's token admission and KV arena."""

    def __init__(self, state):
        self.state = state

    def score(self, spec, rows, documents):
        state = self.state
        parts = spec.prompt_token_parts
        if len(parts) != len(spec.aliases) + 1:
            raise ValueError("AI.SCORE needs tokenized prompt parts")
        if len(spec.aliases) == 1:
            prefixes = [parts[0]]
            suffixes = [chain_tokens(documents[spec.aliases[0]][row[0]], parts[1])
                        for row in rows]
            partners = None
            positions = [(0, index) for index in range(len(rows))]
        else:
            left, right = spec.aliases
            anchors = list(dict.fromkeys(row[0] for row in rows))
            candidates = list(dict.fromkeys(row[1] for row in rows))
            anchor_index = {value: index for index, value in enumerate(anchors)}
            candidate_index = {value: index
                               for index, value in enumerate(candidates)}
            prefixes = [chain_tokens(parts[0], documents[left][doc], parts[1])
                        for doc in anchors]
            suffixes = [chain_tokens(documents[right][doc], parts[2])
                        for doc in candidates]
            partners = [[] for _ in anchors]
            positions = []
            for a, b in rows:
                index = anchor_index[a]
                positions.append((index, len(partners[index])))
                partners[index].append(candidate_index[b])
        keys = [("score", spec.name, index) for index in range(len(prefixes))]
        async_scores = AsyncScores(state["torch"], state["async_answers"].ans)
        answers, _, fresh = run_join(
            state["torch"], state["arena"], state["pipeline"], async_scores,
            prefixes, [suffixes], state["chunk_tokens"], anchor_keys=keys,
            attention_mode=FILTER_ATTENTION,
            anchor_partners=(None if partners is None
                             else lambda key: [partners[key[2]]]),
        )
        total = sum(len(prefixes[a]) + len(suffixes[
            position if partners is None else partners[a][position]
        ]) for a, position in positions)
        return RerankerBatch(
            tuple(answers[0][a][position] for a, position in positions),
            fresh_tokens=fresh, cached_tokens=total - fresh,
        )
