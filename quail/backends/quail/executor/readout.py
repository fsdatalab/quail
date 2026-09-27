"""Readouts: what an operator takes from the final hidden states.

The forward pass returns the final-normed hidden state of each answer
row. A readout scores it against the retained answer rows of the
output head and turns the scores into what the operator needs: a
TRUE/FALSE bit for AI_FILTER and AI_JOIN, a yes-against-no sigmoid
for AI.SCORE. Each readout copies its result off the GPU without
stalling the stream; result() waits on that copy.
"""

import numpy as np

from quail.backends.quail.executor.model import answer_weights
from quail.logical import true_false_ids


class AnswerRows:
    """The retained output rows for one query's two answer classes.

    Args:
        torch: The torch module.
        F: torch.nn.functional.
        model: The loaded model holding the retained answer rows.
        true_ids: Token ids of the first class (TRUE, or yes).
        false_ids: Token ids of the second class (FALSE, or no).
    """

    def __init__(self, torch, F, model, true_ids, false_ids):
        self.F = F
        self.allowed = sorted(set(true_ids) | set(false_ids))
        self.weights = answer_weights(model, self.allowed)
        self.true_cols = torch.tensor(
            [i for i, t in enumerate(self.allowed) if t in set(true_ids)],
            device="cuda")
        self.false_cols = torch.tensor(
            [i for i, t in enumerate(self.allowed) if t in set(false_ids)],
            device="cuda")

    @classmethod
    def from_tokenizer(cls, torch, F, model, tokenizer):
        """Rows for the TRUE and FALSE spellings the tokenizer produces."""
        true_ids, false_ids = true_false_ids(tokenizer)
        return cls(torch, F, model, true_ids, false_ids)

    def logits(self, normed):
        """Per row, the best logit of each class over its spellings."""
        scores = self.F.linear(normed, self.weights)
        return (scores.index_select(1, self.true_cols).amax(dim=1),
                scores.index_select(1, self.false_cols).amax(dim=1))

    def __call__(self, normed):
        """TRUE/FALSE bits read synchronously, for tests and probes."""
        t, f = self.logits(normed)
        return (t > f).int().cpu().tolist()


class AsyncAnswers:
    """Non-blocking TRUE/FALSE readout for AI_FILTER and AI_JOIN.

    submit() returns an event and pinned host buffer; result() waits on
    the event and reads the answers without stalling the GPU stream.
    """

    dtype = None    # answers arrive as Python lists of bits

    def __init__(self, torch, rows):
        self.torch = torch
        self.rows = rows

    def submit(self, normed):
        torch = self.torch
        t, f = self.rows.logits(normed)
        bits = (t > f).to(torch.uint8)
        host = torch.empty(bits.shape[0], dtype=torch.uint8,
                           pin_memory=True)
        host.copy_(bits, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return event, host

    @staticmethod
    def result(handle):
        event, host = handle
        event.synchronize()
        return [int(b) for b in host.tolist()]


class AsyncScores:
    """Non-blocking yes-against-no sigmoid readout for AI.SCORE."""

    dtype = np.float32

    def __init__(self, torch, rows):
        self.torch = torch
        self.rows = rows
        self.weights = rows.weights.float()
        self.available = []

    def submit(self, normed):
        torch, rows = self.torch, self.rows
        logits = rows.F.linear(normed.float(), self.weights)
        yes = logits.index_select(1, rows.true_cols).amax(dim=1)
        no = logits.index_select(1, rows.false_cols).amax(dim=1)
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
