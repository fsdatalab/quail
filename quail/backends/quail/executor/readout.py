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


class AsyncLabelLogprobs:
    """Non-blocking full-vocabulary log probabilities of chosen tokens.

    Each row gets log p(t) for every target token t, normalized over
    the whole vocabulary at temperature 1. Logits are computed in the
    head's dtype and normalized in float32, as vLLM computes returned
    log probabilities. Rows go through the head in blocks, so no rows
    by vocabulary matrix larger than one block is held.

    With rows > 1 an answer is a (rows, targets) record: the rows of
    one suffix in order, NaN past the suffix's own rows.
    """

    # rows per head block: every block reads the whole head, so few big
    # blocks; 512 x 151,936 logits in bf16 and float32 are 445 MiB
    BLOCK_ROWS = 512

    def __init__(self, torch, F, head, targets, rows=1):
        self.torch = torch
        self.F = F
        self.head = head
        self.targets = torch.tensor(list(targets), device=head.device,
                                    dtype=torch.long)
        self.rows = rows
        self.dtype = np.dtype((np.float32, (rows, len(targets)) if rows > 1
                               else (len(targets),)))
        self.available = []

    def logprobs(self, normed):
        """Per row, the target tokens' log probabilities, on the device."""
        torch = self.torch
        out = torch.empty((normed.shape[0], len(self.targets)),
                          dtype=torch.float32, device=normed.device)
        for start in range(0, normed.shape[0], self.BLOCK_ROWS):
            block = normed[start:start + self.BLOCK_ROWS]
            logits = self.F.linear(block.to(self.head.dtype), self.head).float()
            norm = torch.logsumexp(logits, dim=1, keepdim=True)
            out[start:start + block.shape[0]] = (
                logits.index_select(1, self.targets) - norm)
        return out

    def submit(self, normed, rows_per_answer=None):
        torch = self.torch
        values = self.logprobs(normed)
        if self.rows > 1:
            if rows_per_answer is None:
                rows_per_answer = [1] * values.shape[0]
            if max(rows_per_answer) > self.rows:
                raise ValueError(
                    f"a suffix has {max(rows_per_answer)} rows to read; "
                    f"the readout holds {self.rows}")
            answers = torch.full((len(rows_per_answer), self.rows,
                                  values.shape[1]), float("nan"),
                                 dtype=torch.float32, device=values.device)
            answer_index = torch.tensor(
                [i for i, n in enumerate(rows_per_answer) for _ in range(n)],
                device=values.device)
            row_index = torch.tensor(
                [r for n in rows_per_answer for r in range(n)],
                device=values.device)
            answers[answer_index, row_index] = values
            values = answers
        elif rows_per_answer is not None and any(
                n != 1 for n in rows_per_answer):
            raise ValueError("this readout holds one row per answer")
        host = self.available.pop() if self.available else None
        if host is None or host.shape[0] < values.shape[0]:
            host = torch.empty(values.shape, dtype=torch.float32,
                               pin_memory=True)
        host[:values.shape[0]].copy_(values, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return event, host, values.shape[0]

    def result(self, handle):
        event, host, count = handle
        event.synchronize()
        values = host[:count].numpy().copy()
        self.available.append(host)
        return values


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
