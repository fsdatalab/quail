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


def answer_rows(rows_per_answer) -> tuple[np.ndarray, np.ndarray]:
    """Return each read row's answer and its row within that answer.

    Args:
        rows_per_answer: The rows each answer reads, in order.

    Returns:
        (answer index, row index), one entry per read row.
    """
    counts = np.asarray(rows_per_answer, dtype=np.int64)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    return (np.repeat(np.arange(len(counts)), counts),
            np.arange(int(counts.sum())) - starts)


class AsyncLabelLogprobs:
    """Asynchronous readout of selected tokens' log probabilities or logits.

    With normalize=True, each token's log probability is normalized over
    the full vocabulary at temperature 1. Computation uses blocks of rows
    to bound memory usage. With normalize=False, only the selected output
    head rows are evaluated and the results are unnormalized logits.

    Args:
        torch: Torch module.
        F: Torch functional module.
        head: Full output head with shape (vocabulary, hidden size).
        targets: Token IDs to return, in column order.
        rows: Maximum rows per answer. Unused rows are filled with NaN.
        normalize: Whether to normalize against the full vocabulary.
    """

    # rows per head block: every block reads the whole head, so few big
    # blocks; 512 x 151,936 logits in bf16 and float32 are 445 MiB
    BLOCK_ROWS = 512

    def __init__(self, torch, F, head, targets, rows=1, normalize=True):
        self.torch = torch
        self.F = F
        self.head = head
        self.targets = torch.tensor(list(targets), device=head.device,
                                    dtype=torch.long)
        self.rows = rows
        self.normalize = normalize
        self.head_rows = None if normalize else head.index_select(
            0, self.targets)
        self.dtype = np.dtype((np.float32, (rows, len(targets)) if rows > 1
                               else (len(targets),)))
        self.available = []

    def logprobs(self, normed):
        """Compute selected token values from normalized hidden states.

        Args:
            normed: Hidden states with shape (rows, hidden size).

        Returns:
            A float32 device tensor with shape (rows, targets). Values are
            full-vocabulary log probabilities when normalize is enabled and
            unnormalized logits otherwise.
        """
        torch = self.torch
        if not self.normalize:
            return self.F.linear(normed.to(self.head.dtype),
                                 self.head_rows).float()
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
            # copied from pinned memory, so the host does not wait for
            # the forward pass ahead of the copy on the stream
            answer_index, row_index = (
                torch.from_numpy(index).pin_memory().to(
                    values.device, non_blocking=True)
                for index in answer_rows(rows_per_answer))
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
