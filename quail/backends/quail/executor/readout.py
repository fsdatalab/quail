"""Readouts: what an operator takes from the final hidden states.

The forward pass returns the final-normed hidden state of each answer
row. A readout scores it against the retained answer rows of the
output head and turns the scores into what the operator needs: a
TRUE/FALSE bit for AI_FILTER and AI_JOIN, a yes-against-no sigmoid
for AI.SCORE. A decision model has no answer rows; its own head
scores the rows that end its no and yes options against the prompt's
last row. Each readout copies its result off the GPU without
stalling the stream; result() waits on that copy.
"""

import math
from pathlib import Path

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
        return self._copy(torch.sigmoid(yes - no))

    def _copy(self, scores):
        """Copy scores to a pinned host buffer behind an event."""
        torch = self.torch
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


class DecisionHead:
    """The option-scoring head of a Decision 2.0 checkpoint, in fp32.

    Each option's row and the prompt's last row pass through their own
    LayerNorm; an option's score is a bilinear term plus a GELU MLP
    term over the two.

    Args:
        torch: The torch module.
        F: torch.nn.functional.
        weights: The tensors of decision_head.safetensors by name.
    """

    def __init__(self, torch, F, weights):
        self.torch = torch
        self.F = F
        self.w = {name: t.float() for name, t in weights.items()}
        self.head_dim = self.w["key.weight"].shape[0]

    @classmethod
    def load(cls, torch, F, path, device="cuda"):
        """Load head/decision_head.safetensors from a converted checkpoint."""
        from safetensors.torch import load_file

        return cls(torch, F, load_file(
            str(Path(path) / "head" / "decision_head.safetensors"),
            device=device))

    def scores(self, options, last):
        """Per prompt, one score per option.

        Args:
            options: (prompts, options, hidden) rows that end each option.
            last: (prompts, hidden) last rows of the prompts.

        Returns:
            (prompts, options) float32 scores.
        """
        F, w = self.F, self.w
        hidden = options.shape[-1]
        option = F.layer_norm(options.float(), (hidden,),
                              w["candidate_norm.weight"], w["candidate_norm.bias"])
        query = F.layer_norm(last.float(), (hidden,),
                             w["query_norm.weight"], w["query_norm.bias"])
        bilinear = (F.linear(option, w["key.weight"])
                    * F.linear(query, w["query.weight"])[:, None, :]).sum(-1)
        mlp = F.linear(
            F.gelu(F.linear(option, w["candidate_mlp.weight"],
                            w["candidate_mlp.bias"])
                   + F.linear(query, w["query_mlp.weight"])[:, None, :]),
            w["scalar.weight"]).squeeze(-1)
        return bilinear / math.sqrt(self.head_dim) + mlp


class DecisionRows:
    """The rows a decision head reads from each answer's trailing rows.

    A stage reads the last ``trailing_rows`` rows of each suffix (its
    ``read_rows``); the head reads, from each answer's rows, the rows at
    ``offsets`` before its last row: one per option, then the last row.
    An answer with another row count, such as a frame entry's single
    row, reads its last row for every position; the stage discards it.

    Args:
        torch: The torch module.
        head: The model's DecisionHead.
        offsets: Distances before the last row, options first, ending in 0.
    """

    def __init__(self, torch, head, offsets):
        self.torch = torch
        self.head = head
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.trailing_rows = int(self.offsets.max()) + 1

    def scores(self, normed, rows_per_answer=None):
        """Per answer, one float32 score per option."""
        if rows_per_answer is None:
            rows_per_answer = [self.trailing_rows] * (
                normed.shape[0] // self.trailing_rows)
        counts = np.asarray(rows_per_answer, dtype=np.int64)
        last = np.cumsum(counts) - 1
        full = counts == self.trailing_rows
        index = last[:, None] - np.where(full[:, None], self.offsets[None, :], 0)
        index = self.torch.from_numpy(index.reshape(-1)).to(
            normed.device, non_blocking=True)
        rows = normed.index_select(0, index).view(
            len(counts), len(self.offsets), normed.shape[-1])
        return self.head.scores(rows[:, :-1], rows[:, -1])

    def _copy(self, values, dtype):
        torch = self.torch
        host = torch.empty(values.shape, dtype=dtype, pin_memory=True)
        host.copy_(values, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return event, host


class AsyncDecisions(DecisionRows):
    """Non-blocking yes/no readout of a decision head, for AI_FILTER and AI_JOIN.

    The options are No then Yes; the bit is 1 when Yes scores above No.
    """

    dtype = None    # answers arrive as Python lists of bits

    def submit(self, normed, rows_per_answer=None):
        scores = self.scores(normed, rows_per_answer)
        return self._copy((scores[:, 1] > scores[:, 0]).to(self.torch.uint8),
                          self.torch.uint8)

    result = staticmethod(AsyncAnswers.result)


class AsyncDecisionScores(DecisionRows):
    """Non-blocking AI.SCORE readout of a decision head: P(Yes) over No and Yes."""

    dtype = np.float32
    rows = None    # scores no answer rows of the output head

    def submit(self, normed, rows_per_answer=None):
        scores = self.scores(normed, rows_per_answer)
        return self._copy(self.torch.sigmoid(scores[:, 1] - scores[:, 0]),
                          self.torch.float32)

    @staticmethod
    def result(handle):
        event, host = handle
        event.synchronize()
        return host.numpy()


class AsyncDecisionChoices(DecisionRows):
    """Non-blocking AI.CLASSIFY readout of a decision head: one score per label."""

    def __init__(self, torch, head, offsets):
        super().__init__(torch, head, offsets)
        self.dtype = np.dtype((np.float32, (len(self.offsets) - 1,)))

    def submit(self, normed, rows_per_answer=None):
        return self._copy(self.scores(normed, rows_per_answer), self.torch.float32)

    result = staticmethod(AsyncDecisionScores.result)
