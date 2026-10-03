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


class AsyncDecisions:
    """Non-blocking yes/no readout of a decision head, for AI_FILTER and AI_JOIN.

    The forward pass returns three rows per answer: the row that ends
    the no option, the row that ends the yes option, and the prompt's
    last row. The bit is 1 when yes scores above no.
    """

    dtype = None    # answers arrive as Python lists of bits

    def __init__(self, torch, head):
        self.torch = torch
        self.head = head

    def submit(self, normed):
        torch = self.torch
        rows = normed.view(-1, 3, normed.shape[-1])
        scores = self.head.scores(rows[:, :2], rows[:, 2])
        bits = (scores[:, 1] > scores[:, 0]).to(torch.uint8)
        host = torch.empty(bits.shape[0], dtype=torch.uint8, pin_memory=True)
        host.copy_(bits, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return event, host

    result = staticmethod(AsyncAnswers.result)


class AsyncDecisionScores(AsyncScores):
    """Non-blocking AI.SCORE readout of a decision head: P(yes) over No and Yes.

    Takes the same three rows per answer as AsyncDecisions.
    """

    def __init__(self, torch, head):
        self.torch = torch
        self.head = head
        self.rows = None
        self.available = []

    def submit(self, normed):
        rows = normed.view(-1, 3, normed.shape[-1])
        scores = self.head.scores(rows[:, :2], rows[:, 2])
        return self._copy(self.torch.sigmoid(scores[:, 1] - scores[:, 0]))
