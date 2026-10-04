"""Decision 2.0 readouts with MLX.

A decision model's own head scores the rows that end its options
against the prompt's last row. submit() starts the computation and
returns; result() waits for it and returns host values.
"""

import math

import numpy as np

LAYER_NORM_EPS = 1e-5


class MlxDecisionHead:
    """The option-scoring head of a Decision 2.0 checkpoint, in fp32.

    Each option's row and the prompt's last row pass through their own
    LayerNorm; an option's score is a bilinear term plus a GELU MLP
    term over the two.

    Args:
        weights: The tensors of decision_head.safetensors by name.
    """

    def __init__(self, weights):
        import mlx.core as mx

        self.w = {name: array.astype(mx.float32) for name, array in weights.items()}
        mx.eval(list(self.w.values()))
        self.head_dim = self.w["key.weight"].shape[0]

    def scores(self, options, last):
        """Per prompt, one score per option.

        Args:
            options: (prompts, options, hidden) rows that end each option.
            last: (prompts, hidden) last rows of the prompts.

        Returns:
            (prompts, options) float32 scores.
        """
        import mlx.core as mx
        import mlx.nn as nn

        w = self.w
        option = mx.fast.layer_norm(
            options.astype(mx.float32), w["candidate_norm.weight"],
            w["candidate_norm.bias"], LAYER_NORM_EPS)
        query = mx.fast.layer_norm(
            last.astype(mx.float32), w["query_norm.weight"],
            w["query_norm.bias"], LAYER_NORM_EPS)
        bilinear = ((option @ w["key.weight"].T)
                    * (query @ w["query.weight"].T)[:, None, :]).sum(-1)
        mlp = (nn.gelu(option @ w["candidate_mlp.weight"].T
                       + w["candidate_mlp.bias"]
                       + (query @ w["query_mlp.weight"].T)[:, None, :])
               @ w["scalar.weight"].T).squeeze(-1)
        return bilinear / math.sqrt(self.head_dim) + mlp


class MlxDecisionRows:
    """The rows a decision head reads from each answer's trailing rows.

    A stage reads the last ``trailing_rows`` rows of each suffix (its
    ``read_rows``); the head reads, from each answer's rows, the rows at
    ``offsets`` before its last row: one per option, then the last row.
    An answer with another row count, such as a frame entry's single
    row, reads its last row for every position; the stage discards it.

    Args:
        head: The model's MlxDecisionHead.
        offsets: Distances before the last row, options first, ending in 0.
    """

    def __init__(self, head, offsets):
        self.head = head
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.trailing_rows = int(self.offsets.max()) + 1

    def scores(self, normed, rows_per_answer=None):
        """Per answer, one float32 score per option."""
        import mlx.core as mx

        if isinstance(normed, list):
            normed = mx.stack(normed)
        if rows_per_answer is None:
            rows_per_answer = [self.trailing_rows] * (
                normed.shape[0] // self.trailing_rows)
        counts = np.asarray(rows_per_answer, dtype=np.int64)
        last = np.cumsum(counts) - 1
        full = counts == self.trailing_rows
        index = last[:, None] - np.where(full[:, None], self.offsets[None, :], 0)
        rows = normed[mx.array(index.reshape(-1), dtype=mx.int32)].reshape(
            len(counts), len(self.offsets), normed.shape[-1])
        return self.head.scores(rows[:, :-1], rows[:, -1])

    @staticmethod
    def _start(values):
        import mlx.core as mx

        mx.async_eval(values)
        return values


class MlxDecisions(MlxDecisionRows):
    """Yes/no readout of a decision head, for AI_FILTER and AI_JOIN.

    The options are No then Yes; the bit is 1 when Yes scores above No.
    """

    dtype = None    # answers arrive as Python lists of bits

    def submit(self, normed, rows_per_answer=None):
        import mlx.core as mx

        scores = self.scores(normed, rows_per_answer)
        return self._start((scores[:, 1] > scores[:, 0]).astype(mx.uint8))

    @staticmethod
    def result(handle):
        return [int(bit) for bit in handle.tolist()]


class MlxDecisionScores(MlxDecisionRows):
    """AI.SCORE readout of a decision head: P(Yes) over No and Yes."""

    dtype = np.float32
    rows = None    # scores no answer rows of the output head

    def submit(self, normed, rows_per_answer=None):
        import mlx.core as mx

        scores = self.scores(normed, rows_per_answer)
        return self._start(mx.sigmoid(scores[:, 1] - scores[:, 0]))

    @staticmethod
    def result(handle):
        return np.array(handle)


class MlxDecisionChoices(MlxDecisionRows):
    """AI.CLASSIFY readout of a decision head: one score per label."""

    def __init__(self, head, offsets):
        super().__init__(head, offsets)
        self.dtype = np.dtype((np.float32, (len(self.offsets) - 1,)))

    def submit(self, normed, rows_per_answer=None):
        return self._start(self.scores(normed, rows_per_answer))

    result = staticmethod(MlxDecisionScores.result)
