"""Problem instances: workload + filters + outcomes + solver limits."""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .configs import DeviceConfig, ModelConfig


@dataclass(frozen=True)
class Instance:
    model: ModelConfig
    device: DeviceConfig
    d: tuple                      # realized document token lengths, len N
    p: tuple                      # filter prompt lengths, len n
    s: tuple                      # conditional selectivities s_1..s_n (s_n unused for work)
    delta: int = 1                # chunk quantum (C9); chunks end on multiples of delta
    max_new_tokens: Optional[int] = None   # Def 5.1(iv) configured cap, or None
    max_seqs: Optional[int] = None

    @property
    def N(self) -> int:
        return len(self.d)

    @property
    def n(self) -> int:
        return len(self.p)

    def check(self) -> None:
        for i, di in enumerate(self.d):
            for j, pj in enumerate(self.p):
                if di + pj > self.model.L_ctx:
                    raise ValueError(f"doc {i} + prompt {j+1} exceeds L_ctx (eq. 23)")

    def chunk_options(self, remaining: int) -> list:
        """Legal Delta values for a doc with `remaining` tokens left:
        multiples of delta, plus the final shorter remainder (sec. 8.1)."""
        opts = list(range(self.delta, remaining, self.delta))
        opts.append(remaining)  # finish exactly
        return opts


def sample_outcomes(inst: Instance, rng: np.random.Generator) -> np.ndarray:
    """One coupled latent outcome matrix X (N x n), Bernoulli(s_j) i.i.d.
    (paper sec. 10.3: the full latent matrix is stored so every policy uses
    the same scenario; length-independent by Assumption 2.2)."""
    return (rng.random((inst.N, inst.n)) < np.asarray(inst.s)).astype(np.int8)


def survival(X: np.ndarray) -> np.ndarray:
    """Y_ij indicator matrix (eq. 2): Y_i1 = 1, Y_ij = prod_{k<j} X_ik."""
    N, n = X.shape
    Y = np.ones((N, n), dtype=np.int8)
    for j in range(1, n):
        Y[:, j] = Y[:, j - 1] & X[:, j - 1]
    return Y
