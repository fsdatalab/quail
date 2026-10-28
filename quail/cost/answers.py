"""How long written answers run: survival curves over answer tokens.

S(t) is the chance an answer is still being written after t tokens,
so S(0) = 1 and S(limit) = 0. Before any answer has ended the curve is
a prior: answer lengths geometric with the planner's expected length,
cut off at the answer limit.
"""

import numpy as np


def prior_survival(expected_tokens: float, limit: int) -> np.ndarray:
    """S(t) for t = 0..limit under geometric lengths with the given mean.

    Args:
        expected_tokens: The mean answer length without the limit.
        limit: The most tokens an answer may have.
    """
    keep = 1.0 - 1.0 / max(1.0, float(expected_tokens))
    curve = keep ** np.arange(limit + 1, dtype=np.float64)
    curve[limit] = 0.0
    return curve
