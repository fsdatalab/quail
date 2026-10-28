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


def expected_branches(curves) -> float:
    """The mean number of one document's answers written per pass.

    A document's answers start together; the document stays in passes
    while any is unfinished. Each pass t writes Σ S_m(t) answers in
    expectation, and the document is in pass t with chance
    1 - Π (1 - S_m(t)), taking answers as independent.

    Args:
        curves: One survival curve per answer of the document.
    """
    width = max(len(curve) for curve in curves)
    stacked = np.zeros((len(curves), width))
    for row, curve in zip(stacked, curves):
        row[:len(curve)] = curve
    passes = float((1.0 - np.prod(1.0 - stacked, axis=0)).sum())
    return float(stacked.sum()) / passes if passes else 0.0
