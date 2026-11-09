"""How long written answers run: survival curves over answer tokens.

S(t) is the chance an answer is still being written after t tokens,
so S(0) = 1 and S(limit) = 0. Before any answer has ended the curve is
a prior: answer lengths geometric with the planner's expected length,
cut off at the answer limit. The curves are computed by Rust's cost
model (``crates/quail-planner/src/cost/answers.rs``), which the planner prices
extraction with.
"""

import _quail_native
import numpy as np


def _array(found: bytes, dtype) -> np.ndarray:
    """A writable array over the little-endian bytes Rust returned."""
    return np.frombuffer(bytearray(found), dtype)


def _int64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def _float64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def prior_survival(expected_tokens: float, limit: int) -> np.ndarray:
    """S(t) for t = 0..limit under geometric lengths with the given mean.

    Args:
        expected_tokens: The mean answer length without the limit.
        limit: The most tokens an answer may have.
    """
    return _array(
        _quail_native.prior_survival(float(expected_tokens), int(limit)), np.float64)


def survival_rows(counts, events, prior, limits) -> np.ndarray:
    """Return Kaplan-Meier curves, one per row.

    Ended answers are events, including answers cut off at the limit.
    Answers still being written are censored at the tokens they have.

    Args:
        counts: Answers observed last at each token, ended or censored,
            one row per curve.
        events: Answers that ended at each token, one row per curve.
        prior: The prior curve of each row, 0 past the row's limit.
        limits: Each row's answer token limit.

    Returns:
        S(t) per row, 0 from each row's limit on.
    """
    counts = _int64(counts)
    found = _quail_native.survival_rows(
        counts, _int64(events), _float64(prior), _int64(limits), counts.shape[1])
    return _array(found, np.float64).reshape(counts.shape)


def answer_survival(lengths, written, expected_tokens, limit) -> np.ndarray:
    """Return S(t), the chance an answer runs past t tokens, t = 0..limit.

    Args:
        lengths: Each answer's length, -1 while it is being written.
        written: The tokens each answer has written.
        expected_tokens: The prior's mean answer length.
        limit: The answer token limit.
    """
    return _array(_quail_native.answer_survival(
        _int64(lengths), _int64(written), float(expected_tokens), int(limit)),
        np.float64)


def remaining_tokens(lengths, written, expected_tokens, limit, *,
                     survival=None) -> np.ndarray:
    """Return E[L - t | L > t], the expected tokens left after t, t = 0..limit.

    With S(u) = P(L > u) and S(limit) = 0, this is the sum of S(u) / S(t)
    over u = t..limit - 1.
    """
    if survival is None:
        survival = answer_survival(lengths, written, expected_tokens, limit)
    return _array(_quail_native.remaining_tokens(_float64(survival)), np.float64)
