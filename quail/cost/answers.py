"""How long written answers run: survival curves over answer tokens.

S(t) is the chance an answer is still being written after t tokens,
so S(0) = 1 and S(limit) = 0. Before any answer has ended the curve is
a prior: answer lengths geometric with the planner's expected length,
cut off at the answer limit.
"""

import numpy as np

# answers the prior counts as, until ended answers outweigh it
PRIOR_WEIGHT = 20.0


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


def survival_rows(counts, events, prior, limits):
    """Return Kaplan-Meier curves, one per row.

    Ended answers are events, including answers cut off at the limit.
    Answers still being written are censored at the tokens they have.
    The prior adds PRIOR_WEIGHT answers' worth of events and risk, so a
    curve starts from it and moves to the answers as they end.

    Args:
        counts: Answers observed last at each token, ended or censored,
            one row per curve.
        events: Answers that ended at each token, one row per curve.
        prior: The prior curve of each row, 0 past the row's limit.
        limits: Each row's answer token limit.

    Returns:
        S(t) per row, 0 from each row's limit on.
    """
    risk = (np.cumsum(counts[:, ::-1], axis=1)[:, ::-1][:, 1:]
            + PRIOR_WEIGHT * prior[:, :-1])
    events = events[:, 1:] + PRIOR_WEIGHT * (prior[:, :-1] - prior[:, 1:])
    hazard = np.divide(events, risk, out=np.zeros(risk.shape), where=risk > 0)
    survival = np.empty(counts.shape)
    survival[:, 0] = 1.0
    np.cumprod(1.0 - hazard, axis=1, out=survival[:, 1:])
    survival[np.arange(counts.shape[1]) >= np.asarray(limits)[:, None]] = 0.0
    return survival


def answer_survival(lengths, written, expected_tokens, limit, *, prior=None):
    """Return S(t), the chance an answer runs past t tokens, t = 0..limit.

    Args:
        lengths: Each answer's length, -1 while it is being written.
        written: The tokens each answer has written.
        expected_tokens: The prior's mean answer length.
        limit: The answer token limit.
        prior: The curve to start from; the geometric prior by default.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    observed = np.where(lengths >= 0, lengths, np.asarray(written, dtype=np.int64))
    counts = np.bincount(observed, minlength=limit + 1)[None, :limit + 1]
    events = np.bincount(lengths[lengths >= 0], minlength=limit + 1)[None, :limit + 1]
    if prior is None:
        prior = prior_survival(expected_tokens, limit)
    return survival_rows(counts, events, np.asarray(prior)[None], [limit])[0]


def remaining_tokens(lengths, written, expected_tokens, limit, *, survival=None):
    """Return E[L - t | L > t], the expected tokens left after t, t = 0..limit.

    With S(u) = P(L > u) and S(limit) = 0, this is the sum of S(u) / S(t)
    over u = t..limit - 1.
    """
    if survival is None:
        survival = answer_survival(lengths, written, expected_tokens, limit)
    tail = np.cumsum(survival[::-1])[::-1]
    return np.divide(tail, survival, out=np.zeros(limit + 1), where=survival > 0)
