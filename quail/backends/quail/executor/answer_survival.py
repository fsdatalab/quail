"""Answer lengths estimated from this run's events and censored answers."""

import numpy as np

from quail.cost.answers import prior_survival


def answer_survival(lengths, written, expected_tokens, limit, prior_weight=20.0):
    """Return S(t), the chance an answer runs past t tokens, t = 0..limit.

    A Kaplan-Meier curve: finished answers are events, including answers
    cut off at the limit; unfinished answers are censored at their last
    token read by the host. The geometric prior contributes fractional
    events and risk counts, so the curve starts from the planner's guess.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    observed = np.where(lengths >= 0, lengths, np.asarray(written, dtype=np.int64))
    counts = np.bincount(observed, minlength=limit + 1)
    events = np.bincount(lengths[lengths >= 0], minlength=limit + 1)
    prior = prior_survival(expected_tokens, limit)
    risk = np.cumsum(counts[::-1])[::-1][1:] + prior_weight * prior[:-1]
    events = events[1:] + prior_weight * (prior[:-1] - prior[1:])
    hazard = np.divide(events, risk, out=np.zeros(limit), where=risk > 0)
    survival = np.r_[1.0, np.cumprod(1.0 - hazard)]
    survival[limit] = 0.0
    return survival


def remaining_tokens(lengths, written, expected_tokens, limit, prior_weight=20.0):
    """Return expected remaining lengths at each written length, 0 through limit."""
    survival = answer_survival(lengths, written, expected_tokens, limit,
                               prior_weight)
    tail = np.r_[np.cumsum(survival[:0:-1])[::-1], 0.0]
    return np.divide(tail, survival, out=np.zeros(limit + 1), where=survival > 0)


def quantile_tokens(lengths, written, expected_tokens, limit, quantile=0.9) -> int:
    """Return the answer length that this share of answers stay within."""
    survival = answer_survival(lengths, written, expected_tokens, limit)
    return int(np.argmax(survival <= 1.0 - quantile))
