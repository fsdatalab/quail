"""Counted work of the cost model's request shapes, as a test oracle.

Tests build expected work from these and compare it with what Rust's
cost model (``crates/quail-planner/src/cost/work.rs``) prices.
"""

from dataclasses import replace

import numpy as np

from quail.common.work import Work, triangle


def scan(prefix: float, suffix: float, *, window: int = 0) -> Work:
    """Compute one document and its first suffix from nothing."""
    n = prefix + suffix
    return Work(tokens=n, pairs=triangle(n), kv_written=n,
                sliding_pairs=triangle(n, window) if window else 0.0,
                state_written=1.0 + 0.0 * np.asarray(n, dtype=float))


def ask(prefix: float, suffix: float, *, window: int = 0) -> Work:
    """Attach one suffix to a prefix already available in KV."""
    return Work(
        tokens=suffix,
        pairs=suffix * prefix + triangle(suffix),
        kv_written=suffix,
        kv_read=prefix,
        sliding_pairs=(triangle(prefix + suffix, window) - triangle(prefix, window)
                       if window else 0.0),
        sliding_kv_read=np.minimum(prefix, window - 1) if window else 0.0,
        state_read=1.0 + 0.0 * np.asarray(prefix, dtype=float),
    )


def stream(prefix: float, suffixes, *, window: int = 0) -> Work:
    """Attach several independent suffixes to one prefix in KV."""
    tokens = pairs = sliding_pairs = count = 0.0
    for suffix in suffixes:
        count += 1
        tokens += suffix
        pairs += suffix * prefix + triangle(suffix)
        if window:
            sliding_pairs += (triangle(prefix + suffix, window)
                              - triangle(prefix, window))
    return Work(
        tokens=tokens,
        pairs=pairs,
        kv_written=tokens,
        kv_read=prefix,
        sliding_pairs=sliding_pairs,
        sliding_kv_read=np.minimum(prefix, window - 1) if window else 0.0,
        state_read=count + 0.0 * np.asarray(prefix, dtype=float),
    )


def packed(state, questions, *, window: int = 0) -> Work:
    """Work of one set-attention pass: a state, then its questions, nothing kept.

    On the full-attention layers the state reads every token and each
    question reads the state and itself, so a lone question reads the
    whole sequence. The other layers read causally through the sequence.
    The state's keys are read once for all questions, then once more
    for each question's own tokens.

    Args:
        state: State tokens, or an array of them per document.
        questions: Each question's tokens, its header included.
        window: Attention window of the sliding layers, or 0.
    """
    questions = tuple(questions)
    asked = float(sum(questions))
    n = state + asked
    pairs = state * n + sum(q * (state + q) for q in questions)
    read = n if len(questions) == 1 else n + state + asked
    return Work(tokens=n, pairs=pairs, kv_written=n, kv_read=read,
                sliding_pairs=triangle(n, window) if window else 0.0)


def decode(prefix, rounds, *, window: int = 0) -> Work:
    """Append accepted tokens over sequential rounds, rereading KV each round.

    Args:
        prefix: Context length, or an array of document context lengths.
        rounds: Number of tokens fed in each fully accepted round.
        window: Attention window, or 0 for full attention.

    Returns:
        Token, attention, and KV work over the rounds.
    """
    work = Work()
    for tokens in rounds:
        step = ask(prefix, tokens, window=window)
        work += replace(step, state_written=step.state_read)
        prefix = prefix + tokens
    if np.ndim(prefix):
        # Scalar suffix lengths apply once to each document in the array.
        work = Work(work.tokens * np.ones_like(prefix), work.pairs,
                    work.kv_written * np.ones_like(prefix), work.kv_read,
                    work.sliding_pairs, work.sliding_kv_read,
                    work.state_written, work.state_read)
    return work
