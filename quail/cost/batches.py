"""Dependent steps priced with ideal packing within each step."""

import math
from dataclasses import replace

from quail.cost.dense_decoder_cost import readout_work
from quail.cost.work import Work


def batch_work(steps, live: float, samples: int, chunk: int, model,
               head_rows: int = 0) -> Work:
    """Pack expected step work without iterating over documents or batches.

    Each step supplies sample work, readout rows, and the probability that
    one document reaches it. Full batches use fractional ideal packing;
    a nonempty step still needs one weight read.
    """
    batches = []
    for work, rows, probability in steps:
        own = work * (live / samples)
        nonempty = (min(1.0, live) if probability >= 1 else
                    -math.expm1(live * math.log1p(-probability)))
        passes = max(own.tokens / chunk, nonempty)
        if passes == 0:
            continue
        own += readout_work(model, rows * live / samples, passes, head_rows)
        batches.append((own, passes))
    return replace(sum((work for work, _ in batches), Work()), batches=tuple(batches))


def cached_profile(memo, key, sources, build):
    """Reuse step totals while retaining the source columns used by the key."""
    if memo is None:
        return build()
    if key not in memo:
        memo[key] = (sources, build())
    return memo[key][1]
