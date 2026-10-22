"""Comparison counts and seconds for a pairwise top-k ranking."""

import math

from quail.cost.budgets import chunk_budget
from quail.cost.score import score_work
from quail.cost.sol import speed_of_light
from quail.specs import DeviceSpec, ModelSpec

# A probability this close to 0.5 is asked again with the documents
# swapped, and the two answers are averaged.
RANK_MARGIN = 0.15
RANK_SEED = 0
NUM_PIVOTS = 1


def expected_comparisons(num_rows, k, num_pivots=1) -> float:
    """Return the comparisons a top-k quickselect is expected to ask.

    Published bounds for quickselect are about 2n to 3.4n comparisons.
    This uses their midpoint, 2.7n, once per pivot, and adds about
    k log2 k comparisons to order the rows that are kept.

    Args:
        num_rows: Rows compared.
        k: Rows kept.
        num_pivots: Pivots drawn in each round.

    Returns:
        The expected number of model comparisons.
    """
    rows = max(0.0, float(num_rows))
    kept = max(0.0, min(float(k), rows))
    if rows <= 1:
        return 0.0
    ordering = 0.0 if kept <= 1 else kept * math.log2(kept)
    return 2.7 * rows * num_pivots + ordering


def kept_rows(comparison: str, bound: int) -> int:
    """Return how many rows a rank filter keeps.

    Args:
        comparison: ``<=`` or ``<``.
        bound: The integer the rank is compared with.

    Returns:
        The rows whose rank passes the filter.

    Raises:
        ValueError: The comparison is not ``<`` or ``<=``.
    """
    if comparison == "<=":
        return bound
    if comparison == "<":
        return max(0, bound - 1)
    raise ValueError(f"AI.RANK filters with < or <=, got {comparison!r}")


def rank_seconds(num_rows, k, token_lengths, mean_tokens, variance,
                 model: ModelSpec, device: DeviceSpec, *,
                 num_pivots=1) -> float:
    """Return the estimated seconds of the comparisons.

    Each comparison is priced like a pair score. The pivot document
    and the text around it are the prefix, reused across the other
    document of that round. The number of rounds is about log2 of the
    rows.

    Args:
        num_rows: Rows compared.
        k: Rows kept.
        token_lengths: Token counts of the three prompt pieces.
        mean_tokens: Mean document length.
        variance: Variance of the document length.
        model: The model that answers the comparisons.
        device: The GPU the model runs on.
        num_pivots: Pivots drawn in each round.

    Returns:
        Estimated model seconds, excluding startup.
    """
    comparisons = expected_comparisons(num_rows, k, num_pivots)
    if comparisons <= 0 or len(token_lengths) < 3:
        return 0.0
    head, middle, tail = token_lengths[:3]
    prefix = head + mean_tokens + middle
    groups = max(1.0, math.log2(max(float(num_rows), 2.0)))
    work = score_work(
        comparisons, mean_tokens, head + middle + tail,
        variance=variance, prefix_tokens=prefix, groups=groups)
    chunk_tokens = chunk_budget(model, device)
    if chunk_tokens < 1:
        return 0.0
    return speed_of_light(work, model, device, chunk_tokens).seconds
