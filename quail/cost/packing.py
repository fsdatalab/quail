"""Stages of a set-attention filter chain: which questions share a pass.

Under set attention a document is read again in every pass, so a
chain of filters on one table can put several questions in one pass.
A later question in a pass is asked of rows that an earlier filter of
the pass removes; a separate stage reads only the rows the stages
before it kept. choose_stages prices each split with packed() and
keeps the cheapest.
"""

from math import inf

from quail.cost.sol import unrounded_seconds
from quail.cost.work import packed
from quail.specs import DeviceSpec, ModelSpec


def choose_stages(question_tokens, selectivities, rows: float,
                  state_tokens: float, longest_state: int, model: ModelSpec,
                  device: DeviceSpec, chunk_tokens: int, *,
                  set_size: int) -> tuple[int, ...]:
    """Split an ordered chain of questions into consecutive stages.

    best(b) is the least cost of the first b questions: the minimum
    over a of best(a - 1) plus the cost of one stage holding questions
    a to b, asked of every row that reaches question a.

    Args:
        question_tokens: Each question's tokens, in chain order.
        selectivities: Each question's expected fraction of rows kept.
        rows: The rows that reach the first question.
        state_tokens: The mean tokens of a pass's state.
        longest_state: The longest state, which bounds a pass's length.
        model: The set-attention model.
        device: The device it runs on.
        chunk_tokens: The chunk budget, which a pass must fit.
        set_size: The most questions in one pass.

    Returns:
        Each question's stage, numbered from 0 in chain order.

    Raises:
        ValueError: A question with the longest state does not fit one
            chunk.
    """
    m = len(question_tokens)
    reach = [float(rows)]
    for selectivity in selectivities:
        reach.append(reach[-1] * selectivity)
    best, start = [0.0] + [inf] * m, [0] * (m + 1)
    for b in range(1, m + 1):
        for a in range(b, 0, -1):
            group = tuple(question_tokens[a - 1:b])
            if (len(group) > set_size
                    or longest_state + sum(group) > chunk_tokens):
                break
            work = packed(state_tokens, group, window=model.sliding_window)
            cost = best[a - 1] + unrounded_seconds(
                work * reach[a - 1], model, device, chunk_tokens)
            if cost < best[b]:
                best[b], start[b] = cost, a
        if best[b] == inf:
            raise ValueError(
                f"question {b} does not fit one {chunk_tokens}-token chunk "
                f"with the longest document")
    stages, b = [0] * m, m
    bounds = []
    while b:
        bounds.append((start[b], b))
        b = start[b] - 1
    for index, (a, b) in enumerate(reversed(bounds)):
        for position in range(a - 1, b):
            stages[position] = index
    return tuple(stages)
