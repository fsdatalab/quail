"""Costs of boolean filters over prepared prompt token counts."""

from quail.cost.sol import unrounded_seconds
from quail.cost.work import Work, ask, scan
from quail.specs import DeviceSpec, ModelSpec


def filter_cost(question_tokens: int, prefix_tokens: float, model: ModelSpec,
                device: DeviceSpec, chunk_tokens: int, *, first: bool) -> float:
    """Return ideal time for one filter evaluation."""
    operation = scan if first else ask
    work = operation(prefix_tokens,
                     question_tokens,
                     window=model.sliding_window)
    return unrounded_seconds(work, model, device, chunk_tokens)


def filter_chain_work(questions, selectivities, prefix_tokens: float,
                      count: float, window: int = 0) -> Work:
    """Return expected work for predicates in their chosen execution order."""
    total = Work()
    for index, (question, selectivity) in enumerate(zip(questions, selectivities)):
        operation = scan if index == 0 else ask
        total = total + operation(prefix_tokens, question, window=window) * count
        count *= selectivity
    return total
