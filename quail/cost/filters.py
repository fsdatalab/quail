"""Costs of boolean filters over prepared prompt token counts."""

from quail.cost.sol import unrounded_seconds
from quail.cost.work import ask, scan
from quail.specs import DeviceSpec, ModelSpec


def filter_cost(question_tokens: int, mean_tokens: float, model: ModelSpec,
                device: DeviceSpec, chunk_tokens: int, *, first: bool) -> float:
    """Return ideal time for one filter evaluation."""
    operation = scan if first else ask
    work = operation(mean_tokens,
                     question_tokens,
                     window=model.sliding_window)
    return unrounded_seconds(work, model, device, chunk_tokens)
