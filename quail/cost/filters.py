"""Costs of boolean filters over prepared prompt token counts."""

from quail.common.sol import unrounded_seconds
from quail.common.work import ask, packed, scan
from quail.specs import DeviceSpec, ModelSpec


def filter_cost(question_tokens: int, mean_tokens: float, model: ModelSpec,
                device: DeviceSpec, chunk_tokens: int, *, first: bool) -> float:
    """Return ideal time for one filter evaluation.

    A set-attention model reads the document again in every pass, so a
    later filter costs as much as the first.
    """
    if model.attention_mask == "set":
        work = packed(mean_tokens, (question_tokens,),
                      window=model.sliding_window)
    else:
        operation = scan if first else ask
        work = operation(mean_tokens, question_tokens,
                         window=model.sliding_window)
    return unrounded_seconds(work, model, device, chunk_tokens)
