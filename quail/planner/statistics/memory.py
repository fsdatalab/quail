"""How a query splits the GPU memory left after the weights."""

from quail.cost import budgets
from quail.planner.statistics.tables import summarize_alias
from quail.specs import DeviceSpec, ModelSpec


def pool_split(model: ModelSpec, device: DeviceSpec, chunk: int,
               summaries) -> tuple[int, int]:
    """Return the query's KV pool split: (every-token pages, sliding pages).

    The worker builds the arena to this split, and every operator's
    cost reads its capacity from it. The pools are split for the
    longest table's mean document length, before filters thin the
    tables.

    Args:
        model: Model spec.
        device: Device spec.
        chunk: Chunk token budget.
        summaries: The AliasStats of every table the query reads.
    """
    longest = max((summary.mean for summary in summaries), default=None)
    return tuple(budgets.arena_pages(model, device, chunk, longest))


def arena_capacity(context, chunk: int) -> int:
    """Return the tokens of document KV the query's arena holds at once."""
    window = context.model.sliding_window
    full, _ = pool_split(context.model, context.device, chunk, (
        summarize_alias(lengths, window, memo=context.memo)
        for lengths in context.document_tokens.values()))
    return full * budgets.PAGE_TOKENS
