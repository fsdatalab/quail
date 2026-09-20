"""Rules for model-only columns, shared by the SQL compiler and the builder.

A PDF provider's ``document`` column holds page references. The
model reads it; a query cannot return it, compare it, or hand it to
an apply() function, and only AI.FILTER over one table may read it:
a join would put a partner's pages after the anchor, an AI.SCORE
reranker takes text.
"""

from __future__ import annotations

from collections.abc import Iterable

from quail.catalog import Catalog, model_only_columns
from quail.logical import ColumnRef, CompileError


def value_columns(catalog: Catalog, provider: str) -> tuple[str, ...]:
    """The columns ``*`` expands to: every column a query may return."""
    source = catalog.get(provider)
    hidden = model_only_columns(source)
    return tuple(name for name in source.columns if name not in hidden)


def check_value_column(catalog: Catalog, ref: ColumnRef, use: str) -> None:
    """Fail when a model-only column is used where a value is needed.

    Args:
        catalog: The session catalog.
        ref: The resolved column.
        use: Where it appears, for the message: "SELECT", "a join
            condition", "apply()".
    """
    if ref.column in model_only_columns(catalog.get(ref.provider)):
        raise CompileError(
            f"column {ref.column!r} of {ref.alias!r} holds page images the "
            f"model reads; it cannot appear in {use}. It can only be the "
            f"document argument of AI.FILTER over {ref.alias!r}")


def check_prompt_columns(catalog: Catalog, refs: Iterable[ColumnRef],
                         function: str, join: bool) -> None:
    """Fail unless every model-only prompt column is AI.FILTER's document.

    Args:
        catalog: The session catalog.
        refs: The prompt's column arguments.
        function: The AI function name as the SQL compiler spells it,
            "AI_FILTER", "AI_SCORE", or "AI_JOIN".
        join: Whether the prompt spans more than one table.
    """
    for ref in refs:
        if ref.column not in model_only_columns(catalog.get(ref.provider)):
            continue
        if function != "AI_FILTER":
            raise CompileError(
                f"{function.replace('_', '.')} cannot read {ref.column!r} of "
                f"{ref.alias!r}: page images go through AI.FILTER only")
        if join:
            raise CompileError(
                f"AI.JOIN over PDF rows is not supported; {ref.alias!r} binds "
                f"page images and can only be filtered on its own")
