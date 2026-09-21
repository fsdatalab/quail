"""Rules for model-only columns, shared by the SQL compiler and the builder.

A PDF provider's ``document`` column holds page references. The
model reads it rendered; a query cannot return it, compare it, or
hand it to an apply() function. AI.FILTER may read it, and so may a
join predicate over two tables when it is the only PDF column in the
prompt: the planner anchors that join on the PDF alias, since the
runtime renders the anchor's pages and not a partner's. An AI.SCORE
reranker takes text. The OCR operator over a PDF provider has a plain
string ``document`` column, so none of this applies to it.
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
            f"column {ref.column!r} of {ref.alias!r} holds the PDF pages the "
            f"model reads; it cannot appear in {use}. It can only be a "
            f"document argument of an AI.FILTER or AI.JOIN prompt")


def check_prompt_columns(catalog: Catalog, refs: Iterable[ColumnRef],
                         function: str, join: bool) -> None:
    """Fail unless a model-only prompt column is read the way pages allow.

    A filter reads one table's pages. A join reads the pages of one of
    its tables, the one the planner will anchor on; two PDF tables in
    one prompt would need a partner's pages rendered, which the
    runtime does not do.

    Args:
        catalog: The session catalog.
        refs: The prompt's column arguments.
        function: The AI function name as the SQL compiler spells it,
            "AI_FILTER" (a filter or a join predicate) or "AI_SCORE".
        join: Whether the prompt spans more than one table.
    """
    pdf_aliases = []
    for ref in refs:
        if ref.column not in model_only_columns(catalog.get(ref.provider)):
            continue
        if function != "AI_FILTER":
            raise CompileError(
                f"{function.replace('_', '.')} cannot read {ref.column!r} of "
                f"{ref.alias!r}: PDF pages go through AI.FILTER, as a "
                f"filter or as a join predicate")
        if ref.alias not in pdf_aliases:
            pdf_aliases.append(ref.alias)
    if join and len(pdf_aliases) > 1:
        raise CompileError(
            f"a join reads the pages of one table; {pdf_aliases} all bind "
            f"PDF pages. Give one side as text, or join them in turn")
