"""The LogicalNode protocol and the subtree walk over it."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from quail.logical.expressions import ColumnRef


class LogicalNode(Protocol):
    """Node in a logical query plan."""

    type_name: ClassVar[str]

    def children(self) -> tuple["LogicalNode", ...]: ...

    def expressions(self) -> tuple[Any, ...]: ...

    def output_schema(self) -> tuple[ColumnRef, ...]: ...

    def validate(self) -> None: ...

    def with_children(self, children: tuple["LogicalNode", ...]) -> "LogicalNode": ...

    def with_expressions(self, expressions: tuple[Any, ...]) -> "LogicalNode": ...

    def explain_fields(self) -> dict[str, Any]: ...


def _subtree(node) -> tuple:
    """Return the node and every node below it, children first."""
    nodes = []
    for child in node.children():
        nodes.extend(_subtree(child))
    nodes.append(node)
    return tuple(nodes)


def with_answer_columns(node, *names: str) -> tuple[ColumnRef, ...]:
    """Return the node's input columns plus answer columns of its table.

    Args:
        node: A one-table operator with `input` and `alias`.
        names: The answer column names the node adds.

    Returns:
        The input schema followed by one ColumnRef per name.
    """
    fields = node.input.output_schema()
    provider = next(
        (field.provider for field in fields if field.alias == node.alias), "")
    return fields + tuple(ColumnRef(node.alias, provider, name) for name in names)
