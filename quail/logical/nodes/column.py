"""The SemanticColumn base: one model call that adds columns to its input rows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical.expressions import (
    ColumnRef,
    CompileError,
    ModelCall,
    _explain,
)
from quail.logical.nodes.base import LogicalNode, _subtree


@dataclass(frozen=True)
class SemanticColumn:
    """One model call over its input rows, adding its answer columns.

    Every input row passes through with the call's answer in the column
    named ``name``, and in any further columns ``added_columns`` names.
    The root Project returns them through an Alias of the same call.

    A subclass names the call kind it takes, the SQL function, and how
    many tables the call may read.
    """

    input: LogicalNode
    call: ModelCall
    name: str

    type_name: ClassVar[str]
    call_kind: ClassVar[str]
    function: ClassVar[str]
    # the number of tables a call may read
    num_aliases: ClassVar[tuple[int, ...]] = (1,)

    @property
    def alias(self) -> str:
        """The table alias the answer columns belong to."""
        return self.call.aliases()[0]

    @property
    def added_columns(self) -> tuple[str, ...]:
        """The names of the columns the node adds, its answer first."""
        return (self.name,)

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (self.call,)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = self.input.output_schema()
        provider = next(
            (field.provider for field in fields if field.alias == self.alias), ""
        )
        return fields + tuple(
            ColumnRef(self.alias, provider, name) for name in self.added_columns
        )

    def validate_name(self) -> None:
        """Raise CompileError unless the answer column's name is allowed."""
        if not self.name or "." in self.name:
            raise CompileError(
                f"{self.function} needs a column name without a dot, "
                f"got {self.name!r}"
            )

    def validate(self) -> None:
        self.call.validate()
        if self.call.kind != self.call_kind:
            raise CompileError(
                f"{type(self).__name__} needs an {self.function} call"
            )
        self.validate_name()
        aliases = self.call.aliases()
        if len(aliases) not in self.num_aliases:
            raise CompileError(
                f"{self.function} reads one document"
                + (", or one from each side of a join"
                   if 2 in self.num_aliases else " column")
            )
        schema = self.input.output_schema()
        present = {field.alias for field in schema}
        missing = set(aliases) - present
        if missing:
            raise CompileError(
                f"{self.function} reads {sorted(missing)}, which its "
                f"input does not produce ({sorted(present)})"
            )
        taken = {field.column for field in schema if field.alias == self.alias}
        for name in self.added_columns:
            if name in taken:
                raise CompileError(f"the name {name!r} is already used")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError(f"{type(self).__name__} needs one input")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if len(expressions) != 1:
            raise CompileError(f"{type(self).__name__} needs one call")
        return replace(self, call=expressions[0])

    def explain_fields(self) -> dict:
        return {"name": self.name, "expression": _explain(self.call)}


def column_nodes(node, kind: type) -> tuple:
    """Return the nodes of one kind at or below a node, children first."""
    return tuple(found for found in _subtree(node) if isinstance(found, kind))
