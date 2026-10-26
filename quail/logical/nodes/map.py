"""The SemanticMap node and the map helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical.expressions import (
    ColumnRef,
    CompileError,
    ModelCall,
    _explain,
)
from quail.logical.nodes.base import LogicalNode, _subtree, with_answer_columns
from quail.logical.nodes.join import _above_joins


@dataclass(frozen=True)
class SemanticMap:
    """One AI.MAP instruction over its input rows, adding the answer column.

    Every input row passes through with the answer in the column named
    ``name``: the text the model writes for the instruction over the
    row's document. The node starts on its table above the table's
    AI.IF SemanticFilter, classifications, label filters, and
    extractions, and below any join; the should_lift_maps rule moves it
    above the joins of a joined table, so the map runs over the
    documents the joins matched. The root Project returns the column
    through an Alias of the same call.
    """

    input: LogicalNode
    call: ModelCall
    name: str

    type_name: ClassVar[str] = "quail.semantic_map"

    @property
    def alias(self) -> str:
        """The table alias the answer column belongs to."""
        return self.call.aliases()[0]

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (self.call,)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return with_answer_columns(self, self.name)

    def validate(self) -> None:
        self.call.validate()
        if self.call.kind != "map":
            raise CompileError("SemanticMap needs an AI.MAP call")
        aliases = self.call.aliases()
        if len(aliases) != 1:
            raise CompileError("AI.MAP reads one document column")
        present = {field.alias for field in self.input.output_schema()}
        if self.alias not in present:
            raise CompileError(
                f"the map reads {self.alias!r}, which its input "
                f"does not produce ({sorted(present)})"
            )
        taken = {
            field.column
            for field in self.input.output_schema()
            if field.alias == self.alias
        }
        if self.name in taken:
            raise CompileError(f"the name {self.name!r} is already used")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("SemanticMap needs one input")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if len(expressions) != 1:
            raise CompileError("SemanticMap needs one call")
        return replace(self, call=expressions[0])

    def explain_fields(self) -> dict:
        return {
            "name": self.name,
            "expression": _explain(self.call),
            "instruction": self.call.instruction,
        }


def semantic_maps(node) -> tuple:
    """Return the SemanticMap nodes at or below a node, children first."""
    return tuple(found for found in _subtree(node) if isinstance(found, SemanticMap))


def mapped_above_joins(root) -> frozenset:
    """Return aliases whose map runs after a join."""
    return _above_joins(semantic_maps(root))
