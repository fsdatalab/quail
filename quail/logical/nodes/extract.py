"""The SemanticExtract node and the extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical.expressions import (
    SPAN_SUFFIX,
    ColumnRef,
    CompileError,
    ModelCall,
    _explain,
)
from quail.logical.nodes.base import LogicalNode, _subtree
from quail.logical.nodes.join import _above_joins


@dataclass(frozen=True)
class SemanticExtract:
    """One AI.EXTRACT field over its input rows, adding the answer columns.

    Every input row passes through with the answer in the column named
    ``name``, which is ``struct.field`` for the ExtractStruct field it
    fills, and its character span in the document, as a (start, end)
    pair, in the column ``name`` plus SPAN_SUFFIX. A row whose document
    does not answer the question gets nulls. The node starts on its
    table above the table's AI.IF SemanticFilter, classifications, and
    label filters, and below any join; the lift_maps rule moves it
    above the joins of a joined table, so the extraction runs
    over the documents the joins matched. The root Project returns the
    columns through an Alias of the same call.
    """

    input: LogicalNode
    call: ModelCall
    name: str

    type_name: ClassVar[str] = "quail.semantic_extract"

    @property
    def alias(self) -> str:
        """The table alias the answer columns belong to."""
        return self.call.aliases()[0]

    @property
    def span_name(self) -> str:
        """The name of the span column."""
        return self.name + SPAN_SUFFIX

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (self.call,)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = self.input.output_schema()
        provider = next(
            (field.provider for field in fields if field.alias == self.alias), ""
        )
        return fields + (
            ColumnRef(self.alias, provider, self.name),
            ColumnRef(self.alias, provider, self.span_name),
        )

    def validate(self) -> None:
        self.call.validate()
        if self.call.kind != "extract":
            raise CompileError("SemanticExtract needs an AI.EXTRACT call")
        if self.name.count(".") != 1 or not all(self.name.split(".")):
            raise CompileError(
                f"an extraction is named after its struct column and field, "
                f"as 'struct.field', got {self.name!r}"
            )
        aliases = self.call.aliases()
        if len(aliases) != 1:
            raise CompileError("AI.EXTRACT reads one document column")
        present = {field.alias for field in self.input.output_schema()}
        if self.alias not in present:
            raise CompileError(
                f"the extraction reads {self.alias!r}, which its input "
                f"does not produce ({sorted(present)})"
            )
        taken = {
            field.column
            for field in self.input.output_schema()
            if field.alias == self.alias
        }
        for name in (self.name, self.span_name):
            if name in taken:
                raise CompileError(f"the name {name!r} is already used")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("SemanticExtract needs one input")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if len(expressions) != 1:
            raise CompileError("SemanticExtract needs one call")
        return replace(self, call=expressions[0])

    def explain_fields(self) -> dict:
        return {
            "name": self.name,
            "expression": _explain(self.call),
            "question": self.call.question,
            "trim": self.call.trim,
        }


def extractions(node) -> tuple:
    """Return the SemanticExtract nodes at or below a node, children first."""
    return tuple(
        found for found in _subtree(node) if isinstance(found, SemanticExtract)
    )


def extracted_above_joins(root) -> frozenset:
    """Return aliases whose extraction runs after a join."""
    return _above_joins(extractions(root))
