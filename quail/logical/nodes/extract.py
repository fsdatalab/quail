"""The SemanticExtract node and the extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass

from quail.logical.expressions import SPAN_SUFFIX, CompileError
from quail.logical.nodes.column import SemanticColumn, column_nodes
from quail.logical.nodes.join import _above_joins


@dataclass(frozen=True)
class SemanticExtract(SemanticColumn):
    """One AI.EXTRACT field over its input rows, adding the answer columns.

    Every input row passes through with the answer in the column named
    ``name``, which is ``struct.field`` for the ExtractStruct field it
    fills, and its character span in the document, as a (start, end)
    pair, in the column ``name`` plus SPAN_SUFFIX. A row whose document
    does not answer the question gets nulls. The node starts on its
    table above the table's AI.IF SemanticFilter, classifications, and
    label filters, and below any join; the should_lift_maps rule moves it
    above the joins of a joined table, so the extraction runs
    over the documents the joins matched. The root Project returns the
    columns through an Alias of the same call.
    """

    type_name = "quail.semantic_extract"
    call_kind = "extract"
    function = "AI.EXTRACT"

    @property
    def span_name(self) -> str:
        """The name of the span column."""
        return self.name + SPAN_SUFFIX

    @property
    def added_columns(self) -> tuple[str, ...]:
        return (self.name, self.span_name)

    def validate_name(self) -> None:
        if self.name.count(".") != 1 or not all(self.name.split(".")):
            raise CompileError(
                f"an extraction is named after its struct column and field, "
                f"as 'struct.field', got {self.name!r}"
            )

    def explain_fields(self) -> dict:
        return {**super().explain_fields(), "question": self.call.question}


def extractions(node) -> tuple:
    """Return the SemanticExtract nodes at or below a node, children first."""
    return column_nodes(node, SemanticExtract)


def extracted_above_joins(root) -> frozenset:
    """Return aliases whose extraction runs after a join."""
    return _above_joins(extractions(root))
