"""The SemanticMap node: one AI.MAP call over one table."""

from __future__ import annotations

from dataclasses import dataclass

from quail.logical.nodes.column import SemanticColumn, column_nodes
from quail.logical.nodes.join import _above_joins


@dataclass(frozen=True)
class SemanticMap(SemanticColumn):
    """One AI.MAP call over its input rows, adding the answer column.

    Every input row passes through with the answer the model writes for
    the instruction in the column named ``name``, a string of at most
    the call's answer limit in tokens. The node starts on its table
    above the table's filters, classifications, and extractions, and
    below any join; the planner may lift it above the joins, to run on
    the documents they keep. The root Project returns the column through an
    Alias of the same call.
    """

    type_name = "quail.semantic_map"
    call_kind = "map"
    function = "AI.MAP"

    def explain_fields(self) -> dict:
        return {**super().explain_fields(),
                "max_answer_tokens": self.call.max_answer_tokens}


def maps(node) -> tuple:
    """Return the SemanticMap nodes at or below a node, children first."""
    return column_nodes(node, SemanticMap)


def mapped_above_joins(root) -> frozenset:
    """Return aliases whose maps run after a join."""
    return _above_joins(maps(root))
