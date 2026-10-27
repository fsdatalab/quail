"""The SemanticClassify node and the classification helpers."""

from __future__ import annotations

from dataclasses import dataclass

from quail.logical.expressions import (
    PROBABILITIES_SUFFIX,
    CompileError,
    model_call,
)
from quail.logical.nodes.base import _subtree
from quail.logical.nodes.column import SemanticColumn, column_nodes
from quail.logical.nodes.join import SemanticJoin, _above_joins


@dataclass(frozen=True)
class SemanticClassify(SemanticColumn):
    """One AI.CLASSIFY call over its input rows, adding the label column.

    Every input row passes through with its label in the column named
    ``name``; with ``probabilities`` a second column, ``name`` plus
    PROBABILITIES_SUFFIX, holds each label's probability. A call over
    one table starts on that table above its AI.IF SemanticFilter and
    below any join; when a join reads the table, the physical planner
    also prices running it above the joins (should_lift_maps). A call over
    two tables sits above the SemanticJoin of the two. A
    Filter testing the label column sits above this node, and the root
    Project returns the column through an Alias of the same call.
    """

    type_name = "quail.semantic_classify"
    call_kind = "label"
    function = "AI.CLASSIFY"
    num_aliases = (1, 2)

    @property
    def probabilities(self) -> bool:
        """Whether the probabilities column is added beside the label."""
        return self.call.probabilities

    @property
    def added_columns(self) -> tuple[str, ...]:
        if self.probabilities:
            return (self.name, self.name + PROBABILITIES_SUFFIX)
        return (self.name,)

    def validate(self) -> None:
        super().validate()
        aliases = self.call.aliases()
        if len(aliases) == 2 and not any(
            isinstance(node, SemanticJoin)
            and set(model_call(node.predicate).aliases()) == set(aliases)
            for node in _subtree(self.input)
        ):
            raise CompileError(
                f"the classification of {aliases[0]!r} x {aliases[1]!r} "
                f"joined rows needs a join of the two"
            )

    def explain_fields(self) -> dict:
        return {**super().explain_fields(), "probabilities": self.probabilities}


def classifications(node) -> tuple:
    """Return the SemanticClassify nodes at or below a node, children first."""
    return column_nodes(node, SemanticClassify)


def classified_above_joins(root) -> frozenset:
    """Return aliases classified one document at a time after a join."""
    return _above_joins(node for node in classifications(root)
                        if len(node.call.aliases()) == 1)
