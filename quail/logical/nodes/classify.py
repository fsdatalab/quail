"""The SemanticClassify node and the classification helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical.expressions import (
    PROBABILITIES_SUFFIX,
    ColumnRef,
    CompileError,
    ModelCall,
    _explain,
    model_call,
)
from quail.logical.nodes.base import LogicalNode, _subtree
from quail.logical.nodes.join import SemanticJoin, _above_joins


@dataclass(frozen=True)
class SemanticClassify:
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

    input: LogicalNode
    call: ModelCall
    name: str

    type_name: ClassVar[str] = "quail.semantic_classify"

    @property
    def probabilities(self) -> bool:
        """Whether the probabilities column is added beside the label."""
        return self.call.probabilities

    @property
    def alias(self) -> str:
        """The table alias the label column belongs to."""
        return self.call.aliases()[0]

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (self.call,)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = self.input.output_schema()
        provider = next(
            (field.provider for field in fields if field.alias == self.alias), ""
        )
        added = [ColumnRef(self.alias, provider, self.name)]
        if self.probabilities:
            added.append(
                ColumnRef(self.alias, provider, self.name + PROBABILITIES_SUFFIX)
            )
        return fields + tuple(added)

    def validate(self) -> None:
        self.call.validate()
        if self.call.kind != "label":
            raise CompileError("SemanticClassify needs an AI.CLASSIFY call")
        if not self.name or "." in self.name:
            raise CompileError(
                f"a classification needs a column name without a dot, got {self.name!r}"
            )
        aliases = self.call.aliases()
        if len(aliases) not in (1, 2):
            raise CompileError(
                "AI.CLASSIFY reads one document, or one from each side of a join"
            )
        present = {field.alias for field in self.input.output_schema()}
        missing = set(aliases) - present
        if missing:
            raise CompileError(
                f"the classification reads {sorted(missing)}, which its "
                f"input does not produce ({sorted(present)})"
            )
        if any(
            field.alias == self.alias and field.column == self.name
            for field in self.input.output_schema()
        ):
            raise CompileError(f"the name {self.name!r} is already used")
        if len(aliases) == 2 and not any(
            isinstance(node, SemanticJoin)
            and set(model_call(node.predicate).aliases()) == set(aliases)
            for node in _subtree(self.input)
        ):
            raise CompileError(
                f"the classification of {aliases[0]!r} x {aliases[1]!r} "
                f"joined rows needs a join of the two"
            )

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("SemanticClassify needs one input")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if len(expressions) != 1:
            raise CompileError("SemanticClassify needs one call")
        return replace(self, call=expressions[0])

    def explain_fields(self) -> dict:
        return {
            "name": self.name,
            "expression": _explain(self.call),
            "probabilities": self.probabilities,
        }


def classifications(node) -> tuple:
    """Return the SemanticClassify nodes at or below a node, children first."""
    return tuple(
        found for found in _subtree(node) if isinstance(found, SemanticClassify)
    )


def classified_above_joins(root) -> frozenset:
    """Return aliases classified one document at a time after a join."""
    return _above_joins(node for node in classifications(root)
                        if len(node.call.aliases()) == 1)
