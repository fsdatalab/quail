"""The SemanticSaeFeatures node: one AI.SAE_FEATURES call over one table."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import pyarrow as pa

from quail.logical.expressions import (
    ColumnRef,
    CompileError,
    _explain,
)
from quail.logical.nodes.base import LogicalNode, _subtree

# the Arrow type of one feature list
FEATURE_LIST = pa.list_(
    pa.struct([("feature", pa.int32()), ("strength", pa.float32())]))


@dataclass(frozen=True)
class SemanticSaeFeatures:
    """One AI.SAE_FEATURES call over its input rows, adding the list column.

    Every input row passes through with its sparse autoencoder (SAE)
    features in the column named ``name``, a FEATURE_LIST holding the
    nonzero features. The call reads one table and generates no text.
    The root Project returns the column through an Alias of the same
    call.
    """

    input: LogicalNode
    call: object    # a ModelCall of kind "features"
    name: str

    type_name: ClassVar[str] = "quail.semantic_sae_features"

    @property
    def alias(self) -> str:
        """The table alias the feature list column belongs to."""
        return self.call.aliases()[0]

    @property
    def pooling(self) -> str:
        """Which tokens become a row's list: 'last' or 'max'."""
        return self.call.pooling

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (self.call,)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = self.input.output_schema()
        provider = next(
            (field.provider for field in fields if field.alias == self.alias), ""
        )
        return fields + (ColumnRef(self.alias, provider, self.name),)

    def validate(self) -> None:
        self.call.validate()
        if self.call.kind != "features":
            raise CompileError(
                "SemanticSaeFeatures needs an AI.SAE_FEATURES call")
        if not self.name or "." in self.name:
            raise CompileError(
                f"a feature call needs a column name without a dot, "
                f"got {self.name!r}")
        aliases = self.call.aliases()
        if len(aliases) != 1:
            raise CompileError("AI.SAE_FEATURES reads one document column")
        present = {field.alias for field in self.input.output_schema()}
        if aliases[0] not in present:
            raise CompileError(
                f"the feature call reads {aliases[0]!r}, which its input "
                f"does not produce ({sorted(present)})")
        if any(
            field.alias == self.alias and field.column == self.name
            for field in self.input.output_schema()
        ):
            raise CompileError(f"the name {self.name!r} is already used")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("SemanticSaeFeatures needs one input")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if len(expressions) != 1:
            raise CompileError("SemanticSaeFeatures needs one call")
        return replace(self, call=expressions[0])

    def explain_fields(self) -> dict:
        return {
            "name": self.name,
            "expression": _explain(self.call),
            "pooling": self.pooling,
        }


def feature_calls(node) -> tuple:
    """Return the SemanticSaeFeatures nodes at or below a node, children first."""
    return tuple(
        found for found in _subtree(node)
        if isinstance(found, SemanticSaeFeatures))
