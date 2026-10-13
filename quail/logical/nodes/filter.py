"""The SemanticFilter and Filter nodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Optional

from quail.logical.expressions import (
    ColumnRef,
    CompileError,
    InList,
    _explain,
    model_call,
    validate_predicate,
)
from quail.logical.nodes.base import LogicalNode
from quail.logical.nodes.classify import classifications


@dataclass(frozen=True)
class SemanticFilter:
    """Predicates over one table's documents, in written order.

    The physical planner chooses the order they run in. ``stop_key``
    names columns of the filtered table once the distinct_pushdown rule
    has found that the query asks only whether any document of each key
    value passes; the executor then skips a key's remaining documents
    once one survives.
    """

    input: LogicalNode
    predicates: tuple  # tuple[FilterPredicate, ...], written order
    stop_key: tuple = ()  # tuple[str, ...]

    type_name: ClassVar[str] = "quail.semantic_filter"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.predicates

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def validate(self) -> None:
        if not self.predicates:
            raise CompileError("SemanticFilter needs at least one predicate")
        for predicate in self.predicates:
            validate_predicate(predicate.expression)
            if len(model_call(predicate.expression).aliases()) != 1:
                raise CompileError("a SemanticFilter predicate reads one table")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("SemanticFilter needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if not expressions:
            raise CompileError("SemanticFilter needs at least one predicate")
        return replace(self, predicates=expressions)

    def explain_fields(self) -> dict:
        return {
            "predicates": len(self.predicates),
            "selectivities": [p.selectivity for p in self.predicates],
            "expressions": [_explain(p.expression) for p in self.predicates],
            "stop_key": list(self.stop_key),
        }


@dataclass(frozen=True)
class Filter:
    """Rows whose condition holds, decided without the model.

    The condition tests the label column a one-table SemanticClassify
    below adds.
    """

    input: LogicalNode
    condition: InList
    selectivity: Optional[float] = None  # fraction of rows that pass;
    #                                       None means not given

    type_name: ClassVar[str] = "quail.filter"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (self.condition,)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def validate(self) -> None:
        self.condition.validate()
        column = self.condition.column
        if not any(
            node.alias == column.alias
            and node.name == column.column
            and len(node.call.aliases()) == 1
            for node in classifications(self.input)
        ):
            raise CompileError(
                f"a Filter tests the label column of a one-table "
                f"classification below it; nothing below computes "
                f"{column.alias}.{column.column}"
            )

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("Filter needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if len(expressions) != 1:
            raise CompileError("Filter needs one condition")
        return replace(self, condition=expressions[0])

    def explain_fields(self) -> dict:
        return {
            "condition": _explain(self.condition),
            "selectivity": self.selectivity,
        }
