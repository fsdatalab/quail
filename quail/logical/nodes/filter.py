"""The SemanticFilter and Filter nodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Optional

from quail.logical.expressions import (
    ColumnRef,
    CompileError,
    RegularPredicate,
    _explain,
    column_name,
    model_call,
    validate_predicate,
)
from quail.logical.nodes.base import LogicalNode
from quail.logical.nodes.classify import classifications
from quail.logical.nodes.extract import extractions
from quail.logical.nodes.unnest import Unnest


@dataclass(frozen=True)
class SemanticFilter:
    """Predicates over one table's documents, in written order.

    The physical planner chooses the order they run in. ``stop_key``
    names columns of the filtered table once the should_push_down_distinct rule
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

    The condition tests a column a one-table SemanticClassify or a
    SemanticExtract below adds: a label, an answer, or an answer's span.
    Above an Unnest, it tests a column the rows below return, such as a
    field column of an unnested list.
    """

    input: LogicalNode
    condition: RegularPredicate
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
        if above_unnest(self.input):
            if column_name(column) not in self.input.result_columns():
                raise CompileError(
                    f"a Filter above an Unnest tests a column the rows below "
                    f"return, got {column_name(column)}")
            return
        if (column.alias, column.column) not in computed_columns(self.input):
            raise CompileError(
                f"a Filter tests a column a one-table classification or "
                f"an extraction below it computes; nothing below computes "
                f"{column.alias}.{column.column}"
            )

    def result_columns(self) -> tuple[str, ...]:
        """Return the columns of the rows below, above an Unnest."""
        return self.input.result_columns()

    def list_types(self) -> dict:
        """Return the list columns of the rows below, above an Unnest."""
        return self.input.list_types()

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
            "condition": str(self.condition),
            "selectivity": self.selectivity,
        }


def computed_columns(node) -> frozenset:
    """Return the columns the classifications and extractions below compute.

    The columns are (alias, column) pairs: labels, answers, and answer
    spans at or below the node.
    """
    columns = {(found.alias, found.name) for found in classifications(node)
               if len(found.call.aliases()) == 1}
    for found in extractions(node):
        columns.update({(found.alias, found.name), (found.alias, found.span_name)})
    return frozenset(columns)


def above_unnest(node) -> bool:
    """Return whether the node is an Unnest or a Filter above one."""
    while isinstance(node, Filter):
        node = node.input
    return isinstance(node, Unnest)
