"""The SemanticFilter and Filter nodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Optional

from quail.common.expressions import (
    And,
    ColumnRef,
    CompileError,
    Not,
    Or,
    RegularPredicate,
    _explain,
    column_name,
    column_references,
    model_call,
    validate_predicate,
)
from quail.logical.nodes.base import LogicalNode
from quail.logical.nodes.column import SemanticColumn, column_nodes
from quail.logical.nodes.scan import Scan
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

    Below the Project, the condition tests columns the one-table
    column operators below add, such as labels, answers and spans, or
    map answers; it may test the table's source columns too, read by
    document ID. Above the Project or an Unnest, it tests the columns
    the result rows below return, by name, such as ``r.stars``,
    ``lease.ends``, or a field column of an unnested list.
    """

    input: LogicalNode
    condition: RegularPredicate | Not | And | Or
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
        references = [reference for test in self.condition.tests()
                      for reference in column_references(test.expression)]
        from quail.logical.nodes.relational import readable_columns, reads_projection

        if reads_projection(self.input):
            returned = readable_columns(self.input)
            for reference in references:
                if column_name(reference) not in returned:
                    raise CompileError(
                        f"a Filter on result rows tests a column the rows "
                        f"below return, got {column_name(reference)}")
            return
        computed = computed_columns(self.input)
        for reference in references:
            if ((reference.alias, reference.column) not in computed
                    and reference.column not in scan_columns(self.input,
                                                             reference.alias)):
                raise CompileError(
                    f"a Filter tests a column a one-table column operator "
                    f"below it computes; nothing below "
                    f"computes {reference.alias}.{reference.column}"
                )

    @property
    def alias(self) -> str:
        """Return the one table a Filter below the Project tests."""
        (alias,) = self.condition.aliases()
        return alias

    def result_columns(self) -> tuple[str, ...]:
        """Return the columns of the result rows below."""
        return self.input.result_columns()

    def list_types(self) -> dict:
        """Return the list columns of the result rows below."""
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
    """Return (alias, column) pairs computed by column operators below."""
    return frozenset((found.alias, name)
                     for found in column_nodes(node, SemanticColumn)
                     if len(found.call.aliases()) == 1
                     for name in found.added_columns)


def scan_columns(node, alias: str) -> frozenset:
    """Return the source columns of the table an alias names, below a node."""
    while not isinstance(node, Scan):
        node = next((child for child in node.children()
                     if alias in {field.alias for field in child.output_schema()}),
                    None)
        if node is None:
            return frozenset()
    if node.alias != alias or node.schema is None:
        return frozenset()
    return frozenset(node.schema.names)


def above_projection(node) -> bool:
    """Return whether the node reads result rows.

    That is an Unnest, or a Filter above an Unnest or the Project; the
    Project itself reads documents.
    """
    from quail.logical.nodes.relational import Project

    while isinstance(node, Filter):
        node = node.input
        if isinstance(node, Project):
            return True
    return isinstance(node, Unnest)
