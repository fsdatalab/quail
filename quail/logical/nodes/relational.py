"""The Project, Aggregate, and Result nodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical.expressions import (
    AggregateCall,
    Aggregation,
    Alias,
    ColumnRef,
    CompileError,
    HavingTest,
    SortKey,
    _explain,
    column_name,
)
from quail.logical.nodes.base import LogicalNode
from quail.logical.nodes.classify import classifications
from quail.logical.nodes.extract import extractions


@dataclass(frozen=True)
class Project:
    """Select source columns and named model outputs."""

    input: LogicalNode
    columns: tuple[ColumnRef | Alias, ...]
    type_name: ClassVar[str] = "quail.logical_project"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.columns

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return tuple(column for column in self.columns if isinstance(column, ColumnRef))

    def validate(self) -> None:
        computed = {(node.call, node.name) for node in classifications(self.input)}
        extracted = {(node.call, node.name) for node in extractions(self.input)}
        for column in self.columns:
            if isinstance(column, Alias):
                column.validate()
                if (
                    column.expression.kind == "label"
                    and (column.expression, column.name) not in computed
                ):
                    raise CompileError(
                        f"the label column {column.name!r} needs a "
                        f"SemanticClassify below the Project"
                    )
                if (
                    column.expression.kind == "extract"
                    and (column.expression, column.name) not in extracted
                ):
                    raise CompileError(
                        f"the answer column {column.name!r} needs a "
                        f"SemanticExtract below the Project"
                    )
            elif not isinstance(column, ColumnRef):
                raise CompileError(
                    f"a projected column is a column reference or a named "
                    f"expression, got {type(column).__name__}"
                )
        names = [
            f"{column.alias}.{column.column}"
            if isinstance(column, ColumnRef)
            else column.name
            for column in self.columns
        ]
        if len(names) != len(set(names)):
            raise CompileError(f"projection names must be unique, got {names}")

    def result_columns(self) -> tuple[str, ...]:
        """Return the selected column names."""
        return tuple(column_name(column) for column in self.columns)

    def with_children(self, children: tuple[LogicalNode, ...]) -> "Project":
        if len(children) != 1:
            raise CompileError("Project needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple) -> "Project":
        return replace(self, columns=expressions)

    def explain_fields(self) -> dict:
        return {
            "columns": [
                column_name(column)
                if isinstance(column, ColumnRef)
                else _explain(column)
                for column in self.columns
            ]
        }


@dataclass(frozen=True)
class Aggregate:
    """Group projected rows and compute aggregate outputs."""

    input: Project
    keys: tuple[str, ...]
    aggregates: tuple[AggregateCall, ...]
    output: tuple[str, ...]
    having: tuple[HavingTest, ...] = ()
    type_name: ClassVar[str] = "quail.logical_aggregate"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (*self.keys, *self.aggregates, *self.having)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = {column_name(ref): ref for ref in self.input.output_schema()}
        return tuple(fields.get(name, ColumnRef("", "", name)) for name in self.output)

    def validate(self) -> None:
        if not isinstance(self.input, Project):
            raise CompileError("Aggregate needs projected input columns")
        self.specification().validate(self.input.result_columns())

    def specification(self) -> Aggregation:
        """Return the grouping expressions and requested outputs."""
        return Aggregation(self.keys, self.aggregates, self.output, self.having)

    def result_columns(self) -> tuple[str, ...]:
        """Return the group keys and aggregate names in output order."""
        return self.output

    def with_children(self, children: tuple[LogicalNode, ...]) -> "Aggregate":
        if len(children) != 1:
            raise CompileError("Aggregate needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple) -> "Aggregate":
        if len(expressions) != len(self.expressions()):
            raise CompileError("Aggregate expression count must stay the same")
        keys_end = len(self.keys)
        aggregates_end = keys_end + len(self.aggregates)
        return replace(
            self,
            keys=expressions[:keys_end],
            aggregates=expressions[keys_end:aggregates_end],
            having=expressions[aggregates_end:],
        )

    def explain_fields(self) -> dict:
        return {"aggregation": str(self.specification())}


@dataclass(frozen=True)
class Result:
    """Order result rows, remove duplicates, and apply an offset and limit."""

    input: LogicalNode
    limit: int | None = None
    order: tuple[SortKey, ...] = ()
    offset: int = 0
    distinct: bool = False
    type_name: ClassVar[str] = "quail.logical_result"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple[SortKey, ...]:
        return self.order

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def result_columns(self) -> tuple[str, ...]:
        """Return column names after projection or aggregation."""
        return self.input.result_columns()

    def validate(self) -> None:
        if not isinstance(self.input, (Project, Aggregate)):
            raise CompileError("Result needs a projection or aggregation")
        if self.limit is not None and self.limit <= 0:
            raise CompileError("LIMIT must be a positive integer")
        if self.offset < 0:
            raise CompileError("OFFSET must be a nonnegative integer")
        projected = set(self.result_columns())
        projection = (
            self.input.input if isinstance(self.input, Aggregate) else self.input
        )
        for key in self.order:
            if not isinstance(key, SortKey):
                raise CompileError(
                    f"an ORDER BY term is a SortKey, got {type(key).__name__}"
                )
            key.validate()
            if (
                isinstance(key.expression, Alias)
                and key.expression not in projection.columns
            ):
                raise CompileError(
                    f"ORDER BY {key.name!r} names an expression the "
                    f"projection does not return"
                )
            if isinstance(self.input, Aggregate) and key.name not in projected:
                raise CompileError(
                    f"ORDER BY {key.name!r} with GROUP BY names a key or "
                    f"an aggregate in the SELECT list"
                )
            if self.distinct and key.name not in projected:
                raise CompileError(
                    f"ORDER BY {key.name!r} with DISTINCT needs the "
                    f"column in the SELECT list"
                )

    def with_children(self, children: tuple[LogicalNode, ...]) -> "Result":
        if len(children) != 1:
            raise CompileError("Result needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple) -> "Result":
        return replace(self, order=expressions)

    def explain_fields(self) -> dict:
        return {
            "order": [str(key) for key in self.order],
            "limit": self.limit,
            "offset": self.offset,
            "distinct": self.distinct,
        }
