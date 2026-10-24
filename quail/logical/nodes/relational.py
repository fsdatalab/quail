"""The Project, Aggregate, Window, and Result nodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical.expressions import (
    AggregateCall,
    Aggregation,
    Alias,
    ColumnRef,
    CompileError,
    ExtractStruct,
    HavingTest,
    SortKey,
    WindowCall,
    _explain,
    aliased_call,
    column_name,
    column_references,
    packed_columns,
)
from quail.logical.nodes.base import LogicalNode
from quail.logical.nodes.classify import classifications
from quail.logical.nodes.extract import extractions
from quail.logical.nodes.filter import above_unnest
from quail.logical.nodes.sae_features import FEATURE_LIST


def _named_schema(input: LogicalNode, output: tuple[str, ...]) -> tuple:
    """Return the input's references for the named outputs, bare names otherwise."""
    fields = {column_name(ref): ref for ref in input.output_schema()}
    return tuple(fields.get(name, ColumnRef("", "", name)) for name in output)


@dataclass(frozen=True)
class Project:
    """Select value expressions over source columns and named model outputs.

    The fields of each ExtractStruct are extraction Aliases among the
    columns; the result returns them packed into the struct's columns.
    """

    input: LogicalNode
    columns: tuple  # value expressions and Aliases
    structs: tuple[ExtractStruct, ...] = ()
    type_name: ClassVar[str] = "quail.logical_project"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.columns

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return tuple(dict.fromkeys(
            ref for column in self.columns if aliased_call(column) is None
            for ref in column_references(column)))

    def validate(self) -> None:
        computed = {(node.call, node.name) for node in classifications(self.input)}
        extracted = {(node.call, node.name) for node in extractions(self.input)}
        for column in self.columns:
            column.validate()
            call = aliased_call(column)
            if call is not None and call.kind == "label" and (
                    call, column.name) not in computed:
                raise CompileError(
                    f"the label column {column.name!r} needs a "
                    f"SemanticClassify below the Project"
                )
            if call is not None and call.kind == "extract" and (
                    call, column.name) not in extracted:
                raise CompileError(
                    f"the answer column {column.name!r} needs a "
                    f"SemanticExtract below the Project"
                )
        names = [column_name(column) for column in self.columns]
        if len(names) != len(set(names)):
            raise CompileError(f"projection names must be unique, got {names}")
        self._validate_structs()

    def _validate_structs(self) -> None:
        extracted = {column.name: column for column in self.columns
                     if aliased_call(column, "extract") is not None}
        members = set()
        for struct in self.structs:
            struct.validate()
            for member in struct.members:
                if member not in extracted:
                    raise CompileError(
                        f"the struct column {struct.name!r} needs the "
                        f"extraction {member!r} in the projection")
                members.add(member)
            if len({extracted[member].expression.aliases()
                    for member in struct.members}) != 1:
                raise CompileError(
                    f"the fields of {struct.name!r} read one document column")
        for name in extracted:
            if name not in members:
                raise CompileError(
                    f"the extraction {name!r} is a field of no struct column")
        others = [column_name(column) for column in self.columns
                  if column_name(column) not in members]
        packed = [name for struct in self.structs
                  for name, _ in struct.packing()]
        if len(set(others + packed)) != len(others) + len(packed):
            raise CompileError(
                f"projection names must be unique, got {others + packed}")

    def packing(self) -> tuple:
        """Return every struct column and its member columns, answers and spans."""
        return tuple(entry for struct in self.structs for entry in struct.packing())

    def result_columns(self) -> tuple[str, ...]:
        """Return the selected column names, with struct fields packed."""
        return packed_columns(
            (column_name(column) for column in self.columns), self.packing())

    def list_types(self) -> dict:
        """Return the Arrow type of each list column the projection returns."""
        return {column.name: FEATURE_LIST for column in self.columns
                if aliased_call(column, "features") is not None}

    def with_children(self, children: tuple[LogicalNode, ...]) -> "Project":
        if len(children) != 1:
            raise CompileError("Project needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple) -> "Project":
        return replace(self, columns=expressions)

    def explain_fields(self) -> dict:
        fields = {"columns": [_explain(column) for column in self.columns]}
        if self.structs:
            fields["structs"] = {struct.name: list(struct.members)
                                 for struct in self.structs}
        return fields


@dataclass(frozen=True)
class Aggregate:
    """Group projected or unnested rows and compute aggregate outputs."""

    input: LogicalNode
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
        return _named_schema(self.input, self.output)

    def validate(self) -> None:
        if not reads_projection(self.input):
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
class Window:
    """Compute window functions over projected or aggregated rows.

    ``output`` lists the result columns in SELECT order: input columns
    and window names. An input column no window output names is read
    by a window only.
    """

    input: LogicalNode
    calls: tuple[WindowCall, ...]
    output: tuple[str, ...]
    type_name: ClassVar[str] = "quail.logical_window"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.calls

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return _named_schema(self.input, self.output)

    def validate(self) -> None:
        if not (reads_projection(self.input) or isinstance(self.input, Aggregate)):
            raise CompileError("Window needs projected or aggregated input columns")
        projected = self.input.result_columns()
        names = list(projected)
        for call in self.calls:
            call.validate(projected)
            names.append(call.name)
        if len(names) != len(set(names)):
            raise CompileError(f"window names must be unique, got {names}")
        for name in self.output:
            if name not in names:
                raise CompileError(
                    f"the window output {name!r} is neither a column nor a window")

    def result_columns(self) -> tuple[str, ...]:
        """Return the input columns and window names in output order."""
        return self.output

    def with_children(self, children: tuple[LogicalNode, ...]) -> "Window":
        if len(children) != 1:
            raise CompileError("Window needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple) -> "Window":
        return replace(self, calls=expressions)

    def explain_fields(self) -> dict:
        return {"windows": [str(call) for call in self.calls]}


@dataclass(frozen=True)
class Result:
    """Order result rows, remove duplicates, and apply an offset and limit.

    ``output`` lists the result columns in SELECT order when it differs
    from the input's, as over an Unnest; empty keeps the input's.
    """

    input: LogicalNode
    limit: int | None = None
    order: tuple[SortKey, ...] = ()
    offset: int = 0
    distinct: bool = False
    output: tuple[str, ...] = ()
    type_name: ClassVar[str] = "quail.logical_result"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple[SortKey, ...]:
        return self.order

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def result_columns(self) -> tuple[str, ...]:
        """Return column names after projection or aggregation."""
        return self.output or self.input.result_columns()

    def validate(self) -> None:
        if not (reads_projection(self.input)
                or isinstance(self.input, (Aggregate, Window))):
            raise CompileError("Result needs a projection, aggregation, or window")
        missing = set(self.output) - set(self.input.result_columns())
        if missing:
            raise CompileError(
                f"the result columns {sorted(missing)} are not columns of "
                f"the rows below")
        if self.limit is not None and self.limit <= 0:
            raise CompileError("LIMIT must be a positive integer")
        if self.offset < 0:
            raise CompileError("OFFSET must be a nonnegative integer")
        projected = set(self.result_columns())
        projection = self.input
        while not isinstance(projection, Project):
            projection = projection.input
        grouped = isinstance(self.input, Aggregate) or (
            isinstance(self.input, Window)
            and isinstance(self.input.input, Aggregate))
        for key in self.order:
            if not isinstance(key, SortKey):
                raise CompileError(
                    f"an ORDER BY term is a SortKey, got {type(key).__name__}"
                )
            key.validate()
            if (
                isinstance(key.expression, Alias)
                and key.expression not in projection.columns
                and key.name not in projected
            ):
                raise CompileError(
                    f"ORDER BY {key.name!r} names an expression the "
                    f"projection does not return"
                )
            if grouped and key.name not in projected:
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
        fields = {
            "order": [str(key) for key in self.order],
            "limit": self.limit,
            "offset": self.offset,
            "distinct": self.distinct,
        }
        if self.output:
            fields["output"] = list(self.output)
        return fields


def reads_projection(node) -> bool:
    """Return whether the node is the Project or an Unnest chain above it."""
    return isinstance(node, Project) or above_unnest(node)
