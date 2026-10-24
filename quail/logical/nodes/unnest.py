"""The Unnest node: one row per element of list columns."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical.expressions import ColumnRef, CompileError
from quail.logical.nodes.base import LogicalNode


@dataclass(frozen=True)
class Unnest:
    """Return one row per element of list columns, zipped by position.

    Each list column in ``columns`` holds structs, and the node replaces
    it with one column per struct field, named ``<column>.<field>``.
    The other columns repeat for each element. A row returns as many
    rows as its longest list; a shorter list gives nulls in the rows
    past its end, and a row whose lists are all empty or null returns
    no rows. Two Unnest nodes over each other return the product of
    their lists.

    The input is the Project, or an Unnest or Filter above it, and
    returns the list columns by name.
    """

    input: LogicalNode
    columns: tuple[str, ...]
    type_name: ClassVar[str] = "quail.logical_unnest"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return ()

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def fields(self) -> dict[str, tuple[str, ...]]:
        """Return the field columns each list column becomes, in field order."""
        types = self.input.list_types()
        return {column: tuple(f"{column}.{name}"
                              for name in types[column].value_type.names)
                for column in self.columns}

    def list_types(self) -> dict:
        """Return the Arrow type of each list column the node still returns."""
        return {name: kind for name, kind in self.input.list_types().items()
                if name not in self.columns}

    def result_columns(self) -> tuple[str, ...]:
        """Return the input's columns, with each list as its field columns."""
        fields = self.fields()
        return tuple(name for column in self.input.result_columns()
                     for name in fields.get(column, (column,)))

    def validate(self) -> None:
        if not self.columns:
            raise CompileError("UNNEST needs at least one list column")
        if len(set(self.columns)) != len(self.columns):
            raise CompileError(f"UNNEST names a list twice, got {self.columns}")
        types = self.input.list_types()
        for column in self.columns:
            if column not in types:
                raise CompileError(
                    f"UNNEST reads a list column the rows below return, got "
                    f"{column!r}; the list columns are {sorted(types)}")
        names = self.result_columns()
        if len(names) != len(set(names)):
            raise CompileError(
                f"UNNEST field columns must not repeat a column name, got "
                f"{names}")

    def with_children(self, children: tuple[LogicalNode, ...]) -> "Unnest":
        if len(children) != 1:
            raise CompileError("Unnest needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple) -> "Unnest":
        if expressions:
            raise CompileError("Unnest has no expressions")
        return self

    def explain_fields(self) -> dict:
        return {"columns": list(self.columns)}

