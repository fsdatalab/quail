"""The Scan node that names a provider's document column."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar

from quail.logical.expressions import (
    ColumnRef,
    CompileError,
    RegularPredicate,
    column_name,
    column_references,
    source_columns,
)


@dataclass(frozen=True)
class Scan:
    """Which column of which provider supplies the document text.

    ``column`` is tokenized for the model: a value expression over the
    table's source columns, such as a column reference or a struct
    field. ``columns`` names the source columns kept as values for the
    result rows, and includes the document's source column only when
    the query returns its text. The
    projection pushdown rule fills ``columns``; before it runs the
    tuple is empty. ``predicates`` are regular predicates, all of which a
    document must pass before any operator reads it. ``schema`` is the
    provider's Arrow schema, which types the plan's result; a front end
    sets it from the catalog.
    """

    provider: str
    alias: str
    column: Any  # a value expression, or "" before a prompt names one
    columns: tuple = ()  # tuple[str, ...]
    predicates: tuple = ()  # tuple[RegularPredicate, ...]
    schema: Any = None  # pa.Schema

    type_name: ClassVar[str] = "quail.scan"

    def children(self) -> tuple:
        return ()

    def expressions(self) -> tuple:
        return self.predicates

    def output_schema(self) -> tuple[ColumnRef, ...]:
        sources = (source_columns(self.column) if self.column
                   else tuple(self.schema.names) if self.schema is not None else ())
        fields = [ColumnRef(self.alias, self.provider, source) for source in sources]
        fields.extend(
            ColumnRef(self.alias, self.provider, column)
            for column in self.columns
            if column not in sources
        )
        return tuple(fields)

    def validate(self) -> None:
        if (not self.provider or not self.alias
                or (not self.column and self.schema is None)):
            raise CompileError("Scan needs a provider, alias, and column")
        if not isinstance(self.column, str):
            self.column.validate()
            if any(ref.alias != self.alias for ref in column_references(self.column)):
                raise CompileError(
                    f"Scan {self.alias!r} reads the document column "
                    f"{self.column} of another table")
        if len(set(self.columns)) != len(self.columns):
            raise CompileError(
                f"Scan {self.alias!r} lists a column twice: {self.columns}"
            )
        for predicate in self.predicates:
            if not isinstance(predicate, RegularPredicate):
                raise CompileError(
                    f"a Scan predicate is a RegularPredicate, got "
                    f"{type(predicate).__name__}"
                )
            predicate.validate()
            if predicate.column.alias != self.alias:
                raise CompileError(
                    f"predicate {predicate} is on the Scan of {self.alias!r}"
                )

    def with_children(self, children: tuple) -> "Scan":
        if children:
            raise CompileError("Scan has no children")
        return self

    def with_expressions(self, expressions: tuple) -> "Scan":
        return replace(self, predicates=tuple(expressions))

    def explain_fields(self) -> dict:
        return {
            "provider": self.provider,
            "alias": self.alias,
            "column": column_name(self.column, qualified=False),
            "columns": list(self.columns),
            "predicates": [str(predicate) for predicate in self.predicates],
        }
