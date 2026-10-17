"""The Scan node that names a provider's document column."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical.expressions import (
    ColumnExpression,
    ColumnRef,
    CompileError,
    column_name,
    source_column,
)
from quail.logical.predicates import RegularPredicate


@dataclass(frozen=True)
class Scan:
    """Which column of which provider supplies the document text.

    ``column`` is tokenized for the model. ``columns`` names the source
    columns kept as values for projections, predicates, joins, callbacks,
    and extractions. It includes ``column`` when they read the document text. The
    projection pushdown rule fills ``columns``; before it runs the
    tuple is empty. ``predicates`` are regular predicates, all of which a
    document must pass before any operator reads it.
    """

    provider: str
    alias: str
    column: str | ColumnExpression
    columns: tuple = ()  # tuple[str, ...]
    predicates: tuple = ()  # tuple[RegularPredicate, ...]

    type_name: ClassVar[str] = "quail.scan"

    def children(self) -> tuple:
        return ()

    def expressions(self) -> tuple:
        return self.predicates

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = [ColumnRef(self.alias, self.provider, source_column(self.column))]
        fields.extend(
            ColumnRef(self.alias, self.provider, column)
            for column in self.columns
            if column != source_column(self.column)
        )
        return tuple(fields)

    def validate(self) -> None:
        if not self.provider or not self.alias or not self.column:
            raise CompileError("Scan needs a provider, alias, and column")
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
