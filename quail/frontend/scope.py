"""The tables a query names and the columns they resolve to.

Both front ends, SQL and the Python builder, resolve column references
through one Scope, so an unknown alias, an unknown or ambiguous
column, and a table read through two document columns raise the same
CompileError from either.
"""

import pyarrow as pa

from quail.catalog import Catalog
from quail.logical import ColumnRef, CompileError
from quail.logical.expressions import (
    ColumnExpression,
    column_name,
    column_reference,
    column_values,
    evaluate_column,
)


class Scope:
    """The (alias, provider) pairs in scope and each alias's document column.

    Args:
        catalog: The registered tables.
    """

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.tables: list[tuple[str, str]] = []
        self.doc_columns: dict[str, str | ColumnExpression] = {}

    def add_table(self, alias: str, provider: str) -> None:
        """Bring one table into scope under alias.

        Raises:
            CompileError: The provider is not registered, or the alias
                is already in scope.
        """
        self.catalog.get(provider)
        if alias in dict(self.tables):
            raise CompileError(f"duplicate table alias {alias!r}")
        self.tables.append((alias, provider))

    def resolve(self, column: str, alias: str | None = None, *,
                tables=None, shown: str = "") -> ColumnRef:
        """Resolve a column, qualified by an alias or found in one table.

        Args:
            column: The column name.
            alias: The qualifying table alias, or None to search the
                tables in scope.
            tables: (alias, provider) pairs to resolve against instead
                of the tables in scope.
            shown: How the reference was written, for the error message.

        Raises:
            CompileError: The alias is unknown, the column is not in the
                aliased table, or no single table has the column.
        """
        scope = dict(self.tables if tables is None else tables)
        if alias is not None:
            if alias not in scope:
                raise CompileError(
                    f"unknown table alias {alias!r}"
                    + (f" in column {shown}" if shown else ""))
            provider = scope[alias]
            if column not in self.catalog.get(provider).columns:
                raise CompileError(
                    f"column {column!r} not in provider {provider!r} "
                    f"(schema: {self.catalog.get(provider).columns})")
            return ColumnRef(alias=alias, provider=provider, column=column)
        owners = [(a, p) for a, p in scope.items()
                  if column in self.catalog.get(p).columns]
        if len(owners) != 1:
            raise CompileError(
                f"column {column!r} is "
                f"{'ambiguous' if owners else 'unknown'}; qualify it "
                f"with a table alias")
        return ColumnRef(alias=owners[0][0], provider=owners[0][1],
                         column=column)

    def note_doc_column(self, ref: ColumnRef | ColumnExpression) -> None:
        """Record the document column a prompt reads from ref's table.

        Raises:
            CompileError: The table was already read through another
                column; its KV is computed from one column only.
        """
        source = column_reference(ref)
        schema = self.catalog.get(source.provider).schema()
        values = evaluate_column(ref, pa.array([], schema.field(source.column).type))
        value_type = column_values(values).type
        if not (pa.types.is_string(value_type) or pa.types.is_large_string(value_type)
                or pa.types.is_string_view(value_type)):
            raise CompileError(f"AI input must be text, got {value_type}")
        document = ref if isinstance(ref, ColumnExpression) else ref.column
        seen = self.doc_columns.get(ref.alias)
        if seen is not None and seen != document:
            raise CompileError(
                f"alias {ref.alias!r} is referenced through two "
                f"columns ({column_name(seen)!r}, {column_name(document)!r}); "
                "predicates over one table must share one document column "
                "so its KV is computed once")
        self.doc_columns[ref.alias] = document

    def resolve_value(self, column: str, alias: str | None = None, *,
                      path=(), function=None, tables=None):
        """Bind a scalar column expression and resolve its Arrow result type."""
        ref = self.resolve(column, alias, tables=tables)
        if not path and function is None:
            return ref
        expression = ColumnExpression(ref, tuple(path), function)
        schema = self.catalog.get(ref.provider).schema()
        expression.evaluate(pa.array([], type=schema.field(ref.column).type))
        return expression
