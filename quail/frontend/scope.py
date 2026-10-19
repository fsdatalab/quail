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
    column_expression,
    column_name,
    column_references,
    evaluate_expression,
    table_alias,
)


class Scope:
    """The (alias, provider) pairs in scope and each alias's document column.

    Args:
        catalog: The registered tables.
    """

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.tables: list[tuple[str, str]] = []
        self.document_columns: dict = {}  # alias -> the document expression

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

    def resolve_value(self, column: str, alias: str | None = None, *,
                      path: tuple = (), function: str | None = None,
                      name: str | None = None, tables=None, shown: str = ""):
        """Resolve a column with the fields, function, and output name on it.

        The expression is checked against the column's type, so an
        unknown field or a function over the wrong type fails here
        rather than at execution.

        Raises:
            CompileError: As resolve(), or the fields or function do
                not apply to the column's type.
        """
        ref = self.resolve(column, alias, tables=tables, shown=shown)
        expression = column_expression(ref, tuple(path), function, name)
        if expression is not ref:
            expression.validate()
            schema = self.catalog.get(ref.provider).schema()
            evaluate_expression(expression, schema.empty_table())
        return expression

    def note_document_column(self, column) -> None:
        """Record the document column a prompt reads from the column's table.

        Raises:
            CompileError: The table was already read through another
                column; its KV is computed from one column only.
        """
        alias = table_alias(column)
        provider = column_references(column)[0].provider
        values = evaluate_expression(
            column, self.catalog.get(provider).schema().empty_table())
        if not (pa.types.is_string(values.type) or pa.types.is_large_string(
                values.type)):
            raise CompileError(
                f"a prompt reads text; {column_name(column)} is {values.type}")
        seen = self.document_columns.get(alias)
        if seen is not None and seen != column:
            raise CompileError(
                f"alias {alias!r} is referenced through two columns "
                f"({column_name(seen, qualified=False)!r}, "
                f"{column_name(column, qualified=False)!r}); predicates "
                f"over one table must share one document column so its KV "
                f"is computed once")
        self.document_columns[alias] = column
