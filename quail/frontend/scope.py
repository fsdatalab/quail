"""The tables a query names and the columns they resolve to.

Both front ends, SQL and the Python builder, resolve column references
through one Scope, so an unknown alias, an unknown or ambiguous
column, and a table read through two document columns raise the same
CompileError from either.
"""

import pyarrow as pa

from quail.catalog import Catalog
from quail.logical import ColumnExpression, ColumnRef, CompileError
from quail.logical.expressions import column_name, column_reference


class Scope:
    """The (alias, provider) pairs in scope and each alias's document column.

    Args:
        catalog: The registered tables.
    """

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.tables: list[tuple[str, str]] = []
        self.document_columns: dict[str, str | ColumnExpression] = {}

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
                      name: str | None = None, tables=None,
                      shown: str = "") -> ColumnRef | ColumnExpression:
        """Resolve a column with the fields, function, and output name on it.

        A plain column resolves to a ColumnRef; one with fields, a
        function, or a name to a ColumnExpression, checked against the
        column's type so an unknown field or a function over the wrong
        type fails here rather than at execution.

        Raises:
            CompileError: As resolve(), or the fields or function do
                not apply to the column's type.
        """
        ref = self.resolve(column, alias, tables=tables, shown=shown)
        if not path and function is None and name is None:
            return ref
        expression = ColumnExpression(ref, tuple(path), function, name)
        expression.validate()
        schema = self.catalog.get(ref.provider).schema()
        expression.evaluate(pa.array([], type=schema.field(ref.column).type))
        return expression

    def note_document_column(self, column: ColumnRef | ColumnExpression) -> None:
        """Record the document column a prompt reads from the column's table.

        Raises:
            CompileError: The table was already read through another
                column; its KV is computed from one column only.
        """
        alias = column_reference(column).alias
        document = (column if isinstance(column, ColumnExpression)
                    else column.column)
        seen = self.document_columns.get(alias)
        if seen is not None and seen != document:
            raise CompileError(
                f"alias {alias!r} is referenced through two columns "
                f"({column_name(seen, qualified=False)!r}, "
                f"{column_name(document, qualified=False)!r}); predicates "
                f"over one table must share one document column so its KV "
                f"is computed once")
        self.document_columns[alias] = document
