"""Rows: what document operators pass to one another.

A row names its documents by their ids, one per alias, and carries
value columns: scores, labels, answers, spans. The ids and the values
are separate, so an operator after this one never guesses which
columns hold ids.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail.execution.ids import document_ids
from quail.execution.result import document_index_schema


@dataclass(frozen=True)
class Rows:
    """Rows over one or two tables: their document ids by alias, and values.

    Attributes:
        ids: alias -> the document positions of each row, int32.
        values: The value columns, one row each; never an id.
    """

    ids: Mapping[str, pa.Array]
    values: pa.Table = field(default_factory=lambda: pa.table({}))

    def __post_init__(self) -> None:
        ids = {alias: document_ids(value) for alias, value in self.ids.items()}
        if not ids:
            raise ValueError("rows name at least one table")
        lengths = {len(value) for value in ids.values()}
        if len(lengths) != 1:
            raise ValueError("every alias names one document per row")
        (length,) = lengths
        values = self.values
        if values.num_columns == 0 and values.num_rows != length:
            values = pa.table({}).slice(0, 0) if length == 0 else _empty(length)
        if values.num_rows != length:
            raise ValueError(
                f"{values.num_rows} value rows for {length} id rows")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "values", values)

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(self.ids)

    @property
    def num_rows(self) -> int:
        return len(next(iter(self.ids.values())))

    def __len__(self) -> int:
        return self.num_rows

    @property
    def column_names(self) -> list[str]:
        return self.values.column_names

    def column(self, name: str) -> pa.ChunkedArray:
        """Return one value column."""
        return self.values.column(name)

    def filter(self, mask) -> Rows:
        """Keep the rows the boolean mask is true for."""
        return Rows({alias: pc.filter(ids, mask) for alias, ids in self.ids.items()},
                    self.values.filter(mask))

    def take(self, indices) -> Rows:
        """Return the rows at the indices, in that order."""
        indices = pa.array(np.asarray(indices))
        return Rows({alias: pc.take(ids, indices) for alias, ids in self.ids.items()},
                    self.values.take(indices))

    def with_column(self, name: str, array) -> Rows:
        """Return the rows with one value column added."""
        return Rows(self.ids, self.values.append_column(name, array))

    def joined(self, prior: Rows, alias: str) -> Rows:
        """Return the rows with the prior's values of the same documents.

        Every row names a document the prior also holds, so the lookup
        by id never misses.
        """
        if not prior.column_names:
            return self
        positions = pc.index_in(self.ids[alias], value_set=prior.ids[alias])
        rows = self
        for name in prior.column_names:
            rows = rows.with_column(name, pc.take(prior.column(name), positions))
        return rows

    def table(self, kind: str = "rows") -> pa.Table:
        """Return the rows as one table: the id columns, then the values.

        The id fields carry the document index schema the result
        assembly joins on.
        """
        schema = document_index_schema(self.ids, kind)
        arrays = [self.ids[name] for name in schema.names]
        for name in self.values.column_names:
            schema = schema.append(self.values.schema.field(name))
            arrays.append(self.values.column(name))
        return pa.Table.from_arrays(arrays, schema=schema)

    def equals(self, other: Rows) -> bool:
        """Return whether the rows hold the same ids and values, in order."""
        return self.table().equals(other.table())

    def to_pydict(self) -> dict:
        """Return the ids and values as Python lists, ids first."""
        return self.table().to_pydict()

    @classmethod
    def from_table(cls, table: pa.Table, aliases: Iterable[str]) -> Rows:
        """Split a table into the rows: the named columns are the ids."""
        aliases = tuple(aliases)
        return cls({alias: table.column(alias) for alias in aliases},
                   table.drop_columns(list(aliases)))

    @classmethod
    def concat(cls, parts: Iterable[Rows]) -> Rows:
        """Stack rows of the same aliases and value columns."""
        parts = list(parts)
        if not parts:
            raise ValueError("no rows to concatenate")
        return cls({alias: pa.concat_arrays([part.ids[alias] for part in parts])
                    for alias in parts[0].aliases},
                   pa.concat_tables([part.values for part in parts]))


def _empty(length: int) -> pa.Table:
    """A table of the length with no columns."""
    return pa.table({"_": pa.nulls(length)}).drop_columns(["_"])
