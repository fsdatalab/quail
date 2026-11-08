"""A region input the executor reads batch by batch, as DataFusion produces it.

Rust's ``InputReader`` numbers the input's rows from 0 in the order
they arrive, and hands on each row's document tokens and the input
columns the executor reads. ``Arrivals`` keeps them by that number, so
it serves as the scan's token store and as its value table while it
grows. Once the input has ended it holds every row.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.common.tokens import ArrowTokenDocuments, DocumentBatches, first_documents


class Arrivals(Sequence):
    """The rows of a region input that have arrived, by row id.

    As a Sequence it holds each row's document tokens. ``column``,
    ``column_names``, and ``take`` read the input columns the executor
    reads, as a value table does.

    Args:
        reader: Rust's InputReader over the input.
    """

    def __init__(self, reader):
        self.reader = reader
        self.documents = DocumentBatches()
        # each row's token count, in row order
        self.lengths = []
        # callables(pyarrow Table) handed each batch's columns as it arrives
        self.listeners = []
        self._schema = reader.schema
        self._tokens = []
        self._batches = []
        self._table = None

    @property
    def ended(self) -> bool:
        """Whether the input has ended and every row has arrived."""
        return self.reader.ended

    @property
    def tokens(self) -> Arrivals:
        """The rows' document tokens, as a scan's token store holds them."""
        return self

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

    def arrive(self, block: bool = False) -> np.ndarray | None:
        """Read the rows that have arrived and return their ids.

        Args:
            block: Wait until rows arrive or the input ends.

        Returns:
            The new rows' ids, as int64; None once the input has ended,
            or, without block, while no rows have arrived.
        """
        found = self.reader.next(block)
        if found is None:
            return None
        ids, tokens, ranges, columns = found
        self.documents.add(ArrowTokenDocuments(tokens, ranges))
        self.lengths.extend(pc.list_value_length(tokens).to_pylist())
        self._tokens.append(tokens)
        self._batches.append(columns)
        self._table = None
        for listener in self.listeners:
            listener(columns)
        return ids.to_numpy().astype(np.int64)

    def wait(self) -> None:
        """Read the input to its end."""
        while self.arrive(block=True) is not None:
            pass

    @property
    def table(self) -> pa.Table:
        """The input columns of every row arrived, one row per id."""
        if self._table is None:
            self._table = (pa.concat_tables(self._batches) if self._batches
                           else self._schema.empty_table())
        return self._table

    @property
    def column_names(self) -> list[str]:
        """The input columns the executor reads."""
        return self._schema.names

    @property
    def projected_columns(self) -> tuple[str, ...]:
        """The input columns the executor reads, as a scan's input names them."""
        return tuple(self._schema.names)

    @property
    def num_rows(self) -> int:
        """The rows arrived."""
        return len(self)

    def column(self, name: str) -> pa.ChunkedArray:
        """One input column's values for every row arrived."""
        return self.table.column(name)

    def take(self, indices) -> pa.Table:
        """The input columns of the rows at indices."""
        return self.table.take(indices)

    def first_documents_with(self, columns) -> pa.Array:
        """Each row's first row with its tokens and its values in columns."""
        return first_documents(
            pa.chunked_array(self._tokens, type=pa.large_list(pa.int32())), columns)

    def texts(self, column) -> ArrivedTexts:
        """The text a document column holds for each row arrived."""
        return ArrivedTexts(self, column)


class ArrivedTexts(Sequence):
    """The text a document column holds for each row of an input arrived.

    Args:
        arrivals: The input's Arrivals.
        column: The document column's expression.
    """

    def __init__(self, arrivals: Arrivals, column):
        self.arrivals = arrivals
        self.column = column
        self._values = None

    def __len__(self):
        return len(self.arrivals)

    def __getitem__(self, index):
        if self._values is None or len(self._values) != len(self.arrivals):
            self._values = self.column.evaluate(self.arrivals)
        return self._values[index]
