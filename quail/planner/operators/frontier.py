"""Where a table's documents stand after an operator: the value lowering passes on."""

from __future__ import annotations

from dataclasses import dataclass

from quail.physical import PortRef


@dataclass(frozen=True)
class Frontier:
    """The output of a table's latest operator, as the next operator reads it.

    Each operator over a table takes the frontier before it and returns
    the one after it, so an operator's input is always the output of
    the operator before it.

    Attributes:
        ids: The port carrying the documents' ids.
        rows: The port carrying the documents' rows, when the operator
            has them; None after an operator that writes ids only.
    """

    ids: PortRef
    rows: PortRef | None = None

    @property
    def source(self) -> PortRef:
        """The port the next operator reads: the rows, else the ids."""
        return self.ids if self.rows is None else self.rows
