"""Coordinator nodes: Exchange, Recombine, Project, Aggregate, Sort, Limit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from quail.logical.expressions import packed_columns

from ..base import (
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    ValueType,
)


@dataclass(frozen=True)
class Exchange(PhysicalNode):
    """Route one alias's documents to the GPU that holds their KV.

    Present only in plans for several GPUs. On one GPU it passes the
    ids through.
    """

    anchor: str = ""

    type_name: ClassVar[str] = "quail.exchange"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            f"ids:{self.anchor}",
            ValueType.DOCUMENT_IDS,
            schema=(self.anchor,),
        ),)

    def attributes(self) -> dict:
        return {"anchor": self.anchor}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(node_id=node_id, inputs=inputs, anchor=attributes["anchor"])


@dataclass(frozen=True)
class Recombine(PhysicalNode):
    """Combine answer relations by exact document ids."""

    alias_order: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.recombine"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            "tuples",
            ValueType.ROWS,
            schema=self.alias_order,
        ),)

    def attributes(self) -> dict:
        return {"alias_order": list(self.alias_order)}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            alias_order=tuple(attributes["alias_order"]),
        )


@dataclass(frozen=True)
class Project(PhysicalNode):
    """Select the requested result columns.

    Attributes:
        columns: The selected columns, in order.
        structs: (struct column, ((field, member column), ...)) pairs.
            Each struct column takes the place of its first member and
            holds one field per member; the members leave the result.
    """

    columns: tuple[str, ...] = ()
    structs: tuple = ()

    type_name: ClassVar[str] = "quail.project"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def result_columns(self) -> tuple[str, ...]:
        """Return the output column names, with struct members packed."""
        return packed_columns(self.columns, self.structs)

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            "rows",
            ValueType.ROWS,
            schema=self.result_columns,
        ),)

    def attributes(self) -> dict:
        return {"columns": list(self.columns),
                "structs": [[name, [list(field) for field in fields]]
                            for name, fields in self.structs]}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            columns=tuple(attributes["columns"]),
            structs=tuple(
                (name, tuple(tuple(field) for field in fields))
                for name, fields in attributes.get("structs", ())),
        )


@dataclass(frozen=True)
class Aggregate(PhysicalNode):
    """Group the result rows and compute aggregates on the coordinator.

    ``keys`` name input columns. ``aggregates`` are (name, function,
    argument) triples, the argument an input column or None for
    ``count(*)``. ``having`` are (name, comparison, value) tests over
    the aggregates. ``columns`` are the output columns, keys and
    aggregate names in result order.
    """

    keys: tuple[str, ...] = ()
    aggregates: tuple[tuple[str, str, str | None], ...] = ()
    having: tuple[tuple[str, str, float], ...] = ()
    columns: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.aggregate"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort("rows", ValueType.ROWS, schema=self.columns),)

    def attributes(self) -> dict:
        return {
            "keys": list(self.keys),
            "aggregates": [list(aggregate) for aggregate in self.aggregates],
            "having": [list(test) for test in self.having],
            "columns": list(self.columns),
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            keys=tuple(attributes["keys"]),
            aggregates=tuple(
                (str(name), str(function),
                 None if argument is None else str(argument))
                for name, function, argument in attributes["aggregates"]),
            having=tuple((str(name), str(comparison), value)
                         for name, comparison, value
                         in attributes.get("having", ())),
            columns=tuple(attributes["columns"]),
        )


@dataclass(frozen=True)
class Sort(PhysicalNode):
    """Order, deduplicate, and bound the result rows on the coordinator.

    ``keys`` are (column, descending, nulls_first) triples over the
    input columns. ``columns`` are the output columns; an input column
    not among them is read by a key only. Distinct runs before the
    sort, and ``offset`` and ``fetch`` after it.
    """

    keys: tuple[tuple[str, bool, bool], ...] = ()
    columns: tuple[str, ...] = ()
    distinct: bool = False
    offset: int = 0
    fetch: int | None = None

    type_name: ClassVar[str] = "quail.sort"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort("rows", ValueType.ROWS, schema=self.columns),)

    def attributes(self) -> dict:
        return {
            "keys": [list(key) for key in self.keys],
            "columns": list(self.columns),
            "distinct": self.distinct,
            "offset": self.offset,
            "fetch": self.fetch,
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            keys=tuple((str(column), bool(descending), bool(nulls_first))
                       for column, descending, nulls_first
                       in attributes["keys"]),
            columns=tuple(attributes["columns"]),
            distinct=bool(attributes.get("distinct", False)),
            offset=int(attributes.get("offset", 0)),
            fetch=(None if attributes.get("fetch") is None
                   else int(attributes["fetch"])),
        )


@dataclass(frozen=True)
class Limit(PhysicalNode):
    """Stop result output after a fixed number of rows."""

    count: int = 0

    type_name: ClassVar[str] = "quail.limit"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort("rows", ValueType.ROWS),)

    def attributes(self) -> dict:
        return {"count": self.count}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(node_id=node_id, inputs=inputs, count=int(attributes["count"]))
