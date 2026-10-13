"""The Scan node that reads one tokenized document input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..base import (
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    ValueType,
)


@dataclass(frozen=True)
class Scan(PhysicalNode):
    """Read one tokenized document input supplied by the coordinator.

    ``predicates`` are the regular predicates, as (column, comparison, value)
    triples; the runtime keeps the documents that pass every one and
    reads nothing else for the rest. ``expected_docs`` is the planner's
    estimate of how many pass, from a sample; None without tests.
    """

    alias: str = ""
    input_id: str = ""
    n_docs: int = 0
    total_tokens: int = 0
    shard_ranges: tuple[tuple[int, int], ...] = ()
    shard_token_loads: tuple[int, ...] = ()
    predicates: tuple[tuple[str, str, Any], ...] = ()
    expected_docs: float | None = None

    type_name: ClassVar[str] = "quail.scan"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    def __post_init__(self) -> None:
        if len(self.shard_ranges) != len(self.shard_token_loads):
            raise ValueError(
                "document shard ranges and token loads must have equal size"
            )
        expected = 0
        for start, stop in self.shard_ranges:
            if start != expected or stop < start:
                raise ValueError(
                    "document shard ranges must be ordered and contiguous"
                )
            expected = stop
        if self.shard_ranges and expected != self.n_docs:
            raise ValueError(
                "document shard ranges must cover every document"
            )

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            f"ids:{self.alias}",
            ValueType.DOCUMENT_IDS,
            schema=(self.alias,),
        ),)

    def attributes(self) -> dict:
        return {
            "alias": self.alias,
            "input_id": self.input_id,
            "n_docs": self.n_docs,
            "total_tokens": self.total_tokens,
            "shard_token_loads": list(self.shard_token_loads),
            "shard_ranges": [list(bounds) for bounds in self.shard_ranges],
            "predicates": [
                [column, comparison,
                 list(value) if isinstance(value, tuple) else value]
                for column, comparison, value in self.predicates],
            "expected_docs": self.expected_docs,
        }

    @property
    def shards(self) -> tuple[range, ...]:
        """Return each worker's contiguous document range."""
        return tuple(range(start, stop) for start, stop in self.shard_ranges)

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            "alias": self.alias,
            "input_id": self.input_id,
            "n_docs": self.n_docs,
            "total_tokens": self.total_tokens,
            "shard_token_loads": list(self.shard_token_loads),
            "shard_docs": [stop - start
                           for start, stop in self.shard_ranges],
            "predicates": [
                f"{column} {comparison}" + ("" if value is None else f" {value!r}")
                for column, comparison, value in self.predicates],
            "expected_docs": self.expected_docs,
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            alias=attributes["alias"],
            input_id=attributes["input_id"],
            n_docs=int(attributes["n_docs"]),
            total_tokens=int(attributes["total_tokens"]),
            shard_ranges=tuple(
                (int(start), int(stop))
                for start, stop in attributes["shard_ranges"]
            ),
            shard_token_loads=tuple(attributes["shard_token_loads"]),
            predicates=tuple(
                (str(column), str(comparison),
                 tuple(value) if isinstance(value, list) else value)
                for column, comparison, value
                in attributes.get("predicates", ())),
            expected_docs=(None if attributes.get("expected_docs") is None
                           else float(attributes["expected_docs"])),
        )
