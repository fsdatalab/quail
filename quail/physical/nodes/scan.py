"""The Scan node that reads one tokenized document input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from quail.logical import RegularPredicate

from ..base import (
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    ValueType,
)


@dataclass(frozen=True)
class Scan(PhysicalNode):
    """Read one tokenized document input supplied by the coordinator.

    ``predicates`` are the regular predicates; the runtime keeps the
    documents that pass every one and reads nothing else for the rest.
    ``expected_rows`` is the planner's estimate of how many pass, from a
    sample; None without tests.
    """

    alias: str = ""
    input_id: str = ""
    num_documents: int = 0
    total_tokens: int = 0
    shard_ranges: tuple[tuple[int, int], ...] = ()
    shard_token_loads: tuple[int, ...] = ()
    predicates: tuple[RegularPredicate, ...] = ()
    expected_rows: float | None = None

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
        if self.shard_ranges and expected != self.num_documents:
            raise ValueError(
                "document shard ranges must cover every document"
            )

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            f"rows:{self.alias}",
            ValueType.ROWS,
            schema=(self.alias,),
        ),)

    @property
    def shards(self) -> tuple[range, ...]:
        """Return each worker's contiguous document range."""
        return tuple(range(start, stop) for start, stop in self.shard_ranges)

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            "alias": self.alias,
            "input_id": self.input_id,
            "num_documents": self.num_documents,
            "total_tokens": self.total_tokens,
            "shard_token_loads": list(self.shard_token_loads),
            "shard_docs": [stop - start
                           for start, stop in self.shard_ranges],
            "predicates": [str(predicate) for predicate in self.predicates],
            "expected_rows": self.expected_rows,
        }
