"""Execution stages with dependencies on materialized query results."""

from dataclasses import dataclass, field
from typing import ClassVar

from quail.physical.base import OutputPort, PhysicalGraph, PhysicalNode, ValueType


@dataclass(frozen=True)
class QueryStage(PhysicalNode):
    """Execute a physical subgraph after its input relations are available.

    The graph is an estimate until the inputs have been materialized.
    Execution replaces it with the graph planned on their actual token
    lengths. A named stage materializes its output for all its readers.
    """

    name: str = ""
    graph: PhysicalGraph | None = None
    settings: dict = field(default_factory=dict)
    estimates: dict = field(default_factory=dict)
    estimated_seconds: float = 0.0
    columns: tuple[str, ...] = ()
    assumptions: tuple[tuple[str, str, int, int], ...] = ()

    type_name: ClassVar[str] = "quail.query_stage"
    runtime_key: ClassVar[str] = type_name

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort("rows", ValueType.ROWS, schema=self.columns),)

    def explain_fields(self) -> dict:
        return {"name": self.name or "query", "materialize": bool(self.name)}
