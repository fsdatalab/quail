"""Model backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from quail.physical import PhysicalNode
from quail.planner.optimizer.physical import (
    ModelRegion,
    PhysicalCandidate,
    PlanningContext,
    SupportResult,
)
from quail.specs import DeviceSpec, ModelSpec


@dataclass(frozen=True)
class GpuContext:
    """Configuration for one GPU executor."""

    gpu_index: int
    gpu_count: int
    model: ModelSpec
    device: DeviceSpec
    query_settings: Mapping[str, Any]


@dataclass(frozen=True)
class BackendExecutionContext:
    """Values available to a backend inside a compute process."""

    request: Any
    graph: Any
    registry: Any
    gpu_count: int
    runtime_state: dict[Any, Any]


class ModelExecution(Protocol):
    """Shared model state for one query on one GPU executor."""

    def execute(
        self,
        node: PhysicalNode,
        inputs: Mapping[str, Any],
    ) -> Any: ...


class ModelBackend(Protocol):
    """Planner and execution factory for one model backend.

    Attributes:
        name: The registered backend name.
        reads_document_text: Whether the backend reads each scanned
            document as text, so the scan carries its text column.
        locates_answers_in_text: Whether the backend locates extracted
            answers by character offsets, so tokenization keeps them.
    """

    name: str
    reads_document_text: bool
    locates_answers_in_text: bool

    def supports(
        self,
        model: ModelSpec,
        device: DeviceSpec,
        gpu_count: int,
    ) -> SupportResult: ...

    def plan(
        self,
        region: ModelRegion,
        context: PlanningContext,
    ) -> Sequence[PhysicalCandidate]:
        """Lower one logical alternative; the planner applies optimizer rules."""
        ...

    def start(self, context: GpuContext) -> ModelExecution: ...

    def execute_request(
        self,
        context: BackendExecutionContext,
    ) -> Any:
        """Execute one request inside a compute process."""
        ...
