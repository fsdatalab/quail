"""Model backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from quail.physical import PhysicalNode
from quail.planning import (
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
    """Planner and execution factory for one model backend."""

    name: str

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
    ) -> Sequence[PhysicalCandidate]: ...

    def start(self, context: GpuContext) -> ModelExecution: ...

    def execute_request(
        self,
        context: BackendExecutionContext,
    ) -> Any:
        """Execute one request inside a compute process."""
        ...
