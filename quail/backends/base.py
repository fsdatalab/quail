"""Model backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

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
class QueryPreparationContext:
    """Client values available while a backend builds its request."""

    query: Any
    plan: Any
    scans: Sequence[Any]
    filters: Mapping[str, Any]
    joins: Sequence[Any]

    def plan_envelope(self, *, include_runtime_data: bool = True) -> dict:
        """Encode the plan with every required extension module."""
        registry = self.query.session.registry
        return self.plan.to_envelope(
            registry.codecs,
            extension_modules=registry.extension_modules,
            include_runtime_data=include_runtime_data,
        )


@dataclass(frozen=True)
class RemoteExecutionContext:
    """Values available to a backend inside the compute process."""

    payload: Mapping[str, Any]
    graph: Any
    registry: Any
    gpu_count: int
    quail_driver: Callable[[], Mapping[str, Any]]

    def run_quail(self) -> Mapping[str, Any]:
        """Run the built in Quail execution path."""
        return self.quail_driver()


@dataclass(frozen=True)
class ResultAssemblyContext:
    """Client values available while a backend builds a query result."""

    query: Any
    plan: Any
    scans: Sequence[Any]
    filters: Mapping[str, Any]
    joins: Sequence[Any]
    output: Mapping[str, Any]
    coordinator_wall_s: float


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

    def prepare(self, context: QueryPreparationContext) -> Mapping[str, Any]:
        """Build the request sent through the compute provider."""
        ...

    def execute_remote(
        self,
        context: RemoteExecutionContext,
    ) -> Mapping[str, Any]:
        """Execute one prepared request inside the compute process."""
        ...

    def assemble(self, context: ResultAssemblyContext) -> Any:
        """Build the public query result from the remote output."""
        ...
