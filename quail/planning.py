"""Shared inputs and results for physical planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from quail.physical import PhysicalGraph
from quail.specs import DeviceSpec, ModelSpec


@dataclass(frozen=True)
class SupportResult:
    """Result of checking one backend configuration."""

    supported: bool
    reason: str | None = None

    @classmethod
    def accept(cls) -> "SupportResult":
        return cls(True)

    @classmethod
    def reject(cls, reason: str) -> "SupportResult":
        return cls(False, reason)


@dataclass(frozen=True)
class ModelRegion:
    """Connected logical model work planned by one backend."""

    logical_plan: Any


@dataclass(frozen=True)
class PlanningContext:
    """Inputs available to physical planners."""

    model: ModelSpec
    device: DeviceSpec
    gpu_count: int
    document_tokens: Mapping[str, Any]
    backend: str
    order: str | None = None


@dataclass(frozen=True)
class PhysicalCandidate:
    """One physical plan offered by a backend."""

    graph: PhysicalGraph | None
    plan: Any
    estimated_seconds: float
    counted_work: Any = None
    reason: str | None = None


class PhysicalOptimizerRule(Protocol):
    """Rewrite a complete typed physical graph."""

    name: str

    def rewrite(
        self,
        graph: PhysicalGraph,
        context: PlanningContext,
    ) -> PhysicalGraph | None: ...


class PhysicalPlanner(Protocol):
    """Produce physical candidates for one logical model region."""

    name: str

    def plan(
        self,
        region: ModelRegion,
        context: PlanningContext,
    ) -> tuple[PhysicalCandidate, ...]: ...


def apply_physical_rules(
    graph: PhysicalGraph,
    rules: tuple[PhysicalOptimizerRule, ...],
    context: PlanningContext,
) -> tuple[PhysicalGraph, tuple[str, ...]]:
    """Apply registered physical graph rules in order."""
    changed = []
    for rule in rules:
        replacement = rule.rewrite(graph, context)
        if replacement is not None and replacement != graph:
            replacement.validate()
            graph = replacement
            changed.append(rule.name)
    return graph, tuple(changed)
