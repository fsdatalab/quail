"""Candidate plans and the interface used by backend planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from quail.logical import LogicalPlan
from quail.physical.base import PhysicalGraph
from quail.planner.optimizer.context import PlanningContext


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
class LogicalAlternative:
    """A logical plan and the rules that produced it."""

    plan: LogicalPlan
    rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhysicalCandidate:
    """One physical plan offered by a backend.

    Attributes:
        graph: The plan's graph; None for a refusal.
        plan: A PhysicalPlan or a Refusal.
        estimated_seconds: The current cost estimate; infinite for a refusal.
            The planner refines backend estimates before selection.
        logical_plan: The logical plan the candidate lowers, when it
            differs from the region's, such as one with its
            classifications moved above the joins. Rules targeting the graph
            read it before this candidate is compared with the others.
        rewrites: Alternative rules that produced this candidate.
    """

    graph: PhysicalGraph | None
    plan: Any
    estimated_seconds: float
    logical_plan: Any = None
    rewrites: tuple[str, ...] = ()


class PhysicalPlanner(Protocol):
    """Lower one logical alternative into physical candidates.

    The common planner refines each graph and compares the resulting
    estimates. Implementations do not run cost-based tree rewrites.
    """

    name: str

    def plan(
        self,
        region: ModelRegion,
        context: PlanningContext,
    ) -> tuple[PhysicalCandidate, ...]: ...
