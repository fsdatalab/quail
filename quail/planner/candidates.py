"""Candidate plans and the logical alternatives they lower."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quail.physical.base import PhysicalGraph


@dataclass(frozen=True)
class ModelRegion:
    """Connected logical model work planned by one backend."""

    logical_plan: str  # as Rust's JSON


@dataclass(frozen=True)
class LogicalAlternative:
    """A logical plan and the rules that produced it."""

    plan: str  # as Rust's JSON
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
            classifications moved above the joins. Physical rules
            read it before this candidate is compared with the others.
        rewrites: Alternative rules that produced this candidate.
    """

    graph: PhysicalGraph | None
    plan: Any
    estimated_seconds: float
    logical_plan: str | None = None
    rewrites: tuple[str, ...] = ()
