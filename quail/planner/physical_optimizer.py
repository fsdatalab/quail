"""Physical optimizer rules and the planner protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from quail.physical.base import PhysicalGraph
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
    # the most noise draws a diffusion model averages per letters read
    # and per one-table AI.SCORE
    canvas_draws: int = 4
    # an attention path forced for every filter and join; None lets
    # the planner choose per node
    attention: str | None = None
    tokenizer: Callable[[str], Any] | None = None
    # join written position -> its equality pairs as a fraction of
    # the cross product; joins without conditions are absent
    pair_fractions: Mapping[int, float] = field(default_factory=dict)
    # alias -> the fraction of its documents the regular predicates are
    # expected to keep; aliases without tests are absent
    scan_fractions: Mapping[str, float] = field(default_factory=dict)
    # the logical plan the graph was planned from; None when the rules
    # run again over a finished plan
    logical_plan: Any = None
    # the plan's settings as the rules see them; a rule adds what the
    # executor needs for its decision, such as the KV retention
    # schedule, and the planner puts the result on the plan
    settings: dict = field(default_factory=dict, compare=False, repr=False)
    # results computed once per query, such as the plan's statistics,
    # shared with the logical rules that planned it
    memo: dict = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class PhysicalCandidate:
    """One physical plan offered by a backend.

    Attributes:
        graph: The plan's graph; None for a refusal.
        plan: A PhysicalPlan or a Refusal.
        estimated_seconds: The plan's estimated seconds; infinite for
            a refusal.
        logical_plan: The logical plan the candidate lowers, when it
            differs from the region's, such as one with its
            classifications moved above the joins. The physical rules
            read it when this candidate is chosen.
    """

    graph: PhysicalGraph | None
    plan: Any
    estimated_seconds: float
    logical_plan: Any = None


class PhysicalOptimizerRule(Protocol):
    """Rewrite a complete typed physical graph.

    A rule may set ``cost_based = True`` when it prices the plans it
    chooses between; explain lists every other rule as heuristic.
    """

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
