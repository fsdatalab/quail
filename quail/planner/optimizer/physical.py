"""Executable graph refinement and the planner protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from quail.physical.base import PhysicalGraph
from quail.planner.optimizer.rules import OptimizerRule
from quail.planner.prompt_lowering import LoweredPrompt, lower_prompt
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


def lowered_prompt(context, prompt) -> LoweredPrompt:
    """Lower a prompt through a context's tokenizer, memoized on the context.

    Raises:
        ValueError: The context has no tokenizer.
    """
    key = ("lowered", prompt)
    if key not in context.memo:
        if context.tokenizer is None:
            raise ValueError("planning needs a tokenizer to lower prompts")
        context.memo[key] = lower_prompt(prompt, context.tokenizer)
    return context.memo[key]


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
    catalog: Any = None
    engine_config: Any = None
    # the plan's settings as the rules see them; a rule adds what the
    # executor needs for its decision, such as the KV retention
    # schedule, and the planner puts the result on the plan
    settings: dict = field(default_factory=dict, compare=False, repr=False)
    # results computed once per query, such as the plan's statistics,
    # shared with the tree rewrites that planned it
    memo: dict = field(default_factory=dict, compare=False, repr=False)

    def lowered(self, prompt) -> LoweredPrompt:
        """Return the prompt lowered with the session tokenizer, once per query."""
        return lowered_prompt(self, prompt)


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


def rewrite_graph(
    graph: PhysicalGraph,
    rules: tuple[OptimizerRule, ...],
    context: PlanningContext,
) -> tuple[PhysicalGraph, tuple[str, ...]]:
    """Apply rules targeting executable graphs in registration order."""
    changed = []
    for rule in rules:
        if rule.target != "physical":
            continue
        replacement = rule.rewrite(graph, context)
        if replacement is not None and replacement != graph:
            replacement.validate()
            graph = replacement
            changed.append(rule.name)
    return graph, tuple(changed)
