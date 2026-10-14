"""Simplify query trees and retain cost-based alternatives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from quail.logical import LogicalPlan
from quail.planner.optimizer.physical import (
    PlanningContext,
    lowered_prompt,
)
from quail.planner.optimizer.rules import OptimizerRule

# Rounds over the rule list before giving up on convergence. Two rules
# that undo each other would otherwise loop forever.
MAX_PASSES = 4


@dataclass(frozen=True)
class LogicalPlanningContext:
    """Session values and statistics available to rules that rewrite query trees.

    Document token counts are exact when the column is tokenized, else
    estimated from a sample. A cost-based rule prices a predicate
    written without a selectivity with the default selectivity.

    Attributes:
        catalog: The session catalog.
        engine_config: The session's EngineConfig.
        model: Model spec; None when no cost-based rule will run.
        device: Device spec; None when no cost-based rule will run.
        gpu_count: GPU count; one model copy runs per GPU.
        document_tokens: alias -> per-document token counts.
        backend: Registered model backend name.
        order: Stage order rule, 'by_cost' or 'as_written'; None picks
            the default rule.
        canvas_draws: Maximum diffusion draws per classification.
        attention: An attention path forced for every filter and join.
        tokenizer: Callable (text -> token list) for prompts.
        pair_fractions: join written position -> the fraction of the
            cross product its equality conditions keep.
        scan_fractions: alias -> the fraction of its documents the
            regular predicates are expected to keep.
        memo: Results a rule computed for one plan root, so a later
            pass or another rule pricing the same root reuses them.
    """

    catalog: Any
    engine_config: Any
    model: Any = None
    device: Any = None
    gpu_count: int = 1
    document_tokens: Mapping[str, Any] = field(default_factory=dict)
    backend: str = "quail"
    order: str | None = None
    canvas_draws: int = 4
    attention: str | None = None
    tokenizer: Callable[[str], Any] | None = None
    pair_fractions: Mapping[int, float] = field(default_factory=dict)
    scan_fractions: Mapping[str, float] = field(default_factory=dict)
    memo: dict = field(default_factory=dict, compare=False, repr=False)

    def lowered(self, prompt):
        """Return the prompt lowered with the session tokenizer, once per query."""
        return lowered_prompt(self, prompt)

    def physical_context(self, logical_plan=None) -> PlanningContext:
        """Return the physical planning context with the same inputs."""
        return PlanningContext(
            model=self.model, device=self.device, gpu_count=self.gpu_count,
            document_tokens=self.document_tokens, backend=self.backend,
            order=self.order, canvas_draws=self.canvas_draws,
            attention=self.attention, tokenizer=self.tokenizer,
            pair_fractions=dict(self.pair_fractions),
            scan_fractions=dict(self.scan_fractions),
            logical_plan=logical_plan, memo=self.memo,
            catalog=self.catalog, engine_config=self.engine_config)

    @classmethod
    def from_physical(cls, context: PlanningContext) -> LogicalPlanningContext:
        """Return a query tree context with the current search inputs."""
        return cls(
            catalog=context.catalog, engine_config=context.engine_config,
            model=context.model, device=context.device, gpu_count=context.gpu_count,
            document_tokens=context.document_tokens, backend=context.backend,
            order=context.order, canvas_draws=context.canvas_draws,
            attention=context.attention, tokenizer=context.tokenizer,
            pair_fractions=context.pair_fractions,
            scan_fractions=context.scan_fractions, memo=context.memo)


def simplify_plan(
    plan: LogicalPlan,
    rules: tuple[OptimizerRule, ...],
    context: LogicalPlanningContext,
    *,
    max_passes: int = MAX_PASSES,
) -> tuple[LogicalPlan, tuple[str, ...]]:
    """Apply universal rules until no rule changes the plan.

    Cost-based rules are deferred to search. Each round offers the current
    plan to every universal rule in registration order. Rounds continue
    until one round changes nothing, so a rewrite
    that enables a later rule is not missed, or until `max_passes`
    rounds have run.

    Returns:
        The rewritten plan and the name of each rule that changed it,
        one entry per change, in the order the changes happened.
    """
    changed = []
    root = plan.root
    for _ in range(max_passes):
        before = root
        for rule in rules:
            if rule.target != "logical" or rule.cost_based:
                continue
            replacement = rule.rewrite(root, context)
            if replacement is not None and replacement != root:
                root = replacement
                changed.append(rule.name)
        if root == before:
            break
    optimized = LogicalPlan(root)
    optimized.validate()
    return optimized, tuple(changed)


@dataclass(frozen=True)
class LogicalAlternative:
    """A logical plan and the rules that produced it."""

    plan: LogicalPlan
    rules: tuple[str, ...] = ()


def logical_alternatives(plan, rules, context, *, max_passes=MAX_PASSES) -> tuple:
    """Apply cost-based tree rewrites, retaining the original first.

    Equivalent roots are deduplicated. Rules run over every known alternative
    for at most max_passes rounds, stopping when no new root is produced.
    """
    alternatives = [LogicalAlternative(plan)]
    for _ in range(max_passes):
        before = len(alternatives)
        for rule in rules:
            if rule.target != "logical" or not rule.cost_based:
                continue
            for candidate in tuple(alternatives):
                root = rule.rewrite(candidate.plan.root, context)
                if root is None or any(
                        root == other.plan.root for other in alternatives):
                    continue
                rewritten = LogicalPlan(root)
                rewritten.validate()
                alternatives.append(LogicalAlternative(
                    rewritten, (*candidate.rules, rule.name)))
        if len(alternatives) == before:
            break
    return tuple(alternatives)
