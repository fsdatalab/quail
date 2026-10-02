"""Logical plan rewriting: whole-plan rules, run until nothing changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from quail.logical import LogicalNode, LogicalPlan
from quail.planner.physical_optimizer import PlanningContext

# Rounds over the rule list before giving up on convergence. Two rules
# that undo each other would otherwise loop forever.
MAX_PASSES = 4


@dataclass(frozen=True)
class LogicalPlanningContext:
    """Session values and statistics available to logical optimizer rules.

    A cost-based rule prices a plan with the model and device specs,
    each scanned table's document token counts (exact when the column
    is tokenized, else estimated from a sample), the fraction of each
    join's cross product its equality conditions keep, and the
    tokenizer for prompts it builds. Selectivity estimates are the
    predicates' own: a predicate written without one is priced with
    the default selectivity.

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
        remarks: Advice the rules leave for the physical plan's
            remarks, such as a cheaper plan a forced setting ruled out.
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
    remarks: list = field(default_factory=list, compare=False, repr=False)
    memo: dict = field(default_factory=dict, compare=False, repr=False)

    def physical_context(self, logical_plan=None) -> PlanningContext:
        """Return the physical planning context with the same inputs."""
        return PlanningContext(
            model=self.model, device=self.device, gpu_count=self.gpu_count,
            document_tokens=self.document_tokens, backend=self.backend,
            order=self.order, canvas_draws=self.canvas_draws,
            attention=self.attention, tokenizer=self.tokenizer,
            pair_fractions=dict(self.pair_fractions),
            logical_plan=logical_plan)


class LogicalOptimizerRule(Protocol):
    """Rewrite a whole logical plan without changing query meaning.

    A rule walks the plan itself, in the direction its rewrite needs:
    from the root toward the scans when it pushes information down,
    from the scans toward the root when it simplifies a node once its
    children are final. It returns the new root, or None when the plan
    is unchanged.
    """

    name: str

    def rewrite(
        self,
        root: LogicalNode,
        context: LogicalPlanningContext,
    ) -> LogicalNode | None: ...


def apply_logical_rules(
    plan: LogicalPlan,
    rules: tuple[LogicalOptimizerRule, ...],
    context: LogicalPlanningContext,
    *,
    max_passes: int = MAX_PASSES,
) -> tuple[LogicalPlan, tuple[str, ...]]:
    """Apply the rules in order, repeating until no rule changes the plan.

    Each round offers the current plan to every rule in registration
    order. Rounds continue until one round changes nothing, so a rewrite
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
            replacement = rule.rewrite(root, context)
            if replacement is not None and replacement != root:
                root = replacement
                changed.append(rule.name)
        if root == before:
            break
    optimized = LogicalPlan(root)
    optimized.validate()
    return optimized, tuple(changed)
