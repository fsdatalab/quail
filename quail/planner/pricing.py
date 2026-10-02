"""Price a logical plan as the Quail planner would run it.

A cost-based logical rule compares candidate plans by the estimated
seconds of the physical plan each would get: the Quail planner's
plan, with the label_scoring rule applied so every classification's
scoring rule is counted. The plans are memoized on the logical
planning context by their root, so two rules pricing the same
candidate, or one rule pricing it on a later pass, build it once.
"""

from __future__ import annotations

from quail.logical import LogicalPlan
from quail.planner.decide import _apply_rules, plan_quail
from quail.planner.physical_rules import LabelScoring
from quail.planner.plan import Refusal


def physical_plan(logical: LogicalPlan, context):
    """Return the Quail planner's physical plan of a logical plan.

    The plan carries every classification's scoring rule; the other
    physical rules, which refine a finished plan, are not applied.

    Args:
        logical: The logical plan to price.
        context: The LogicalPlanningContext with the statistics.

    Returns:
        A PhysicalPlan, or a Refusal when the plan cannot run.
    """
    key = ("physical_plan", logical.root)
    if key not in context.memo:
        physical_context = context.physical_context(logical)
        physical = plan_quail(
            logical, model=context.model, device=context.device,
            doc_tokens=context.document_tokens, gpus=context.gpu_count,
            order=context.order, pair_fractions=context.pair_fractions,
            context=physical_context)
        if not isinstance(physical, Refusal):
            physical = _apply_rules(
                physical, (LabelScoring(),), physical_context, "")
        context.memo[key] = physical
    return context.memo[key]


def estimated_seconds(logical: LogicalPlan, context) -> float | None:
    """Return a logical plan's estimated seconds, or None when refused."""
    physical = physical_plan(logical, context)
    return None if isinstance(physical, Refusal) else physical.estimated_seconds
