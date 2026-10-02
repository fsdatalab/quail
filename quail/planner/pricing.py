"""Price a logical plan as the Quail planner would run it.

A cost-based logical rule compares candidate plans by the estimated
seconds of the physical plan each would get: the filter_order and
join_order rules decide the candidate's filter orders, stage order,
and anchors, the Quail planner builds the physical plan, and the
label_scoring rule picks each classification's scoring rule so its
seconds are counted. The plans are memoized on the logical planning
context by the candidate's root with those decisions cleared, so two
rules pricing the same candidate, or one rule pricing it on a later
pass, build it once.
"""

from __future__ import annotations

from quail.logical import LogicalPlan
from quail.planner.decide import _apply_rules, plan_quail
from quail.planner.logical_optimizer import apply_logical_rules
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
    from quail.planner.logical_rules import FilterOrder, JoinOrder, undecided

    key = ("physical_plan", undecided(logical.root))
    if key not in context.memo:
        decided, _ = apply_logical_rules(
            logical, (FilterOrder(), JoinOrder()), context)
        physical_context = context.physical_context(decided)
        physical = plan_quail(
            decided, model=context.model, device=context.device,
            doc_tokens=context.document_tokens, gpus=context.gpu_count,
            order=context.order, pair_fractions=context.pair_fractions,
            context=physical_context)
        if not isinstance(physical, Refusal):
            physical = _apply_rules(
                physical, (LabelScoring(),), physical_context)
        context.memo[key] = physical
    return context.memo[key]


def estimated_seconds(logical: LogicalPlan, context) -> float | None:
    """Return a logical plan's estimated seconds, or None when refused."""
    physical = physical_plan(logical, context)
    return None if isinstance(physical, Refusal) else physical.estimated_seconds
