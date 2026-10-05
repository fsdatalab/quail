"""Price a logical plan by the estimated seconds of its physical plan.

Cost-based logical rules compare candidate plans with these functions.
The filter_order and join_order rules decide each candidate's filter
orders, stage order, and anchors. build_physical_plan builds the
physical plan, and the label_scoring rule picks each classification's
scoring rule so its seconds are counted. Plans are memoized on the
logical planning context, keyed by the candidate's root with those
decisions cleared, so each candidate is built once per query.
"""

from __future__ import annotations

from quail.logical import LogicalPlan
from quail.planner.api import _apply_rules
from quail.planner.build import build_physical_plan
from quail.planner.logical_optimizer import apply_logical_rules
from quail.planner.physical_rules import LabelScoring
from quail.planner.plan import Refusal


def physical_plan(logical: LogicalPlan, context):
    """Return build_physical_plan's plan for a logical plan.

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
        physical = build_physical_plan(
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
