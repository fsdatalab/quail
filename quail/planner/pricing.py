"""Price a logical plan by the estimated seconds of its physical plan.

build_physical_plan builds the physical plan, choosing filter order and
join order and anchors, and the label_scoring rule picks each
classification's scoring rule so its seconds are counted. The Quail
backend prices its candidates this way, and a cost-based logical rule
can compare plans with the same functions. Plans are memoized on the
context, keyed by the logical plan's root, so each is built once per
query.
"""

from __future__ import annotations

from dataclasses import replace

from quail.logical import LogicalPlan
from quail.planner.api import apply_plan_rules
from quail.planner.build import build_physical_plan
from quail.planner.logical_optimizer import LogicalPlanningContext
from quail.planner.physical_rules import LabelScoring
from quail.planner.plan import Refusal


def physical_plan(logical: LogicalPlan, context):
    """Return a logical plan's physical plan with its classifications scored.

    The plan carries every classification's scoring rule; the other
    physical rules, which refine a chosen plan, are not applied.

    Args:
        logical: The logical plan to price.
        context: A PlanningContext, or a LogicalPlanningContext from a
            logical rule.

    Returns:
        A PhysicalPlan, or a Refusal when the plan cannot run.
    """
    if isinstance(context, LogicalPlanningContext):
        context = context.physical_context(logical)
    key = ("physical_plan", logical.root)
    if key not in context.memo:
        context = replace(context, logical_plan=logical)
        physical = build_physical_plan(
            logical, model=context.model, device=context.device,
            doc_tokens=context.document_tokens, gpus=context.gpu_count,
            order=context.order, pair_fractions=context.pair_fractions,
            context=context)
        if not isinstance(physical, Refusal):
            physical = apply_plan_rules(physical, (LabelScoring(),), context)
        context.memo[key] = physical
    return context.memo[key]


def estimated_seconds(logical: LogicalPlan, context) -> float | None:
    """Return a logical plan's estimated seconds, or None when refused."""
    physical = physical_plan(logical, context)
    return None if isinstance(physical, Refusal) else physical.estimated_seconds
