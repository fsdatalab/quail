"""Apply graph rules and update each candidate's settings and cost estimates."""

from dataclasses import replace

from quail.physical.base import PhysicalGraph
from quail.planner.optimizer.context import PlanningContext
from quail.planner.optimizer.rules import OptimizerRule
from quail.planner.plan import inference_seconds
from quail.planner.validation import ClassifyRefusedError


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


def apply_plan_rules(plan, rules, context):
    """Refine a plan by applying its registered rules in order.

    The rules see the plan's settings on the context and may add to
    them. Changes to inference estimates update both the plan total and
    its search cost, and the node estimates are refreshed for explain.

    Args:
        plan: A PhysicalPlan.
        rules: Registered rules in order; only graph targets apply.
        context: The PlanningContext.

    Returns:
        The rewritten PhysicalPlan, or a Refusal when a classification
        no rule can score refuses the plan.
    """
    context = replace(context, settings=dict(plan.settings))
    try:
        graph, changed = rewrite_graph(plan.graph, rules, context)
    except ClassifyRefusedError as refused:
        return refused.refusal()
    if not changed and context.settings == dict(plan.settings):
        return plan
    difference = inference_seconds(graph.nodes) - inference_seconds(plan.nodes)
    seconds = plan.estimated_seconds + difference
    if "search_seconds" in plan.settings:
        context.settings["search_seconds"] = (
            plan.settings["search_seconds"] + difference)
    return replace(
        plan, nodes=graph.nodes, root=graph.root, estimated_seconds=seconds,
        settings=context.settings,
        estimates=plan.estimator(graph) if plan.estimator else plan.estimates)
