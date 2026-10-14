"""Simplify query trees and retain cost-based alternatives."""

from quail.logical import LogicalPlan
from quail.planner.optimizer.candidates import LogicalAlternative
from quail.planner.optimizer.context import LogicalPlanningContext
from quail.planner.optimizer.rules import OptimizerRule

# Rules that undo each other must not loop forever.
MAX_PASSES = 4


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
