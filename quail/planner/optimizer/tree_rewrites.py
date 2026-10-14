"""Simplify query trees and collect alternatives before physical planning."""

from dataclasses import replace

from quail.logical import LogicalPlan
from quail.planner.optimizer.candidates import LogicalAlternative
from quail.planner.optimizer.context import PlanningContext
from quail.planner.optimizer.rules import LogicalRule

# Rules that undo each other must not loop forever.
MAX_PASSES = 4


def simplify_plan(
    plan: LogicalPlan,
    rules: tuple[LogicalRule, ...],
    context: PlanningContext,
    *,
    max_passes: int = MAX_PASSES,
) -> tuple[LogicalPlan, tuple[str, ...]]:
    """Apply logical rewrites until no rule changes the plan.

    Each round offers the current plan to every logical rule in registration
    order. Stop after an unchanged round or max_passes rounds. Alternative
    trees are collected separately by logical_alternatives.

    Returns:
        The rewritten plan and the name of each rule that changed it,
        one entry per change, in the order the changes happened.
    """
    changed = []
    root = plan.root
    for _ in range(max_passes):
        before = root
        for rule in rules:
            if not isinstance(rule, LogicalRule):
                continue
            replacement = rule.rewrite(
                root, replace(context, logical_plan=LogicalPlan(root), settings={}))
            if replacement is not None and replacement != root:
                root = replacement
                changed.append(rule.name)
        if root == before:
            break
    optimized = LogicalPlan(root)
    optimized.validate()
    return optimized, tuple(changed)


def logical_alternatives(plan, rules, context, *, max_passes=MAX_PASSES) -> tuple:
    """Collect alternative query trees, retaining the original first.

    Equivalent roots are deduplicated. Rules run over every known alternative
    for at most max_passes rounds, stopping when no new root is produced.
    """
    alternatives = [LogicalAlternative(plan)]
    for _ in range(max_passes):
        before = len(alternatives)
        for rule in rules:
            if not isinstance(rule, LogicalRule):
                continue
            for candidate in tuple(alternatives):
                roots = rule.alternatives(
                    candidate.plan.root,
                    replace(context, logical_plan=candidate.plan, settings={}))
                for root in roots:
                    if any(root == other.plan.root for other in alternatives):
                        continue
                    rewritten = LogicalPlan(root)
                    rewritten.validate()
                    alternatives.append(LogicalAlternative(
                        rewritten, (*candidate.rules, rule.name)))
        if len(alternatives) == before:
            break
    return tuple(alternatives)
