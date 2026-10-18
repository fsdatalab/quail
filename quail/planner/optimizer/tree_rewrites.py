"""Apply logical rules to every candidate before physical planning."""

from dataclasses import replace

from quail.logical import LogicalPlan
from quail.planner.optimizer.candidates import LogicalAlternative
from quail.planner.optimizer.context import PlanningContext
from quail.planner.optimizer.rules import LogicalRule

# Rules that undo each other must not loop forever.
MAX_PASSES = 4


def logical_alternatives(
    plan: LogicalPlan,
    rules: tuple[LogicalRule, ...],
    context: PlanningContext,
    *,
    max_passes: int = MAX_PASSES,
) -> tuple[LogicalAlternative, ...]:
    """Rewrite each candidate and deduplicate the returned trees in order.

    Each rule's returned roots replace its input candidate. Stop after a
    round leaves the roots unchanged, or after max_passes rounds.
    """
    candidates = [LogicalAlternative(plan)]
    for _ in range(max_passes):
        before = [candidate.plan.root for candidate in candidates]
        for rule in rules:
            if not isinstance(rule, LogicalRule):
                continue
            rewritten = []
            for candidate in candidates:
                root = candidate.plan.root
                roots = rule.rewrites(
                    root, replace(context, logical_plan=candidate.plan, settings={}))
                if not isinstance(roots, list) or not roots:
                    raise ValueError(
                        f"{rule.name}.rewrites() must return a nonempty list of roots")
                for replacement in roots:
                    if any(replacement == other.plan.root for other in rewritten):
                        continue
                    if replacement == root:
                        rewritten.append(candidate)
                        continue
                    plan = LogicalPlan(replacement)
                    plan.validate()
                    rewritten.append(LogicalAlternative(
                        plan, (*candidate.rules, rule.name)))
            candidates = rewritten
        if [candidate.plan.root for candidate in candidates] == before:
            break
    return tuple(candidates)
