"""Generate equivalent logical plans for physical cost comparison."""

from dataclasses import dataclass
from typing import Protocol

from quail.logical import LogicalNode, LogicalPlan
from quail.planner.optimizer.physical import PlanningContext


@dataclass(frozen=True)
class LogicalAlternative:
    """A logical plan and the rules that produced it."""

    plan: LogicalPlan
    rules: tuple[str, ...] = ()


class LogicalAlternativeRule(Protocol):
    """Offer equivalent logical roots without discarding the original."""

    name: str

    def alternatives(self, root: LogicalNode,
                     context: PlanningContext) -> tuple[LogicalNode, ...]: ...


def logical_alternatives(plan, rules, context, *, max_passes=4) -> tuple:
    """Expand logical alternatives, retaining the original first.

    Equivalent roots are deduplicated. Rules run over every known alternative
    for at most max_passes rounds, stopping when no new root is produced.
    """
    alternatives = [LogicalAlternative(plan)]
    for _ in range(max_passes):
        before = len(alternatives)
        for rule in rules:
            for candidate in tuple(alternatives):
                for root in rule.alternatives(candidate.plan.root, context):
                    if any(root == other.plan.root for other in alternatives):
                        continue
                    rewritten = LogicalPlan(root)
                    rewritten.validate()
                    alternatives.append(LogicalAlternative(
                        rewritten, (*candidate.rules, rule.name)))
        if len(alternatives) == before:
            break
    return tuple(alternatives)
