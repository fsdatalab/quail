"""Logical plan rewriting: whole-plan rules, run until nothing changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from quail.logical import LogicalNode, LogicalPlan

# Rounds over the rule list before giving up on convergence. Two rules
# that undo each other would otherwise loop forever.
MAX_PASSES = 4


@dataclass(frozen=True)
class LogicalPlanningContext:
    """Session values available to logical optimizer rules."""

    catalog: Any
    engine_config: Any


class LogicalOptimizerRule(Protocol):
    """Rewrite a whole logical plan without changing query meaning.

    A rule walks the plan itself, in the direction its rewrite needs:
    from the root toward the scans when it pushes information down,
    from the scans toward the root when it simplifies a node once its
    children are final (see `rewrite_bottom_up`). It returns the new
    root, or None when the plan is unchanged.
    """

    name: str

    def rewrite(
        self,
        root: LogicalNode,
        context: LogicalPlanningContext,
    ) -> LogicalNode | None: ...


def rewrite_bottom_up(
    root: LogicalNode,
    rewrite: Callable[[LogicalNode], LogicalNode],
) -> LogicalNode:
    """Rewrite children first, then offer each node to `rewrite`.

    `rewrite` returns the node itself to leave it alone. When nothing
    changes, the returned root is the same object that was passed in.
    """

    def visit(node):
        children = tuple(visit(child) for child in node.children())
        if children != node.children():
            node = node.with_children(children)
        return rewrite(node)

    return visit(root)


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
