"""Generic logical plan rewriting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from quail.logical import LogicalNode, LogicalPlan


@dataclass(frozen=True)
class LogicalPlanningContext:
    """Session values available to logical optimizer rules."""

    catalog: Any
    engine_config: Any


class LogicalOptimizerRule(Protocol):
    """Rewrite one logical node without changing query meaning."""

    name: str

    def rewrite(
        self,
        node: LogicalNode,
        context: LogicalPlanningContext,
    ) -> LogicalNode | None: ...


def apply_logical_rules(
    plan: LogicalPlan,
    rules: tuple[LogicalOptimizerRule, ...],
    context: LogicalPlanningContext,
) -> tuple[LogicalPlan, tuple[str, ...]]:
    """Apply registered rules from children to parent."""
    changed = []

    def rewrite(node):
        children = tuple(rewrite(child) for child in node.children())
        if children != node.children():
            node = node.with_children(children)
        for rule in rules:
            replacement = rule.rewrite(node, context)
            if replacement is not None and replacement != node:
                node = replacement
                changed.append(rule.name)
        return node

    optimized = LogicalPlan(rewrite(plan.root))
    optimized.validate()
    return optimized, tuple(changed)
