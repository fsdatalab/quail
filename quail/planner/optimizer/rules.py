"""Shared interface for universal and cost-based optimizer rules."""

from typing import Any, Literal, Protocol


class OptimizerRule(Protocol):
    """Transform a query tree or executable graph without changing its meaning.

    Universal rules apply whenever their conditions hold. Cost-based rules
    propose query tree alternatives for the planner to price, or choose graph
    implementations by cost. Return None when the input is unchanged.
    """

    name: str
    target: Literal["logical", "physical"]
    cost_based: bool

    def rewrite(self, plan: Any, context: Any) -> Any: ...
