"""Interfaces for query tree rules and executable graph rules."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quail.logical import LogicalNode
    from quail.physical import PhysicalGraph
    from quail.planner.optimizer.context import PlanningContext


class LogicalRule:
    """Simplify a query tree or propose equivalent trees for cost comparison."""

    name: str

    def rewrite(
        self, root: "LogicalNode", context: "PlanningContext",
    ) -> "LogicalNode | None":
        """Return a replacement root, or None to keep the input."""
        return None

    def alternatives(
        self, root: "LogicalNode", context: "PlanningContext",
    ) -> "tuple[LogicalNode, ...]":
        """Return other roots to compare after applying physical rules."""
        return ()


class PhysicalRule(ABC):
    """Update an executable graph and its candidate's settings."""

    name: str

    @abstractmethod
    def rewrite(
        self, graph: "PhysicalGraph", context: "PlanningContext",
    ) -> "PhysicalGraph | None":
        """Return a replacement graph, or None to keep the input."""
