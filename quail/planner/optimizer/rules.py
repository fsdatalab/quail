"""Interfaces for query tree rules and executable graph rules."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quail.logical import LogicalNode
    from quail.physical import PhysicalGraph
    from quail.planner.optimizer.context import PlanningContext


class LogicalRule(ABC):
    """Return equivalent query trees to compare after physical planning."""

    name: str

    @abstractmethod
    def rewrites(
        self, root: "LogicalNode", context: "PlanningContext",
    ) -> "list[LogicalNode]":
        """Return candidate roots, including root if it should remain a candidate.

        Return [root] for no change, [replacement] to replace it, or multiple
        roots to compare. The list must not be empty.
        """


class PhysicalRule(ABC):
    """Update an executable graph and its candidate's settings."""

    name: str

    @abstractmethod
    def rewrite(
        self, graph: "PhysicalGraph", context: "PlanningContext",
    ) -> "PhysicalGraph | None":
        """Return a replacement graph, or None to keep the input."""
