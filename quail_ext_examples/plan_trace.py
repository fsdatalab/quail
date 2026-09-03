"""Example execution observer for physical node metrics."""

from __future__ import annotations


class PlanTrace:
    """Record the rows processed by each physical node."""

    name = "example.plan_trace"

    def __init__(self):
        self._nodes = []

    def after_node(self, node, result) -> None:
        self._nodes.append({
            "node_id": node.node_id,
            "node_type": node.type_name,
            "input_rows": result.metrics.input_rows,
            "output_rows": result.metrics.output_rows,
        })

    def report(self) -> dict:
        return {"nodes": list(self._nodes)}


def register_quail_extension(registry) -> None:
    """Register the plan trace observer."""
    registry.register_observer(PlanTrace.name, PlanTrace)
