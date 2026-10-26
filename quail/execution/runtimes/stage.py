"""Run a query stage once its materialized inputs are available."""

from quail.execution.metrics import NodeMetrics, NodeResult


class QueryStageRuntime:
    """Execute a stage through the coordinator's request dispatcher."""

    def execute(self, node, inputs, context) -> NodeResult:
        result = context.functions[node.runtime_key](node, inputs)
        return NodeResult({"rows": result}, NodeMetrics(
            output_rows=result.known_num_rows(),
            fresh_tokens=result.report.get("fresh_tokens", 0),
            cached_tokens=result.report.get("cached_tokens", 0) or 0))
