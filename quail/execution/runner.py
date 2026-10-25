"""Generic execution for typed physical graphs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol

from quail.execution.ask_once import asked_once
from quail.execution.metrics import NodeMetrics, NodeResult, RunResult
from quail.physical import (
    ExecutionLocation,
    PhysicalGraph,
    PhysicalNode,
    PortRef,
    Project,
    Recombine,
)
from quail.physical.pipelines import build_pipelines


class NodeRuntime(Protocol):
    """Runtime implementation for one physical node type."""

    def execute(
        self,
        node: PhysicalNode,
        inputs: Mapping[str, Any],
        context: "ExecutionContext",
    ) -> NodeResult: ...


class ExecutionObserver(Protocol):
    """Observe node results without changing plan values."""

    name: str

    def after_node(self, node: PhysicalNode, result: NodeResult) -> None: ...

    def report(self) -> Mapping[str, Any]: ...


@dataclass
class ExecutionContext:
    """State shared while one physical graph runs."""

    runtimes: Mapping[str, NodeRuntime]
    model_execution: Any = None
    sources: Mapping[str, Any] = field(default_factory=dict)
    project: Callable[[Project, Any], Any] | None = None
    recombine: Callable[[Recombine, Mapping[str, Any]], Any] | None = None
    model_inputs: Callable[
        [PhysicalNode, Mapping[str, Any], "ExecutionContext"],
        Mapping[str, Any],
    ] | None = None
    model_result: Callable[
        [PhysicalNode, NodeResult, "ExecutionContext"], None,
    ] | None = None
    state: dict[str, Any] = field(default_factory=dict)
    observers: tuple[ExecutionObserver, ...] = ()
    functions: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    gpus: int = 1
    run_pipeline: Callable[..., Mapping[str, "NodeResult"]] | None = None
    # alias -> the value column holding each document's key document;
    # empty leaves every model operator asking per row
    ask_once: Mapping[str, str] = field(default_factory=dict)

def _check_ports(node, result) -> None:
    expected = {output.name for output in node.outputs}
    missing = expected - set(result.outputs)
    extra = set(result.outputs) - expected
    if missing or extra:
        raise ValueError(
            f"runtime {node.runtime_key!r} returned wrong ports "
            f"for {node.node_id!r}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}")


class GenericRunner:
    """Graph runner that executes operators after their inputs are ready.

    Compatible graph paths run together once their external inputs are ready.
    Each document advances through the path while its KV remains resident.
    """

    def run(
        self,
        graph: PhysicalGraph,
        context: ExecutionContext,
        initial_outputs: Mapping[PortRef, Any] | None = None,
        initial_metrics: Mapping[str, NodeMetrics] | None = None,
    ) -> RunResult:
        graph.validate(runtime_keys=set(context.runtimes))
        values: dict[tuple[str, str], Any] = {
            (ref.node_id, ref.port): value
            for ref, value in (initial_outputs or {}).items()
        }
        node_results: dict[str, NodeResult] = {}

        def record(node, result, started=None):
            if started is not None and result.metrics.wall_s == 0.0:
                # runtimes that do not time themselves are timed here
                result = replace(result, metrics=replace(
                    result.metrics, wall_s=time.perf_counter() - started))
            _check_ports(node, result)
            for port, value in result.outputs.items():
                values[(node.node_id, port)] = value
            node_results[node.node_id] = result
            for observer in context.observers:
                observer.after_node(node, result)

        completed = tuple(dict.fromkeys(ref.node_id for ref in (initial_outputs or ())))
        pipelines = (build_pipelines(graph, completed)
                     if context.run_pipeline else {})
        for node in graph.topological_nodes():
            pipeline = pipelines.get(node.node_id)
            if pipeline is not None:
                if node is not pipeline[-1]:
                    continue
                inside = {member.node_id for member in pipeline}
                inputs = {
                    member.node_id: {
                        port.name: values[(port.source.node_id,
                                           port.source.port)]
                        for port in member.inputs
                        if port.source.node_id not in inside}
                    for member in pipeline}
                asked = asked_once(graph, pipeline, inputs, context)
                started = time.perf_counter()
                results = context.run_pipeline(
                    pipeline, inputs if asked is None else asked.inputs, context)
                if asked is not None:
                    results = asked.expanded(results)
                # a pipeline's wall is charged once, to its first member
                for index, member in enumerate(pipeline):
                    record(member, results[member.node_id],
                           started if index == 0 else None)
                continue
            supplied = {
                output.name: values[(node.node_id, output.name)]
                for output in node.outputs
                if (node.node_id, output.name) in values
            }
            if supplied:
                expected = {output.name for output in node.outputs}
                if set(supplied) != expected:
                    raise ValueError(
                        f"initial outputs for {node.node_id!r} are partial"
                    )
                result = NodeResult(
                    supplied,
                    (initial_metrics or {}).get(node.node_id, NodeMetrics()),
                )
            else:
                inputs = {
                    input_port.name: values[
                        (input_port.source.node_id, input_port.source.port)
                    ]
                    for input_port in node.inputs
                }
                asked = asked_once(graph, (node,), {node.node_id: inputs}, context)
                started = time.perf_counter()
                result = context.runtimes[node.runtime_key].execute(
                    node, inputs if asked is None else asked.inputs[node.node_id],
                    context)
                if asked is not None:
                    result = asked.expanded({node.node_id: result})[node.node_id]
                record(node, result, started)
                continue
            record(node, result)

        root = (graph.root.node_id, graph.root.port)
        metrics = NodeMetrics()
        for node in graph.topological_nodes():
            metrics = metrics + node_results[node.node_id].metrics
        return RunResult(values[root], node_results, metrics)


def compute_subgraph(graph: PhysicalGraph) -> PhysicalGraph:
    """Return model nodes and the input nodes they read."""
    by_id = {node.node_id: node for node in graph.nodes}
    selected = {
        node.node_id for node in graph.nodes
        if node.location is ExecutionLocation.GPU_EXECUTOR
        or node.backend is not None
    }

    def include_inputs(node_id: str) -> None:
        for input_port in by_id[node_id].inputs:
            source_id = input_port.source.node_id
            if source_id not in selected:
                selected.add(source_id)
                include_inputs(source_id)

    for node_id in tuple(selected):
        include_inputs(node_id)
    nodes = tuple(
        node for node in graph.topological_nodes()
        if node.node_id in selected
    )
    if not nodes:
        raise ValueError("physical graph has no compute operation")
    root_node = nodes[-1]
    return PhysicalGraph(
        nodes,
        PortRef(root_node.node_id, root_node.outputs[0].name),
    )
