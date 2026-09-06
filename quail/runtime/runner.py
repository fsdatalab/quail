"""Generic execution for typed physical graphs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Mapping, Protocol

from quail.physical import (
    AdaptiveJoinPlan,
    DocumentInput,
    Exchange,
    ExecutionLocation,
    HashJoin,
    Limit,
    PhysicalGraph,
    PhysicalNode,
    PortRef,
    Project,
    ValueType,
)
from quail.runtime.result import (
    build_result_declaration,
    IndexRelation,
    QueryResult,
    true_answer_rows,
)


@dataclass(frozen=True)
class NodeMetrics:
    """Metrics reported by one physical node."""

    wall_s: float = 0.0
    input_rows: int = 0
    output_rows: int = 0
    evaluated_documents: int = 0
    evaluated_document_pairs: int = 0
    fresh_tokens: int = 0
    cached_tokens: int = 0
    kv_hits: int = 0
    kv_misses: int = 0
    kv_removals: int = 0
    kv_recomputations: int = 0
    regret_tokens: int = 0
    peak_gpu_bytes: int = 0
    extension: Mapping[str, Any] = field(default_factory=dict)

    def __add__(self, other: "NodeMetrics") -> "NodeMetrics":
        return NodeMetrics(
            wall_s=self.wall_s + other.wall_s,
            input_rows=self.input_rows + other.input_rows,
            output_rows=self.output_rows + other.output_rows,
            evaluated_documents=(
                self.evaluated_documents + other.evaluated_documents
            ),
            evaluated_document_pairs=(
                self.evaluated_document_pairs
                + other.evaluated_document_pairs
            ),
            fresh_tokens=self.fresh_tokens + other.fresh_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            kv_hits=self.kv_hits + other.kv_hits,
            kv_misses=self.kv_misses + other.kv_misses,
            kv_removals=self.kv_removals + other.kv_removals,
            kv_recomputations=(
                self.kv_recomputations + other.kv_recomputations
            ),
            regret_tokens=self.regret_tokens + other.regret_tokens,
            peak_gpu_bytes=max(self.peak_gpu_bytes, other.peak_gpu_bytes),
            extension={**self.extension, **other.extension},
        )


def scalar_node_metrics(nodes: Mapping[str, "NodeResult"]) -> dict:
    """Return the scalar metrics of each node, keyed by node id."""
    scalar_fields = tuple(
        field.name for field in fields(NodeMetrics)
        if field.name != "extension"
    )
    return {
        node_id: {
            name: getattr(result.metrics, name)
            for name in scalar_fields
        }
        for node_id, result in nodes.items()
    }


@dataclass(frozen=True)
class NodeResult:
    """Outputs and metrics produced by one physical node."""

    outputs: Mapping[str, Any]
    metrics: NodeMetrics = NodeMetrics()


@dataclass(frozen=True)
class RunResult:
    """Root value and per node results from one graph run."""

    value: Any
    nodes: Mapping[str, NodeResult]
    metrics: NodeMetrics


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
    hash_join: Callable[[HashJoin, Mapping[str, Any]], Any] | None = None
    adaptive_join: Callable[
        [AdaptiveJoinPlan, Mapping[str, Any], "ExecutionContext"],
        NodeResult,
    ] | None = None
    model_inputs: Callable[
        [PhysicalNode, Mapping[str, Any], "ExecutionContext"],
        Mapping[str, Any],
    ] | None = None
    model_result: Callable[
        [PhysicalNode, NodeResult, "ExecutionContext"], None,
    ] | None = None
    state: dict[str, Any] = field(default_factory=dict)
    observers: tuple[ExecutionObserver, ...] = ()

    def execute_graph(self, graph: PhysicalGraph) -> RunResult:
        """Execute a child graph with the same model state."""
        return GenericRunner().run(graph, self)


class GenericRunner:
    """Execute nodes after all their inputs are available."""

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
        metrics = NodeMetrics()

        for node in graph.topological_nodes():
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
                node_results[node.node_id] = result
                metrics = metrics + result.metrics
                for observer in context.observers:
                    observer.after_node(node, result)
                continue
            inputs = {
                input_port.name: values[
                    (input_port.source.node_id, input_port.source.port)
                ]
                for input_port in node.inputs
            }
            started = time.perf_counter()
            result = context.runtimes[node.runtime_key].execute(
                node, inputs, context
            )
            if result.metrics.wall_s == 0.0:
                # a runtime that does not time itself is timed here, so
                # every executed node carries its wall seconds
                result = replace(result, metrics=replace(
                    result.metrics,
                    wall_s=time.perf_counter() - started,
                ))
            expected = {output.name for output in node.outputs}
            missing = expected - set(result.outputs)
            extra = set(result.outputs) - expected
            if missing or extra:
                raise ValueError(
                    f"runtime {node.runtime_key!r} returned wrong ports; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}")
            for port, value in result.outputs.items():
                values[(node.node_id, port)] = value
            node_results[node.node_id] = result
            metrics = metrics + result.metrics
            for observer in context.observers:
                observer.after_node(node, result)

        root = (graph.root.node_id, graph.root.port)
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


class DocumentInputRuntime:
    """Read a prepared source registered by document alias."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, DocumentInput):
            raise TypeError(type(node).__name__)
        if node.input_id not in context.sources:
            raise KeyError(
                f"no prepared source for input {node.input_id!r}"
            )
        value = context.sources[node.input_id]
        return NodeResult({f"ids:{node.alias}": value})


class ExchangeRuntime:
    """Return the survivor ids named by the exchange outputs."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Exchange):
            raise TypeError(type(node).__name__)
        by_source_port = {
            input_port.source.port: inputs[input_port.name]
            for input_port in node.inputs
        }
        return NodeResult({
            output.name: by_source_port[output.name]
            for output in node.outputs
        })


class HashJoinRuntime:
    """Run the configured exact answer relation join."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, HashJoin):
            raise TypeError(type(node).__name__)
        if context.hash_join is None:
            import pyarrow as pa


            answer_tables = []
            survivors = {}
            for input_port in node.inputs:
                value = inputs[input_port.name]
                if not isinstance(value, pa.Table):
                    raise TypeError(
                        "HashJoin inputs must be Arrow tables"
                    )
                if input_port.value_type is ValueType.JOIN_ANSWERS:
                    answer_tables.append(true_answer_rows(value))
                elif input_port.value_type is ValueType.DOCUMENT_IDS:
                    if len(value.column_names) != 1:
                        raise ValueError(
                            "a survivor relation needs one alias column"
                        )
                    alias = value.column_names[0]
                    survivors[alias] = value.column(alias).combine_chunks()
                else:
                    raise TypeError(
                        f"HashJoin cannot read {input_port.value_type.value}"
                    )
            declaration, schema = build_result_declaration(
                answer_tables,
                survivors,
                node.alias_order[0],
            )
            value = IndexRelation(declaration, schema)
        else:
            value = context.hash_join(node, inputs)
        return NodeResult({"tuples": value})


class ProjectRuntime:
    """Run the configured result projection."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Project):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Project needs one input")
        value = next(iter(inputs.values()))
        if context.project is not None:
            value = context.project(node, value)
        return NodeResult({"rows": value})


class LimitRuntime:
    """Limit a table, batch, or Python sequence."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Limit):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Limit needs one input")
        value = next(iter(inputs.values()))

        if isinstance(value, QueryResult):
            value = value.with_limit(node.count)
        elif hasattr(value, "slice"):
            value = value.slice(0, node.count)
        else:
            value = value[:node.count]
        return NodeResult({"rows": value})


class ModelNodeRuntime:
    """Delegate one model node to the shared model execution object."""

    def execute(self, node, inputs, context) -> NodeResult:
        if context.model_execution is None:
            raise RuntimeError("model node has no model execution object")
        if context.model_inputs is not None:
            inputs = context.model_inputs(node, inputs, context)
        result = context.model_execution.execute(node, inputs)
        if not isinstance(result, NodeResult):
            raise TypeError("model execution must return NodeResult")
        if context.model_result is not None:
            context.model_result(node, result, context)
        return result


def built_in_runtimes() -> dict[str, NodeRuntime]:
    """Return runtimes for the backend independent physical nodes."""
    return {
        DocumentInput.runtime_key: DocumentInputRuntime(),
        Exchange.runtime_key: ExchangeRuntime(),
        HashJoin.runtime_key: HashJoinRuntime(),
        Project.runtime_key: ProjectRuntime(),
        Limit.runtime_key: LimitRuntime(),
    }
