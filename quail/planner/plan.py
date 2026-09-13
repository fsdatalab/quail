"""Data structures produced by planning: PhysicalPlan and Refusal."""

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
    Exchange,
    Foreign,
    GraphValidationError,
    InputPort,
    PhysicalGraph,
    PhysicalNode,
    PortRef,
    validate_streams,
)
from quail.physical.codec import plan_envelope
from quail.specs import MODELS


@dataclass(frozen=True)
class CorpusStats:
    n_docs: int
    total_tokens: int
    max_doc_tokens: int

    @property
    def mean_doc_tokens(self) -> float:
        return self.total_tokens / max(1, self.n_docs)

@dataclass(frozen=True)
class Refusal:
    """Returned when a configuration cannot execute the query."""
    reasons: tuple
    constraint: str    # "weights_need_more_cards" | "suffix_over_chunk"
    #                    | "unknown_model"
    needed: float
    available: float
    unit: str          # "cards" | "tokens" | "bytes"


class PlanEditError(ValueError):
    """Raised by insert, remove, and move when an edit breaks a rule."""


@dataclass(frozen=True)
class PhysicalPlan:
    """A typed physical graph with its settings and estimates.

    estimates holds, per node id, the node's own estimated seconds
    and, for a filter chain a later join anchors on, the recompute it
    would pay if its KV were released instead of pinned. estimator
    recomputes them for an edited graph; it is not compared or shown.
    """

    model: str
    device: str
    workers: int
    backend: str = "quail"
    estimated_seconds: float = 0.0
    nodes: tuple = ()          # typed nodes in topological order
    remarks: tuple = ()
    settings: Mapping[str, Any] = field(default_factory=dict, repr=False)
    root: PortRef | None = None
    estimates: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict, repr=False)
    estimator: Any = field(default=None, repr=False, compare=False)
    graph: PhysicalGraph = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("a physical plan needs at least one node")
        if not all(isinstance(node, PhysicalNode) for node in self.nodes):
            raise TypeError("PhysicalPlan nodes must implement PhysicalNode")
        root = self.root or PortRef(
            self.nodes[-1].node_id, self.nodes[-1].outputs[0].name
        )
        graph = PhysicalGraph(tuple(self.nodes), root)
        graph.validate()
        validate_streams(graph)
        object.__setattr__(self, "nodes", graph.nodes)
        object.__setattr__(self, "root", graph.root)
        object.__setattr__(self, "graph", graph)
        if self.estimator is not None and not self.estimates:
            object.__setattr__(self, "estimates", self.estimator(graph))


    def _edge(self, producer: str, consumer: str) -> tuple:
        """The one input port of consumer that reads producer."""
        try:
            source = self.graph.node(producer)
            target = self.graph.node(consumer)
        except KeyError as error:
            raise PlanEditError(
                f"no node {error.args[0]!r} in the plan") from error
        ports = [port for port in target.inputs
                 if port.source.node_id == source.node_id]
        if len(ports) != 1:
            raise PlanEditError(
                f"{consumer!r} reads {len(ports)} outputs of {producer!r}; "
                f"an edit needs exactly one edge between them")
        return source, target, ports[0]

    def _rebuild(self, nodes) -> "PhysicalPlan":
        """A new plan over nodes: pins re-derived, validated, re-estimated.

        The plan's seconds are the planner's search estimate plus the
        recompute expected at every chain an edit unpinned (a chain
        the planner itself left unpinned already retains its survivors
        by schedule and is not counted twice).
        """
        nodes = _rederive_pins(tuple(nodes))
        try:
            edited = replace(self, nodes=nodes, root=None, estimates={})
        except GraphValidationError as error:
            raise PlanEditError(str(error)) from error
        base = self.settings.get("search_seconds")
        if base is None:
            return edited
        scheduled = set(self.settings.get("retention", {}).get("initial", {}))
        recompute = sum(
            edited.estimates.get(node.node_id, {}).get(
                "release_recompute_seconds", 0.0)
            for node in edited.nodes
            if isinstance(node, AiFilter) and not node.pin_survivors
            and node.alias not in scheduled)
        return replace(edited, estimated_seconds=base + recompute,
                       estimates=edited.estimates)

    def insert(self, node: PhysicalNode, *, between: tuple) -> "PhysicalPlan":
        """Put node on the edge from producer to consumer.

        The consumer's port that read the producer reads the new node
        instead, and the new node reads the producer. No other node
        changes. The node must have no inputs yet and exactly one
        output of the value type the edge carries.
        """
        producer, consumer, port = self._edge(*between)
        if any(existing.node_id == node.node_id for existing in self.nodes):
            raise PlanEditError(
                f"the plan already has a node {node.node_id!r}")
        if node.inputs:
            raise PlanEditError(
                f"{node.node_id!r} already has inputs; insert wires its "
                f"one input itself")
        outputs = node.outputs
        if len(outputs) != 1 or outputs[0].value_type is not port.value_type:
            raise PlanEditError(
                f"{node.node_id!r} needs exactly one output of type "
                f"{port.value_type.value} to sit on the edge from "
                f"{producer.node_id!r} to {consumer.node_id!r}")
        wired = node.with_inputs((InputPort(
            "input:0", port.value_type, port.source, port.schema),))
        rewired = consumer.with_inputs(tuple(
            replace(existing, source=PortRef(node.node_id, outputs[0].name))
            if existing is port else existing
            for existing in consumer.inputs))
        nodes = []
        for existing in self.nodes:
            if existing.node_id == consumer.node_id:
                nodes.append(wired)
                nodes.append(rewired)
            else:
                nodes.append(existing)
        return self._rebuild(nodes)

    def remove(self, node_id: str) -> "PhysicalPlan":
        """Take a node out; its consumers read its producer instead.

        Allowed only for Foreign, Barrier, and Exchange nodes with one
        input and one output of the same value type. Removing a Scan,
        AiFilter, AiJoin, Recombine, Project, or Limit would change
        what the query means.
        """
        try:
            node = self.graph.node(node_id)
        except KeyError as error:
            raise PlanEditError(f"no node {node_id!r} in the plan") from error
        if not isinstance(node, (Foreign, Barrier, Exchange)):
            raise PlanEditError(
                f"{node_id!r} is a {type(node).__name__}; removing it would "
                f"change what the query means (only Foreign, Barrier, and "
                f"Exchange nodes can be removed)")
        if len(node.inputs) != 1 or len(node.outputs) != 1 \
                or node.inputs[0].value_type is not node.outputs[0].value_type:
            raise PlanEditError(
                f"{node_id!r} needs one input and one output of the same "
                f"type to be removed; it has {len(node.inputs)} and "
                f"{len(node.outputs)}")
        upstream = node.inputs[0].source
        gone = PortRef(node_id, node.outputs[0].name)
        nodes = []
        for existing in self.nodes:
            if existing.node_id == node_id:
                continue
            if any(port.source == gone for port in existing.inputs):
                existing = existing.with_inputs(tuple(
                    replace(port, source=upstream) if port.source == gone
                    else port for port in existing.inputs))
            nodes.append(existing)
        return self._rebuild(nodes)

    def move(self, node_id: str, *, between: tuple) -> "PhysicalPlan":
        """Remove a node and insert it on another edge."""
        node = self.graph.node(node_id)
        return self.remove(node_id).insert(
            node.with_inputs(()), between=between)

    def to_envelope(self, codecs) -> dict:
        """Encode the typed graph for a process boundary."""
        return plan_envelope(
            backend=self.backend,
            model=self.model,
            device=self.device,
            workers=self.workers,
            graph=self.graph,
            codecs=codecs,
            settings=self.settings,
        )

def _rederive_pins(nodes: tuple) -> tuple:
    """Re-derive pin_survivors, keep_kv, and hold_tokens from the shape.

    A chain pins its survivors only when its stream reaches the join
    anchored on its alias through per-batch nodes alone; otherwise
    its survivors go through the retention pool.
    """
    graph = PhysicalGraph(nodes, PortRef(nodes[-1].node_id,
                                         nodes[-1].outputs[0].name))
    joins = {node.anchor: node for node in nodes if isinstance(node, AiJoin)}
    out = []
    for node in nodes:
        if not isinstance(node, AiFilter) or node.alias not in joins:
            out.append(node)
            continue
        pinnable = _stream_reaches_join(graph, node)
        hold = max((stage.anchor_frame_tokens
                    for stage in joins[node.alias].stages), default=0)
        out.append(replace(
            node,
            pin_survivors=pinnable,
            keep_kv=not pinnable,
            hold_tokens=hold if pinnable else 0,
            arena_writes=True))
    return tuple(out)


def _stream_reaches_join(graph, chain) -> bool:
    """Whether a chain's survivors reach its join through per-batch nodes."""
    # only this chain is pinned in the trial
    trial = PhysicalGraph(tuple(
        replace(node, pin_survivors=node.node_id == chain.node_id)
        if isinstance(node, AiFilter) else node
        for node in graph.nodes), graph.root)
    try:
        validate_streams(trial)
    except GraphValidationError:
        return False
    return True


def resolve_model(name: str, models=None):
    """Return a ModelSpec by name, or a Refusal if unknown."""
    if models is None:
        models = MODELS
    if name in models:
        return models[name]
    return Refusal(
        reasons=(f"{name!r} names no registered ModelSpec; known: "
                 f"{sorted(models)}",),
        constraint="unknown_model", needed=1, available=0, unit="specs")


@dataclass(frozen=True)
class EngineConfig:
    """Top-level engine configuration."""
    gpus: int = 1
    model: str = "qwen3-4b-fp8"
    backend: str = "quail"
    device: str = "h100-sxm"
    # sum the CUDA event pair each forward chunk records into gpu_s;
    # off by default so a run never pays for a measurement it does not read
    gpu_timing: bool = False
