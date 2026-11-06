"""A physical plan, a refusal to plan, and the pass that sets document KV lifetimes."""

import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from quail.physical import (
    PhysicalGraph,
    PhysicalNode,
    PortRef,
)
from quail.physical.codec import plan_envelope


@dataclass(frozen=True)
class Refusal:
    """Returned when a configuration cannot execute the query."""
    reasons: tuple
    constraint: str    # "weights_need_more_cards" | "suffix_over_chunk"
    #                    | "unknown_model" | "sliding_kv_needs_more_memory"
    #                    | "flash_attention_4_unsupported"
    needed: float
    available: float
    unit: str          # "cards" | "tokens" | "bytes" | "version"


def refusal_from_fields(fields: dict) -> Refusal:
    """Return the Refusal whose fields Rust's planner wrote as JSON."""
    return Refusal(**{**fields, "reasons": tuple(fields["reasons"])})


class RefusalError(RuntimeError):
    """Raised when run() is called on a refused plan."""

    def __init__(self, refusal: Refusal):
        self.refusal = refusal
        super().__init__(
            f"{refusal.constraint}: needed {refusal.needed} "
            f"{refusal.unit}, available {refusal.available}. "
            + " ".join(refusal.reasons))


class PlanEditError(ValueError):
    """Raised by insert, remove, and move when an edit breaks a rule."""


@dataclass(frozen=True)
class PhysicalPlan:
    """A typed physical graph with its settings and estimates.

    estimates holds, per node id, the node's own estimated seconds
    and, for a filter sequence a later join anchors on, the recompute it
    would pay if its KV were released instead of pinned. estimator, a
    PlanPricer, recomputes them and the plan's seconds for an edited
    graph; it is not compared or shown.
    logical_plan records the selected tree for query execution and explain.
    """

    model: str
    device: str
    gpus: int
    backend: str = "quail"
    estimated_seconds: float = 0.0
    nodes: tuple = ()          # typed nodes in topological order
    settings: Mapping[str, Any] = field(default_factory=dict, repr=False)
    root: PortRef | None = None
    estimates: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict, repr=False)
    estimator: Any = field(default=None, repr=False, compare=False)
    # the logical plan the graph lowers, as Rust's JSON
    logical_plan: str | None = field(default=None, repr=False, compare=False)
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
        object.__setattr__(self, "nodes", graph.nodes)
        object.__setattr__(self, "root", graph.root)
        object.__setattr__(self, "graph", graph)
        if self.estimator is not None and not self.estimates:
            object.__setattr__(self, "estimates", self.estimator(graph)[1])


    def _edit(self, edit: dict) -> "PhysicalPlan":
        """Apply an edit in Rust: rewire, set KV retention, validate, price.

        The estimate includes recomputation when an edit separates filters
        from a following join and their survivors were not already retained.

        Raises:
            PlanEditError: The edit names a missing node or edge, removes a
                node that changes what the query means, or leaves an
                invalid graph.
        """
        import _quail_native

        from quail.physical.codec import built_in_codecs, decode_graph, encode_graph

        edit_with = getattr(self.estimator, "edit", None)
        if edit_with is not None:
            retained = set(self.settings.get("retention", {}).get("initial", {}))
            graph, seconds, estimates = edit_with(self.graph, edit, retained)
            return replace(self, nodes=graph.nodes, root=graph.root,
                           estimated_seconds=seconds, estimates=estimates)
        codecs = {codec.type_name: codec for codec in built_in_codecs()}
        found, message = _quail_native.edit_graph(
            json.dumps(encode_graph(self.graph, codecs)), json.dumps(edit))
        if message is not None:
            raise PlanEditError(message)
        graph = decode_graph(json.loads(found), codecs)
        return replace(self, nodes=graph.nodes, root=graph.root, estimates={})

    def insert(self, node: PhysicalNode, *, between: tuple) -> "PhysicalPlan":
        """Put node on the edge from producer to consumer.

        The consumer's port that read the producer reads the new node
        instead, and the new node reads the producer. No other node
        changes. The node must have no inputs yet and exactly one
        output of the value type the edge carries.
        """
        from quail.physical.codec import built_in_codecs

        codecs = {codec.type_name: codec for codec in built_in_codecs()}
        return self._edit({"op": "insert", "node": codecs[node.type_name].encode(node),
                           "between": list(between)})

    def remove(self, node_id: str) -> "PhysicalPlan":
        """Take a node out; its consumers read its producer instead.

        Allowed only for Barrier and Exchange nodes with one
        input and one output of the same value type. Removing a Scan,
        AiFilter, AiJoin, Recombine, Project, or Limit would change
        what the query means.
        """
        return self._edit({"op": "remove", "node_id": node_id})

    def move(self, node_id: str, *, between: tuple) -> "PhysicalPlan":
        """Remove a node and insert it on another edge."""
        return self._edit({"op": "move", "node_id": node_id,
                           "between": list(between)})

    def to_envelope(self, codecs) -> dict:
        """Encode the typed graph for a process boundary."""
        return plan_envelope(
            backend=self.backend,
            model=self.model,
            device=self.device,
            gpus=self.gpus,
            graph=self.graph,
            codecs=codecs,
            settings=self.settings,
        )
