"""A physical plan, a refusal to plan, and the edits that change a plan."""

import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import _quail_native

from quail.physical import PhysicalGraph, PortRef


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


def _no_tokens(_text: str) -> list:
    """Tokenize nothing: pricing an edited graph reads no new prompt."""
    return []


def _estimates(estimates: dict) -> dict:
    """The per-node estimates with their member lists as tuples."""
    return {node_id: {key: tuple(value) if isinstance(value, list) else value
                      for key, value in entry.items()}
            for node_id, entry in estimates.items()}


def graph_from_envelope(envelope: Mapping[str, Any]) -> PhysicalGraph:
    """Read and validate the graph of a plan envelope.

    Raises:
        ValueError: The envelope has a missing or unknown field, or its
            graph does not read or is invalid.
    """
    text = json.dumps(envelope)
    _quail_native.read_and_write_plan(text)
    return PhysicalGraph.from_json(json.dumps(envelope["graph"]))


def plan_from_json(found: dict):
    """Return the PhysicalPlan, or the Refusal, that Rust's planner wrote.

    Args:
        found: A planned region as ``NativePlanner.plan`` returns it, read
            from JSON: the envelope, the estimates, the operators, and,
            for Quail's backend, the pricer and the context fields.
    """
    if "refusal" in found:
        return refusal_from_fields(found["refusal"])
    envelope = found["envelope"]
    graph = PhysicalGraph.from_json(json.dumps(envelope["graph"]))
    settings = dict(envelope["settings"])
    if "candidates" in settings:
        settings["candidates"] = tuple(
            {**candidate, "rewrites": tuple(candidate["rewrites"])}
            for candidate in settings["candidates"])
    pricing = ({"fields": found["fields"], "pricer": found["pricer"]}
               if "pricer" in found else None)
    return PhysicalPlan(
        model=envelope["model"], device=envelope["device"], gpus=envelope["gpus"],
        backend=envelope["backend"],
        estimated_seconds=found.get("estimated_seconds", 0.0),
        nodes=graph.nodes, settings=settings, root=graph.root,
        estimates=_estimates(found.get("estimates", {})), pricing=pricing,
        operators=found.get("operators"))


@dataclass(frozen=True)
class PhysicalPlan:
    """A typed physical graph with its settings and estimates.

    estimates holds, per node id, the node's own estimated seconds
    and, for a filter sequence a later join anchors on, the recompute it
    would pay if its KV were released instead of pinned. pricing holds
    the planning context's fields and the pricer Rust wrote, which
    price an edited graph again; a plan without it is not priced again.
    operators records the selected operators for query execution and explain.
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
    pricing: Mapping[str, Any] | None = field(default=None, repr=False,
                                              compare=False)
    # the operators the graph lowers, as Rust's JSON
    operators: str | None = field(default=None, repr=False, compare=False)
    graph: PhysicalGraph = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("a physical plan needs at least one node")
        root = self.root or PortRef(
            self.nodes[-1].node_id, self.nodes[-1].outputs[0].name
        )
        graph = PhysicalGraph(tuple(self.nodes), root)
        graph.validate()
        object.__setattr__(self, "nodes", graph.nodes)
        object.__setattr__(self, "root", graph.root)
        object.__setattr__(self, "graph", graph)

    def _edit(self, edit: dict) -> "PhysicalPlan":
        """Apply an edit in Rust: rewire, set KV retention, validate, price.

        The estimate includes recomputation when an edit separates filters
        from a following join and their survivors were not already retained.

        Raises:
            PlanEditError: The edit names a missing node or edge, removes a
                node that changes what the query means, or leaves an
                invalid graph.
        """
        graph = json.loads(self.graph.to_json())
        if self.pricing is not None:
            retained = sorted(self.settings.get("retention", {}).get("initial", {}))
            found, message = _quail_native.plan_operator(
                "edit_plan", json.dumps(self.pricing["fields"]),
                json.dumps({"pricer": self.pricing["pricer"], "graph": graph,
                            "edit": edit, "retained": retained}),
                _no_tokens)
            if message is not None:
                raise PlanEditError(message)
            found = json.loads(found)
            edited = PhysicalGraph.from_json(json.dumps(found["graph"]))
            return replace(self, nodes=edited.nodes, root=edited.root,
                           estimated_seconds=found["seconds"],
                           estimates=_estimates(found["estimates"]))
        found, message = _quail_native.edit_graph(json.dumps(graph), json.dumps(edit))
        if message is not None:
            raise PlanEditError(message)
        edited = PhysicalGraph.from_json(found)
        return replace(self, nodes=edited.nodes, root=edited.root, estimates={})

    def insert(self, node, *, between: tuple) -> "PhysicalPlan":
        """Put node on the edge from producer to consumer.

        The consumer's port that read the producer reads the new node
        instead, and the new node reads the producer. No other node
        changes. The node must have no inputs yet and exactly one
        output of the value type the edge carries.
        """
        return self._edit({"op": "insert", "node": json.loads(node.to_json()),
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

    def to_envelope(self) -> dict:
        """Encode the typed graph for a process boundary."""
        self.graph.validate_backend(self.backend)
        return {
            "backend": self.backend,
            "model": self.model,
            "device": self.device,
            "gpus": self.gpus,
            "settings": dict(self.settings),
            "graph": json.loads(self.graph.to_json()),
        }
