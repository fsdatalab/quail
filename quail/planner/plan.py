"""Data structures produced by planning: PhysicalPlan and Refusal."""

import json
from dataclasses import dataclass, field

from quail.physical import (
    AdaptiveJoinPlan,
    AnchoredJoin,
    PhysicalGraph,
    PhysicalNode,
    PortRef,
)
from quail.physical.codec import NodeCodec, built_in_codecs, plan_envelope


@dataclass(frozen=True)
class CorpusStats:
    n_docs: int
    total_tokens: int
    max_doc_tokens: int

    @property
    def mean_doc_tokens(self) -> float:
        return self.total_tokens / max(1, self.n_docs)

    @classmethod
    def from_doc_tokens(cls, doc_tokens) -> "CorpusStats":
        toks = [int(t) for t in doc_tokens]
        return cls(n_docs=len(toks), total_tokens=sum(toks),
                   max_doc_tokens=max(toks) if toks else 0)


@dataclass(frozen=True)
class Refusal:
    """Returned when a configuration cannot execute the query."""
    reasons: tuple
    constraint: str    # "weights_need_more_cards" | "suffix_over_chunk"
    #                    | "unknown_model"
    needed: float
    available: float
    unit: str          # "cards" | "tokens" | "bytes"


@dataclass(frozen=True)
class PhysicalPlan:
    model: str
    device: str
    workers: int
    tensor_parallel: int
    kv_dtype: str              # always "bf16"
    chunk_tokens: int          # the batch size (activation/index bound)
    admission_tokens: int      # KV residency (the arena)
    order_rule: str            # "as_written" | "by_cost"
    order_source: str          # which rule chose it, for explain()
    backend: str = "quail"
    estimated_seconds: float = 0.0
    limit: int | None = None   # output row cap; None = no limit
    nodes: tuple = ()          # typed nodes in topological order
    remarks: tuple = ()
    root: PortRef | None = None
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

    def node(self, node_id: str) -> PhysicalNode:
        return self.graph.node(node_id)

    def nodes_by_type(self, type_name: str) -> list:
        """Return outer and expected child nodes with one type name."""
        nodes = list(self.nodes)
        for node in self.nodes:
            if isinstance(node, AdaptiveJoinPlan):
                nodes.extend(node.expected_nodes)
        return [
            node for node in nodes
            if node.type_name == type_name
        ]

    def expected_join_nodes(self) -> tuple:
        """Return the join nodes selected from planning estimates."""
        adaptive = [
            node for node in self.nodes
            if isinstance(node, AdaptiveJoinPlan)
        ]
        if adaptive:
            return adaptive[0].expected_nodes
        return tuple(
            node for node in self.nodes if isinstance(node, AnchoredJoin)
        )

    def expected_join_stages(self) -> tuple:
        """Return expected join stages in execution order."""
        return tuple(
            stage
            for node in self.expected_join_nodes()
            if isinstance(node, AnchoredJoin)
            for stage in node.stages
        )

    def to_envelope(
        self,
        codecs,
        *,
        extension_modules=(),
        include_runtime_data=True,
    ) -> dict:
        """Encode the typed graph for a process boundary."""
        return plan_envelope(
            backend=self.backend,
            model=self.model,
            device=self.device,
            workers=self.workers,
            graph=self.graph,
            codecs=codecs,
            extension_modules=tuple(extension_modules),
            include_runtime_data=include_runtime_data,
        )

    def to_json(self) -> str:
        d = dict(self.__dict__)
        d.pop("graph")
        d.pop("nodes")
        d["root"] = {
            "node_id": self.graph.root.node_id,
            "port": self.graph.root.port,
        }
        codecs = {codec.type_name: codec for codec in built_in_codecs()}
        for node in self.graph.nodes:
            codecs.setdefault(node.type_name, NodeCodec(type(node)))
        d["physical_plan"] = self.to_envelope(
            codecs, include_runtime_data=False
        )
        return json.dumps(d, indent=2)


def resolve_model(name: str):
    """Return a ModelSpec by name, or a Refusal if unknown."""
    from quail.specs import MODELS
    if name in MODELS:
        return MODELS[name]
    return Refusal(
        reasons=(f"{name!r} names no registered ModelSpec; known: "
                 f"{sorted(MODELS)}",),
        constraint="unknown_model", needed=1, available=0, unit="specs")


@dataclass(frozen=True)
class EngineConfig:
    """Top-level engine configuration."""
    gpus: int = 1
    model: str = "qwen3-4b-fp8"
    backend: str = "quail"
