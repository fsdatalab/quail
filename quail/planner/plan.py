"""Data structures produced by planning: PhysicalPlan and Refusal."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from quail.physical import PhysicalGraph, PhysicalNode, PortRef
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


@dataclass(frozen=True)
class PhysicalPlan:
    model: str
    device: str
    workers: int
    backend: str = "quail"
    estimated_seconds: float = 0.0
    nodes: tuple = ()          # typed nodes in topological order
    remarks: tuple = ()
    settings: Mapping[str, Any] = field(default_factory=dict, repr=False)
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

    def to_envelope(
        self,
        codecs,
        *,
        extension_modules=(),
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
            settings=self.settings,
        )

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
