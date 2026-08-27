"""Data structures produced by planning: PhysicalPlan and Refusal."""

import json
from dataclasses import dataclass


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
    limit: int | None = None   # output row cap; None = no limit
    nodes: tuple = ()          # dataflow graph in topological order;
    #                            each node dict has "id", "op", "inputs"
    remarks: tuple = ()

    def node(self, node_id: str) -> dict:
        for n in self.nodes:
            if n["id"] == node_id:
                return n
        raise KeyError(node_id)

    def nodes_by_op(self, op: str) -> list:
        return [n for n in self.nodes if n["op"] == op]

    def to_json(self) -> str:
        d = dict(self.__dict__)
        # shard index lists are working data, not part of the report
        nodes = []
        for n in self.nodes:
            n = dict(n)
            if "shards" in n:
                n["shard_docs"] = [len(s) for s in n["shards"]]
                n["shard_tokens"] = n.pop("shard_token_loads", None)
                del n["shards"]
            nodes.append(n)
        d["nodes"] = nodes
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
    """Top-level engine configuration: GPU count and model."""
    gpus: int = 1
    model: str = "qwen3-4b-fp8"
