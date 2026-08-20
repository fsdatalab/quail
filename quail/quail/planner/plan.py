"""PhysicalPlan and Refusal: what planning produces.

A PhysicalPlan is a JSON tree of settings plus per-operator token
counts - the counts are arithmetic the decisions already produced, so
they come free. There is no wall prediction anywhere: every decision
either needs no constants at all or reduces to a break-even
inequality, and nothing in the system consumes an estimated wall.
"""

import json
from dataclasses import dataclass
from typing import Optional


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
class StoreSpec:
    """A pinned host KV store: measured read bandwidth, bytes/s.
    `warm` means KV for this corpus is already saved from an earlier
    query. capacity_bytes caps what it may hold (None = unbounded)."""
    read_bw: float
    warm: bool = False
    capacity_bytes: Optional[float] = None


@dataclass(frozen=True)
class Refusal:
    """The answer when this configuration cannot execute the query:
    the violated constraint, what was needed, what was available -
    instead of running something degraded."""
    reasons: tuple
    constraint: str    # "weights_need_more_cards" | "suffix_over_chunk"
    #                    | "store_needed_but_disabled" | "unknown_model"
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
    calibration_source: str    # "calibrated" | "spec-scaled from ..."
    store_min_doc_tokens: int = 0    # documents at or above this
    #                                  length use the KV store; 0 =
    #                                  no store, 1 = everything stores
    operators: tuple = ()      # ordered operator dicts, JSON-able
    remarks: tuple = ()

    def to_json(self) -> str:
        d = dict(self.__dict__)
        # shard index lists are working data, not part of the report
        ops = []
        for op in self.operators:
            op = dict(op)
            if "shards" in op:
                op["shard_docs"] = [len(s) for s in op["shards"]]
                op["shard_tokens"] = op.pop("shard_token_loads", None)
                del op["shards"]
            ops.append(op)
        d["operators"] = ops
        return json.dumps(d, indent=2)


def resolve_model(name: str):
    """A registered ModelSpec, or the unknown-model Refusal."""
    from quail.specs import MODELS
    if name in MODELS:
        return MODELS[name]
    return Refusal(
        reasons=(f"{name!r} names no registered ModelSpec; known: "
                 f"{sorted(MODELS)}",),
        constraint="unknown_model", needed=1, available=0, unit="specs")


@dataclass(frozen=True)
class EngineConfig:
    """The three knobs, mapping directly onto Modal resources."""
    gpus: int = 1
    cpu_memory_gb: int = 64
    model: str = "qwen3-4b-fp8"
