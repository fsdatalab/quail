"""Metrics and results reported by physical nodes and graph runs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from numbers import Real
from typing import Any, Mapping


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
    peak_gpu_bytes: int = 0
    gpu_s: float = 0.0
    chunks: int = 0
    extension: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_scalars(cls, metrics: Mapping[str, Any]) -> "NodeMetrics":
        """Rebuild metrics saved by scalar_node_metrics.

        A saved number that is not a standard field, such as a filter's
        borrowed_prefix_tokens, returns to the extension mapping.
        """
        names = {field.name for field in fields(cls)} - {"extension"}
        return cls(
            **{name: value for name, value in metrics.items() if name in names},
            extension={name: value for name, value in metrics.items()
                       if name not in names})

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
            peak_gpu_bytes=max(self.peak_gpu_bytes, other.peak_gpu_bytes),
            gpu_s=self.gpu_s + other.gpu_s,
            chunks=self.chunks + other.chunks,
            extension=merge_metric_extensions(self.extension, other.extension),
        )


def merge_metric_extensions(left, right) -> dict:
    """Sum extension counters and retain the latest descriptive values."""
    merged = dict(left)
    for name, value in right.items():
        if isinstance(value, Real) and not isinstance(value, bool):
            value += left.get(name, 0)
        merged[name] = value
    return merged


def scalar_node_metrics(nodes: Mapping[str, "NodeResult"]) -> dict:
    """Return the scalar metrics of each node, keyed by node id.

    Extension metrics that are plain numbers, such as a filter's
    borrowed_prefix_tokens, are included beside the standard fields.
    """
    scalar_fields = tuple(
        field.name for field in fields(NodeMetrics)
        if field.name != "extension"
    )
    return {
        node_id: {
            **{name: getattr(result.metrics, name) for name in scalar_fields},
            **{name: value
               for name, value in result.metrics.extension.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)},
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
