"""The AiTopK node: pairwise ranking inside one physical operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..base import OutputPort, ValueType
from .inference import AiInference, InferenceSpec


@dataclass(frozen=True)
class RankSpec(InferenceSpec):
    """Settings for keeping the k best rows under pairwise comparisons.

    ``aliases`` names the same table twice, so the scorer treats each
    comparison as a pair and shares the pivot document's prefix.
    ``k`` is the number of rows kept. ``bound`` is the integer in the
    rank filter, and ``comparison`` is ``<=`` or ``<``.
    """

    k: int = 0
    comparison: str = "<="
    bound: int = 0
    partition: tuple[str, ...] = ()
    num_pivots: int = 1
    margin: float = 0.15
    seed: int = 0
    id_column: str = "id"
    expected_comparisons: float = 0.0


@dataclass(frozen=True)
class AiTopK(AiInference):
    """Keep the k best rows by pairwise model comparisons.

    The comparisons run inside this node, so the graph does not grow
    with the number of rounds. The output rows are in rank order, and
    the rank column numbers them from 1 inside each partition.
    """

    spec: RankSpec | None = None

    spec_type: ClassVar[type[InferenceSpec]] = RankSpec
    type_name: ClassVar[str] = "quail.ai_top_k"
    runtime_key: ClassVar[str] = type_name

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        spec = self.spec
        if spec is None:
            return ()
        return (OutputPort(
            "rows", ValueType.ROWS, schema=(spec.aliases[0], spec.name)),)

    @property
    def rows_port(self) -> str:
        return "rows"

    def explain_fields(self) -> Mapping[str, Any]:
        spec = self.spec
        return {
            "model": self.model,
            "output": None if spec is None else spec.name,
            "k": 0 if spec is None else spec.k,
            "comparison": "" if spec is None else spec.comparison,
            "partition": [] if spec is None else list(spec.partition),
            "num_pivots": 1 if spec is None else spec.num_pivots,
            "margin": 0.0 if spec is None else spec.margin,
            "expected_comparisons": (
                0 if spec is None else spec.expected_comparisons),
            "estimated_seconds": (
                0 if spec is None else spec.estimated_seconds),
            "expected_rows": 0 if spec is None else spec.expected_rows,
        }
