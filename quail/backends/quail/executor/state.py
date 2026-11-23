"""Resources retained with a loaded model and with one query."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.chunk import InputStaging
    from quail.backends.quail.executor.extract import Vocabulary
    from quail.backends.quail.executor.models.base import ModelPipeline
    from quail.backends.quail.executor.readout import (
        AnswerRows,
        AsyncAnswers,
        AsyncLabelLogprobs,
        AsyncScores,
    )
    from quail.common.decode import DecodeCost
    from quail.common.tokens import Tokenizer
    from quail.specs.base import ModelSpec


@dataclass
class LoadedModelState:
    """Resources and reusable buffers retained until the model is released."""

    model: Any
    arena: KVArena
    pipeline: ModelPipeline
    model_spec: ModelSpec
    input_staging: InputStaging | None = None
    label_readout: AsyncLabelLogprobs | None = None
    decision_head: Any = None
    tokenizer: Tokenizer | None = None
    vocabulary: Vocabulary | None = None
    sae: Any = None


@dataclass
class QueryExecutionState:
    """Readouts and execution settings replaced at each query binding."""

    loaded_model: LoadedModelState
    torch: ModuleType
    async_answers: AsyncAnswers
    answer_rows: AnswerRows
    chunk_tokens: int
    async_scores: AsyncScores | None = None
    score_readout: Any = None
    gpu_timing: bool = False
    knee_rows: int = 0
    decode_cost: DecodeCost | None = None


class GpuRuntime(Protocol):
    """Loaded GPU resources used by graph execution and worker dispatch."""

    torch: ModuleType
    arena: KVArena
    pipeline: ModelPipeline
    spec: ModelSpec
    execution: Any
