"""Resources retained with a loaded model and with one query."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.loop import InputStaging
    from quail.backends.quail.executor.models.base import ModelPipeline
    from quail.backends.quail.executor.readout import (
        AnswerRows,
        AsyncAnswers,
        AsyncLabelLogprobs,
        AsyncScores,
    )
    from quail.specs.base import ModelSpec


@dataclass
class LoadedModelState:
    """Resources and reusable buffers retained until the model is released."""

    model: Any
    arena: KVArena
    pipeline: ModelPipeline
    model_spec: ModelSpec | None = None
    input_staging: InputStaging | None = None
    label_readout: AsyncLabelLogprobs | None = None


@dataclass
class QueryExecutionState:
    """Readouts and execution settings replaced at each query binding."""

    loaded_model: LoadedModelState
    torch: ModuleType
    async_answers: AsyncAnswers
    answer_rows: AnswerRows
    chunk_tokens: int
    async_scores: AsyncScores | None = None
    gpu_timing: bool = False
