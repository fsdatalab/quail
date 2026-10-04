"""Resources retained with a loaded model and with one query."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.device import DeviceImplementation
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
    # the device implementation's reusable transfer buffers, when it has any
    input_staging: Any = None
    label_readout: AsyncLabelLogprobs | None = None
    # a decision model's option-scoring head; None for other models
    decision_head: Any = None


@dataclass
class QueryExecutionState:
    """Readouts and execution settings replaced at each query binding."""

    loaded_model: LoadedModelState
    implementation: DeviceImplementation
    async_answers: AsyncAnswers
    answer_rows: AnswerRows
    chunk_tokens: int
    async_scores: AsyncScores | None = None
    # the AI.SCORE readout of a model that does not score with answer
    # rows; None scores yes against no over answer_rows
    score_readout: Any = None
    gpu_timing: bool = False
