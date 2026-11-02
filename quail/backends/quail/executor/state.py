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
    from quail.backends.quail.executor.tokenizer import ModelTokenizer
    from quail.cost.decode import DecodeCost
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
    # a decision model's option-scoring head; None for other models
    decision_head: Any = None
    # the tokenizer AI.MAP and AI.EXTRACT read, and the word index of
    # AI.EXTRACT; None for a model that runs neither
    tokenizer: ModelTokenizer | None = None
    vocabulary: Vocabulary | None = None
    # the SAE encoder AI.SAE_FEATURES reads, loaded on first use
    sae: Any = None
    # a set-attention model's document ends joined with the state tail,
    # its tokenizer loaded on first use
    state_tails: Any = None


@dataclass
class QueryExecutionState:
    """Readouts and execution settings replaced at each query binding."""

    loaded_model: LoadedModelState
    torch: ModuleType
    async_answers: AsyncAnswers
    answer_rows: AnswerRows
    chunk_tokens: int
    async_scores: AsyncScores | None = None
    # the AI.SCORE readout of a model that does not score with answer
    # rows; None scores yes against no over answer_rows
    score_readout: Any = None
    gpu_timing: bool = False
    # rows at which a pass's dense projections become compute-bound
    knee_rows: int = 0
    decode_cost: DecodeCost | None = None


class GpuRuntime(Protocol):
    """Loaded GPU resources used by graph execution and worker dispatch."""

    torch: ModuleType
    arena: KVArena
    pipeline: ModelPipeline
    spec: ModelSpec
    execution: Any
