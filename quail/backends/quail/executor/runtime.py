"""Model state and operator dispatch for the Quail executor."""

from __future__ import annotations

from quail.backends.base import GpuContext
from quail.backends.quail.executor.operators import operator_types
from quail.backends.quail.executor.region import region_of
from quail.backends.quail.executor.state import LoadedModelState, QueryExecutionState
from quail.common.budgets import compute_knee
from quail.common.decode import DecodeCost


class QuailModelExecution:
    """Quail model state shared by model nodes on one GPU executor."""

    # the runner groups the operators a document flows through into chains
    groups_operators = True

    def __init__(self, context: GpuContext):
        self.context = context
        self.loaded_model: LoadedModelState | None = None
        self._query: QueryExecutionState | None = None
        self.operator_types = operator_types()

    @property
    def query(self) -> QueryExecutionState:
        """Return the current query state.

        Raises:
            RuntimeError: No query has been bound to this executor.
        """
        if self._query is None:
            raise RuntimeError("Quail model execution has no bound query")
        return self._query

    def bind_loaded_model(self, *, model, arena, pipeline,
                          decision_head=None, tokenizer=None) -> None:
        """Attach the loaded model and discard the previous model's caches.

        Args:
            model: The loaded model.
            arena: Its KV arena.
            pipeline: Its forward pass pipeline.
            decision_head: A decision model's option-scoring head.
            tokenizer: The model's Tokenizer, for a model that runs
                AI.MAP and AI.EXTRACT; the word index is built from it here.
        """
        from quail.backends.quail.executor.extract import Vocabulary

        self.close()
        self.loaded_model = LoadedModelState(
            model=model, arena=arena, pipeline=pipeline,
            model_spec=self.context.model,
            decision_head=decision_head, tokenizer=tokenizer,
            vocabulary=None if tokenizer is None else Vocabulary(tokenizer))

    def close(self) -> None:
        """Drop every reference to the loaded model so its memory can go."""
        self._query = None
        self.loaded_model = None

    def bind_query(self, *, torch, async_answers, answer_rows,
                   chunk_tokens: int, async_scores=None) -> None:
        """Attach state that is valid for the current query.

        answer_rows are the retained output rows every readout scores
        against; async_answers is the TRUE/FALSE readout over them.
        async_scores is the AI.SCORE readout of a model that does not
        score with answer rows, or None to score yes against no over
        answer_rows.
        """
        loaded = self.loaded_model
        if loaded is None:
            raise RuntimeError("Quail model execution has no loaded model")
        device = getattr(self.context, "device", None)
        decode_cost = (None if device is None else DecodeCost.from_specs(
            self.context.model, device))
        self._query = QueryExecutionState(
            loaded_model=loaded,
            torch=torch,
            async_answers=async_answers,
            answer_rows=answer_rows,
            chunk_tokens=chunk_tokens,
            score_readout=async_scores,
            knee_rows=self._knee_rows(), decode_cost=decode_cost,
        )

    def _knee_rows(self) -> int:
        """Rows at which a pass becomes compute-bound, 0 without a device."""
        device = getattr(self.context, "device", None)
        return 0 if device is None else int(compute_knee(self.context.model, device))

    def execute_pipeline(self, nodes, inputs, context) -> dict:
        """Run a chain of model nodes in the context's region on this GPU."""
        return region_of(self, context).run(nodes, inputs)

