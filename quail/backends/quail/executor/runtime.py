"""Model state and operator dispatch for the Quail executor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quail.backends.base import GpuContext
from quail.backends.quail.executor.operators import execution_parts
from quail.backends.quail.executor.operators.score import QuailScorer
from quail.backends.quail.executor.pipeline import execute_pipeline
from quail.backends.quail.executor.state import LoadedModelState, QueryExecutionState
from quail.backends.quail.reranker import RerankerModelExecution
from quail.cost.budgets import compute_knee
from quail.physical import AiClassify, AiScore, PhysicalNode


class QuailModelExecution:
    """Quail model state shared by model nodes on one GPU executor."""

    def __init__(self, context: GpuContext):
        self.context = context
        self.loaded_model: LoadedModelState | None = None
        self._query: QueryExecutionState | None = None
        self._reranker: RerankerModelExecution | None = None

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
                          decision_head=None) -> None:
        """Attach the loaded model and discard the previous model's caches."""
        self.close()
        self.loaded_model = LoadedModelState(
            model=model, arena=arena, pipeline=pipeline,
            model_spec=self.context.model,
            decision_head=decision_head)

    def close(self) -> None:
        """Drop every reference to the loaded model so its memory can go."""
        self._reranker = None
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
        if loaded.input_staging is not None:
            # Retain allocated transfer buffers, not the previous query's tokens.
            loaded.input_staging.fixed_tokens.clear()
        self._reranker = None
        self._query = QueryExecutionState(
            loaded_model=loaded,
            torch=torch,
            async_answers=async_answers,
            answer_rows=answer_rows,
            chunk_tokens=chunk_tokens,
            score_readout=async_scores,
            knee_rows=self._knee_rows(),
        )

    def _knee_rows(self) -> int:
        """Rows at which a pass becomes compute-bound, 0 without a device."""
        device = getattr(self.context, "device", None)
        return 0 if device is None else int(compute_knee(self.context.model, device))

    def execute(
        self,
        node: PhysicalNode,
        inputs: Mapping[str, Any],
    ) -> Any:
        implementation = execution_parts().get(node.type_name)
        if implementation is not None and implementation.execute_group is not None:
            return implementation.execute_group(
                self.query, (node,), inputs)[node.node_id]
        if isinstance(node, (AiScore, AiClassify)):
            # a score or classification on its own runs through the
            # reranker readout, not the document scheduler
            execution = self._reranker_execution(inputs["documents"])
            self.query.gpu_timing = bool(inputs.get("gpu_timing", False))
            return execution.execute(node, inputs["score_inputs"])
        raise TypeError(
            f"Quail cannot execute physical node {node.type_name!r}")

    def _reranker_execution(self, documents) -> RerankerModelExecution:
        """Return the reranker bound to this query's document tokens."""
        execution = self._reranker
        if execution is None or execution.documents is not documents:
            execution = RerankerModelExecution(GpuContext(
                gpu_index=self.context.gpu_index,
                gpus=self.context.gpus,
                model=self.context.model, device=self.context.device,
                query_settings={
                    "documents": documents,
                    "reranker": QuailScorer(self.query),
                },
            ))
            self._reranker = execution
        return execution

    def execute_pipeline(self, pipeline, inputs, context) -> dict:
        """Execute a model pipeline with this executor's loaded state."""
        return execute_pipeline(self.query, pipeline, inputs, context)
