"""Quail model backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from quail.backends.base import GpuContext
from quail.backends.quail.executor.models import supported_archs
from quail.backends.quail.executor.operators.filter import execute_filter
from quail.backends.quail.executor.operators.join import execute_join
from quail.backends.quail.executor.pipeline import execute_pipeline
from quail.backends.quail.executor.score import QuailScorer
from quail.backends.quail.executor.state import LoadedModelState, QueryExecutionState
from quail.backends.quail.worker import execute_quail_request, prepare_quail_request
from quail.execution.reranker import RerankerModelExecution
from quail.logical import has_score, shared_preamble
from quail.logical.prompts import true_false_token_ids
from quail.physical import AiFilter, AiJoin, AiScore, Barrier, PhysicalNode
from quail.planner import plan_quail
from quail.planner.classify import has_label, joined_classification_refusal
from quail.planner.physical_optimizer import (
    ModelRegion,
    PhysicalCandidate,
    PlanningContext,
    SupportResult,
)
from quail.planner.plan import Refusal
from quail.planner.reranker import plan_reranker


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

    def bind_loaded_model(self, *, model, arena, pipeline) -> None:
        """Attach the loaded model and discard the previous model's caches."""
        self.close()
        self.loaded_model = LoadedModelState(
            model=model, arena=arena, pipeline=pipeline,
            model_spec=getattr(self.context, "model", None))

    def close(self) -> None:
        """Drop every reference to the loaded model so its memory can go."""
        self._reranker = None
        self._query = None
        self.loaded_model = None

    def bind_query(self, *, torch, async_answers, answer_rows,
                   chunk_tokens: int) -> None:
        """Attach state that is valid for the current query.

        answer_rows are the retained output rows every readout scores
        against; async_answers is the TRUE/FALSE readout over them.
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
        )

    def execute(
        self,
        node: PhysicalNode,
        inputs: Mapping[str, Any],
    ) -> Any:
        if isinstance(node, AiScore):
            execution = self._reranker_execution(inputs["documents"])
            self.query.gpu_timing = bool(inputs.get("gpu_timing", False))
            if "score_rows" in inputs:
                return execution.execute_rows(node, inputs["score_rows"])
            return execution.execute(node, inputs["score_inputs"])
        if not isinstance(node, (AiFilter, AiJoin)):
            raise TypeError(
                f"Quail cannot execute physical node {node.type_name!r}")
        if isinstance(node, AiFilter):
            return execute_filter(self.query, node, inputs)
        return execute_join(self.query, node, inputs)

    def _reranker_execution(self, documents) -> RerankerModelExecution:
        """Return the reranker bound to this query's document tokens."""
        execution = self._reranker
        if execution is None or execution.documents is not documents:
            execution = RerankerModelExecution(GpuContext(
                gpu_index=self.context.gpu_index,
                gpu_count=self.context.gpu_count,
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


def expected_join_nodes(plan) -> tuple[PhysicalNode, ...]:
    """Return the join nodes selected from Quail's planning estimates."""
    return tuple(
        node for node in plan.nodes if isinstance(node, (AiJoin, Barrier))
    )


def expected_join_stages(plan) -> tuple:
    """Return Quail's expected join stages in execution order."""
    return tuple(
        stage
        for node in expected_join_nodes(plan)
        if isinstance(node, AiJoin)
        for stage in node.stages
    )


class QuailBackend:
    """Plan and start Quail model execution."""

    name = "quail"
    runtime_package = "vllm==0.26.0"

    def supports(self, model, device, gpu_count: int) -> SupportResult:
        if device.name not in {"h100-sxm", "rtx-pro-6000-blackwell-server"}:
            return SupportResult.reject(
                f"Quail does not support device {device.name!r}")
        if model.arch not in supported_archs():
            return SupportResult.reject(
                f"Quail does not support model {model.name!r}: "
                f"no forward pass for architecture {model.arch!r}")
        if gpu_count not in {1, 2, 4, 8}:
            return SupportResult.reject(
                "Quail requires 1, 2, 4, or 8 GPUs")
        return SupportResult.accept()

    def plan(
        self,
        region: ModelRegion,
        context: PlanningContext,
    ) -> tuple[PhysicalCandidate, ...]:

        if has_label(region.logical_plan):
            if context.model.role == "reranker":
                return (PhysicalCandidate(None, Refusal(
                    reasons=("a reranker model cannot run AI.CLASSIFY",),
                    constraint="reranker_only_scores", needed=1, available=0,
                    unit="AI.CLASSIFY expressions"), float("inf")),)
        scored = has_score(region.logical_plan)
        if context.model.role == "reranker":
            if not scored:
                refusal = Refusal(
                    reasons=("a reranker model can only be used with AI.SCORE",),
                    constraint="reranker_only_scores",
                    needed=1,
                    available=0,
                    unit="AI.SCORE expressions",
                )
                return (PhysicalCandidate(None, refusal, float("inf")),)
        if scored:
            return plan_reranker(region, context, backend_name=self.name)

        plan = plan_quail(
            region.logical_plan,
            model=context.model,
            device=context.device,
            doc_tokens=context.document_tokens,
            gpus=context.gpu_count,
            order=context.order,
            pair_fractions=context.pair_fractions,
            context=context,
        )
        if not hasattr(plan, "graph"):
            return (
                PhysicalCandidate(
                    graph=None,
                    plan=plan,
                    estimated_seconds=float("inf"),
                ),
            )
        refusal = joined_classification_refusal(plan.graph, context.gpu_count)
        if refusal is not None:
            return (PhysicalCandidate(None, refusal, float("inf")),)
        plan = self._bind_runtime_data(plan, region, context)
        return (
            PhysicalCandidate(
                graph=plan.graph,
                plan=plan,
                estimated_seconds=plan.estimated_seconds,
            ),
        )

    def start(self, context: GpuContext) -> QuailModelExecution:
        return QuailModelExecution(context)

    def _bind_runtime_data(self, plan, region, context):
        """Put tokenized prompts and answer tokens in the physical plan."""
        tokenizer = context.tokenizer
        operators = region.logical_plan.operators()
        filters, joins = operators.filters, operators.joins
        encoded_nodes = []
        for node in plan.nodes:
            if isinstance(node, AiFilter):
                predicates = filters[node.alias]
                questions = tuple(
                    tuple(predicates[stage.written_pos].prompt.tail_token_ids)
                    for stage in node.stages
                )
                if any(not question for question in questions):
                    raise ValueError("filter prompts have no token ids")
                node = replace(node, question_token_ids=questions)
            elif isinstance(node, AiJoin):
                stages = []
                for stage in node.stages:
                    prompt = joins[stage.written_pos].prompt
                    runtime_ids = {
                        alias: (tuple(label), tuple(frame))
                        for alias, label, frame in prompt.label_token_ids
                    }
                    stages.append(replace(
                        stage,
                        frame_token_ids=runtime_ids[stage.anchor][1],
                        label_token_ids=tuple(
                            (alias, runtime_ids[alias][0])
                            for alias in stage.partners
                        ),
                        tail_token_ids=tuple(prompt.tail_token_ids),
                    ))
                node = replace(node, stages=tuple(stages))
            encoded_nodes.append(node)

        true_ids, false_ids = (
            true_false_token_ids(tokenizer) if tokenizer is not None
            else ([], []))
        prompts = operators.prompts
        pre_ids = (
            list(tokenizer(shared_preamble(context.model.turn_prefix)))
            if tokenizer is not None else
            list(prompts[0].preamble_token_ids) if prompts else []
        )
        return replace(
            plan,
            nodes=tuple(encoded_nodes),
            root=plan.root,
            settings={
                **plan.settings,
                "true_ids": true_ids,
                "false_ids": false_ids,
                "pre_ids": pre_ids,
                "filter_limit": (
                    None if any(
                        isinstance(node, AiJoin)
                        for node in encoded_nodes
                    ) else region.logical_plan.root.limit
                ),
            },
        )

    def prepare_request(self, context) -> None:
        """Boot the GPU for a request before its documents are ready."""
        prepare_quail_request(context)

    def execute_request(self, context) -> Any:
        """Run one Quail request inside a compute process."""
        return execute_quail_request(context)
