"""Quail model backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from quail.backends.base import GpuContext
from quail.backends.quail.worker import execute_quail_request, prepare_quail_request
from quail.executor import loop
from quail.logical import SHARED_PRE
from quail.physical import (
    AnchoredJoin,
    Exchange,
    PackedFilter,
    PhysicalNode,
)
from quail.planner import collect_operators, plan_quail
from quail.planning import (
    ModelRegion,
    PhysicalCandidate,
    PlanningContext,
    SupportResult,
)
from quail.runtime.runner import NodeMetrics, NodeResult
from quail.runtime.tokens import DocumentKeys


class QuailModelExecution:
    """Quail model state shared by model nodes on one GPU executor."""

    def __init__(self, context: GpuContext):
        self.context = context
        self._state: dict[str, Any] = {}

    @property
    def state(self) -> Mapping[str, Any]:
        """Return Quail's private loaded model state."""
        return self._state

    def bind_loaded_model(self, *, model, arena, pipeline) -> None:
        """Attach the loaded model objects owned by this executor."""
        self._state.update(model=model, arena=arena, pipeline=pipeline)

    def bind_query(self, *, torch, async_answers, chunk_tokens: int) -> None:
        """Attach state that is valid for the current query."""
        self._state.update(
            torch=torch,
            async_answers=async_answers,
            chunk_tokens=chunk_tokens,
        )

    def execute(
        self,
        node: PhysicalNode,
        inputs: Mapping[str, Any],
    ) -> Any:
        if not isinstance(node, (PackedFilter, AnchoredJoin)):
            raise TypeError(
                f"Quail cannot execute physical node {node.type_name!r}")
        return self._execute_quail_node(node, inputs)

    def _execute_quail_node(
        self,
        node: PackedFilter | AnchoredJoin,
        inputs: Mapping[str, Any],
    ) -> Any:

        missing = {
            "torch", "async_answers", "chunk_tokens",
            "arena", "pipeline",
        } - set(self._state)
        if missing:
            raise RuntimeError(
                f"Quail model execution is missing state {sorted(missing)}")
        torch = self._state["torch"]
        arena = self._state["arena"]
        pipeline = self._state["pipeline"]
        async_answers = self._state["async_answers"]
        chunk_tokens = self._state["chunk_tokens"]

        if isinstance(node, PackedFilter):

            document_ids = inputs["document_ids"]
            retain_survivors = inputs.get("retain_survivors", ())
            if retain_survivors is False:
                retain_survivors = ()
            answers, _, tokens = loop.run_filter(
                torch,
                arena,
                pipeline,
                async_answers,
                inputs["documents"],
                [list(question) for question in node.question_token_ids],
                chunk_tokens,
                limit=inputs.get("limit"),
                arena_writes=node.arena_writes,
                arena_keys=DocumentKeys(node.alias, document_ids),
                retain_survivors=retain_survivors,
            )
            global_answers = {
                document_ids[int(local)]: row
                for local, row in answers.items()
            }
            survivors = sorted(
                document
                for document, row in global_answers.items()
                if len(row) == len(node.question_token_ids) and all(row)
            )
            return NodeResult(
                outputs={
                    f"ids:{node.alias}": survivors,
                    f"filter_answers:{node.alias}": global_answers,
                },
                metrics=NodeMetrics(
                    input_rows=len(document_ids),
                    output_rows=len(survivors),
                    evaluated_documents=len(global_answers),
                    fresh_tokens=tokens,
                ),
            )

        answers, _, tokens = loop.run_join(
            torch,
            arena,
            pipeline,
            async_answers,
            inputs["prefixes"],
            inputs["stage_suffixes"],
            chunk_tokens,
            stage_frames=inputs["stage_frames"],
            anchor_keys=inputs["anchor_keys"],
            anchor_done=inputs["anchor_done"],
        )
        anchor_ids = list(inputs["anchor_ids"])
        group = inputs["group"]
        last = answers[-1] if answers else {}
        matched = {
            anchor_ids[int(local)]
            for local, row in last.items() if any(row)
        }
        if group[-1]["semantics"] == "anti":
            survivors = [document for document in anchor_ids
                         if document not in matched]
        else:
            survivors = [document for document in anchor_ids
                         if document in matched]
        outputs = {f"ids:{node.anchor}": survivors}
        for stage, stage_answers in zip(node.stages, answers):
            outputs[f"join_answers:{stage.written_pos}"] = {
                "rows": stage_answers,
                "anchor_index": anchor_ids,
                "partner_index": inputs["partner_indices"][
                    stage.written_pos
                ],
                "anchor": node.anchor,
                "partners": list(stage.partners),
                "semantics": stage.semantics,
                "selectivity": stage.selectivity,
                "written_pos": stage.written_pos,
            }
        return NodeResult(
            outputs=outputs,
            metrics=NodeMetrics(
                input_rows=len(anchor_ids),
                output_rows=len(survivors),
                evaluated_document_pairs=sum(
                    sum(len(row) for row in stage.values())
                    for stage in answers
                ),
                kv_hits=inputs.get("kv_round", {}).get("hits", 0),
                kv_misses=inputs.get("kv_round", {}).get("misses", 0),
                regret_tokens=inputs.get("kv_round", {}).get("regret_tokens", 0),
                fresh_tokens=tokens,
                extension={"answers": answers},
            ),
        )


def expected_join_nodes(plan) -> tuple[PhysicalNode, ...]:
    """Return the join nodes selected from Quail's planning estimates."""
    return tuple(
        node for node in plan.nodes if isinstance(node, (AnchoredJoin, Exchange))
    )


def expected_join_stages(plan) -> tuple:
    """Return Quail's expected join stages in execution order."""
    return tuple(
        stage
        for node in expected_join_nodes(plan)
        if isinstance(node, AnchoredJoin)
        for stage in node.stages
    )


class QuailBackend:
    """Plan and start Quail model execution."""

    name = "quail"
    runtime_package = "vllm==0.26.0"

    def supports(self, model, device, gpu_count: int) -> SupportResult:
        if model.name not in {"qwen3-4b-fp8", "qwen3-32b-fp8"}:
            return SupportResult.reject(
                f"Quail does not support model {model.name!r}")
        if device.name != "h100-sxm":
            return SupportResult.reject(
                f"Quail does not support device {device.name!r}")
        if gpu_count not in {1, 2, 4, 8}:
            return SupportResult.reject(
                "Quail requires 1, 2, 4, or 8 GPUs in one container")
        return SupportResult.accept()

    def plan(
        self,
        region: ModelRegion,
        context: PlanningContext,
    ) -> tuple[PhysicalCandidate, ...]:

        plan = plan_quail(
            region.logical_plan,
            model=context.model,
            device=context.device,
            doc_tokens=context.document_tokens,
            gpus=context.gpu_count,
            order=context.order,
        )
        if not hasattr(plan, "graph"):
            return (
                PhysicalCandidate(
                    graph=None,
                    plan=plan,
                    estimated_seconds=float("inf"),
                ),
            )
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
        _, filters, joins = collect_operators(region.logical_plan)
        encoded_nodes = []
        for node in plan.nodes:
            if isinstance(node, PackedFilter):
                predicates = filters[node.alias]
                questions = tuple(
                    tuple(predicates[stage.written_pos].prompt.tail_token_ids)
                    for stage in node.stages
                )
                if any(not question for question in questions):
                    raise ValueError("filter prompts have no token ids")
                node = replace(node, question_token_ids=questions)
            elif isinstance(node, AnchoredJoin):
                stages = []
                for stage in node.stages:
                    prompt = joins[stage.written_pos].predicate
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

        true_ids = set()
        false_ids = set()
        if tokenizer is not None:
            for word in ("TRUE", " TRUE", "True", " True"):
                tokens = tokenizer(word)
                if tokens:
                    true_ids.add(tokens[0])
            for word in ("FALSE", " FALSE", "False", " False"):
                tokens = tokenizer(word)
                if tokens:
                    false_ids.add(tokens[0])
        prompts = [
            predicate.prompt
            for predicates in filters.values()
            for predicate in predicates
        ] + [logical_join.predicate for logical_join in joins]
        pre_ids = (
            list(tokenizer(SHARED_PRE)) if tokenizer is not None else
            list(prompts[0].preamble_token_ids) if prompts else []
        )
        return replace(
            plan,
            nodes=tuple(encoded_nodes),
            root=plan.root,
            settings={
                **plan.settings,
                "true_ids": sorted(true_ids),
                "false_ids": sorted(false_ids),
                "pre_ids": pre_ids,
                "filter_limit": (
                    None if any(
                        isinstance(node, AnchoredJoin)
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
