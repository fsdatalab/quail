"""Quail model backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from quail.backends.base import GpuContext
from quail.backends.quail.graph import filter_result, stage_partner_lists
from quail.backends.quail.worker import execute_quail_request, prepare_quail_request
from quail.executor import loop
from quail.executor.attention import FILTER_ATTENTION, JOIN_ATTENTION
from quail.logical import SHARED_PRE
from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
    PhysicalNode,
)
from quail.planner import collect_operators, plan_quail
from quail.planning import (
    ModelRegion,
    PhysicalCandidate,
    PlanningContext,
    SupportResult,
)
from quail.runtime.runner import NodeMetrics, NodeResult, SurvivorStream
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
        if not isinstance(node, (AiFilter, AiJoin)):
            raise TypeError(
                f"Quail cannot execute physical node {node.type_name!r}")
        return self._execute_quail_node(node, inputs)

    def _execute_quail_node(
        self,
        node: AiFilter | AiJoin,
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

        if isinstance(node, AiFilter):
            document_ids = inputs["document_ids"]
            if node.pin_survivors:
                # the join that consumes this chain drives it and fills
                # the holder; the result is complete once it has
                stream = SurvivorStream(node, document_ids)

                def finalize(node=node, stream=stream,
                             document_ids=document_ids):
                    if "answers" not in stream.holder:
                        raise RuntimeError(
                            f"{node.node_id!r} pins its survivors but no "
                            "join consumed its stream")
                    return filter_result(
                        node, stream.holder["answers"],
                        stream.holder["tokens"], document_ids)

                return NodeResult(
                    outputs={
                        f"ids:{node.alias}": stream,
                        f"filter_answers:{node.alias}": {},
                    },
                    finalize=finalize,
                )
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
            return filter_result(node, answers, tokens, document_ids)

        stage_frames = inputs["stage_frames"]
        stream = inputs.get("anchor_stream")
        source = None
        if stream is not None:
            filter_node = stream["node"]
            if not filter_node.arena_writes or not filter_node.pin_survivors:
                raise ValueError(
                    "a filter chain that streams into a join must write "
                    "its KV to the arena and pin its survivors")
            filter_ids = stream["document_ids"]
            # the chain hands each passing document over with its KV
            # pinned; pages cover the join's largest frame so the join
            # never claims a page of its own for a streamed anchor
            source = loop.FilterStream(
                torch,
                arena,
                pipeline,
                async_answers,
                stream["documents"],
                [list(question)
                 for question in filter_node.question_token_ids],
                chunk_tokens,
                arena_writes=True,
                arena_keys=DocumentKeys(filter_node.alias, filter_ids),
                hold_survivors=True,
                hold_extra_tokens=max(
                    filter_node.hold_tokens,
                    *(len(frame) for frame in stage_frames)),
                attention_mode=FILTER_ATTENTION,
            )
        lists_for = inputs.get("anchor_partners")
        answers, _, tokens = loop.run_join(
            torch,
            arena,
            pipeline,
            async_answers,
            inputs["prefixes"],
            inputs["stage_suffixes"],
            chunk_tokens,
            stage_frames=stage_frames,
            anchor_keys=inputs["anchor_keys"],
            anchor_done=inputs["anchor_done"],
            anchor_source=source,
            attention_mode=JOIN_ATTENTION if source is not None else None,
            anchor_partners=(
                None if lists_for is None else lambda key: lists_for(key[1])),
        )
        if source is not None:
            anchor_ids = [filter_ids[document] for document in source.held]
            kv_round = {"hits": len(anchor_ids), "misses": 0,
                        "regret_tokens": 0}
            stream["holder"].update(
                answers=source.answers, tokens=source.tokens,
                held=list(source.held))
        else:
            anchor_ids = list(inputs["anchor_ids"])
            kv_round = inputs.get("kv_round") or {}
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
        partner_lists = stage_partner_lists(group, lists_for, anchor_ids)
        for stage, stage_answers, members in zip(
                node.stages, answers, partner_lists):
            outputs[f"join_answers:{stage.written_pos}"] = {
                "rows": stage_answers,
                "anchor_index": anchor_ids,
                "partner_index": inputs["partner_indices"][
                    stage.written_pos
                ],
                "anchor_partners": members,
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
                kv_hits=kv_round.get("hits", 0),
                kv_misses=kv_round.get("misses", 0),
                regret_tokens=kv_round.get("regret_tokens", 0),
                fresh_tokens=tokens,
                extension={"answers": answers},
            ),
        )


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
        if model.name not in {"qwen3-4b-fp8", "qwen3-32b-fp8"}:
            return SupportResult.reject(
                f"Quail does not support model {model.name!r}")
        if device.name not in {"h100-sxm", "rtx-pro-6000-blackwell-server"}:
            return SupportResult.reject(
                f"Quail does not support device {device.name!r}")
        if gpu_count not in {1, 2, 4, 8}:
            return SupportResult.reject(
                "Quail requires 1, 2, 4, or 8 GPUs")
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
            pair_fractions=context.pair_fractions,
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
