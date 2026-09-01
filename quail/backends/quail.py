"""Quail model backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from quail.backends.base import GpuContext
from quail.physical import AnchoredJoin, PackedFilter, PhysicalNode
from quail.planning import (
    ModelRegion,
    PhysicalCandidate,
    PlanningContext,
    SupportResult,
)


class QuailModelExecution:
    """Quail model state shared by model nodes on one GPU executor."""

    def __init__(
        self,
        context: GpuContext,
        dispatch: Callable[[PhysicalNode, Mapping[str, Any]], Any] | None = None,
    ):
        self.context = context
        self._dispatch = dispatch
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

    def set_dispatch(
        self,
        dispatch: Callable[[PhysicalNode, Mapping[str, Any]], Any],
    ) -> None:
        """Set the typed node dispatcher after the GPU loop starts."""
        self._dispatch = dispatch

    def execute(
        self,
        node: PhysicalNode,
        inputs: Mapping[str, Any],
    ) -> Any:
        if not isinstance(node, (PackedFilter, AnchoredJoin)):
            raise TypeError(
                f"Quail cannot execute physical node {node.type_name!r}")
        if self._dispatch is None:
            return self._execute_quail_node(node, inputs)
        return self._dispatch(node, inputs)

    def _execute_quail_node(
        self,
        node: PackedFilter | AnchoredJoin,
        inputs: Mapping[str, Any],
    ) -> Any:
        from quail.executor.loop import run_filter, run_join
        from quail.runtime.runner import NodeMetrics, NodeResult

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
            document_ids = list(inputs["document_ids"])
            answers, _, tokens = run_filter(
                torch,
                arena,
                pipeline,
                async_answers,
                inputs["documents"],
                [list(question) for question in node.question_token_ids],
                chunk_tokens,
                limit=inputs.get("limit"),
                arena_writes=node.arena_writes,
                arena_keys=[(node.alias, document)
                            for document in document_ids],
                retain_survivors=inputs.get("retain_survivors", ()),
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

        answers, _, tokens = run_join(
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
            if stage.semantics == "full":
                outputs[f"pairs:{stage.written_pos}"] = {
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
                fresh_tokens=tokens,
                extension={"answers": answers},
            ),
        )


class QuailBackend:
    """Plan and start Quail model execution."""

    name = "quail"

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
        from quail.planner.decide import _plan_quail

        plan = _plan_quail(
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
                    reason=getattr(plan, "constraint", "planning refused"),
                ),
            )
        return (
            PhysicalCandidate(
                graph=plan.graph,
                plan=plan,
                estimated_seconds=plan.estimated_seconds,
            ),
        )

    def start(self, context: GpuContext) -> QuailModelExecution:
        return QuailModelExecution(context)
