"""Pinned vLLM model-runner adapter without the vLLM scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter_ns
from typing import Any, Mapping

from .batching import BatchPlan, WorkKind
from .custom import DocumentState, RunnerOutput
from .kv import KVPageAllocator
from .protocol import FilterQuery


VLLM_COMMIT = "568afb3a13806beb53bb2e6bd518269357b237c0"


@dataclass
class _KnownRequest:
    prompt_token_ids: list[int]
    block_ids: list[int]
    computed_tokens: int
    output_tokens: int


class VLLMModelRunner:
    def __init__(
        self,
        *,
        query: FilterQuery,
        kv: KVPageAllocator,
        model_executor,
        sampling_params,
        kv_cache_groups: int = 1,
        multigroup_cascade: bool = False,
    ):
        self.query = query
        self.kv = kv
        self.model_executor = model_executor
        self.sampling_params = sampling_params
        self.kv_cache_groups = int(kv_cache_groups)
        self.multigroup_cascade = bool(multigroup_cascade)
        self._known: dict[str, _KnownRequest] = {}
        self._finished_pending: set[str] = set()

    def flush_finished(self) -> None:
        if not self._finished_pending:
            return
        SchedulerOutput = _scheduler_types()[-1]
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = set(self._finished_pending)
        self.model_executor.execute_model(output)
        self._finished_pending.clear()

    def execute(
        self,
        batch: BatchPlan,
        states: Mapping[int, DocumentState],
    ) -> RunnerOutput:
        if any(chunk.kind is WorkKind.FUSED_FILTER for chunk in batch.chunks):
            return self._execute_cascade(batch, states)
        scheduler_output = self._build_scheduler_output(batch, states)
        started_ns = perf_counter_ns()
        model_output = self.model_executor.execute_model(scheduler_output)
        needs_sampling = any(
            chunk.kind in (
                WorkKind.INITIAL_FILTER,
                WorkKind.FILTER,
                WorkKind.DECODE,
            )
            for chunk in batch.chunks
        )
        if model_output is None and needs_sampling:
            model_output = self.model_executor.sample_tokens(None)
        ended_ns = perf_counter_ns()
        answers = self._read_answers(batch, model_output)
        self._mark_completed(batch)
        return RunnerOutput(
            answers=answers,
            started_ns=started_ns,
            ended_ns=ended_ns,
            extra={
                "vllm_commit": VLLM_COMMIT,
                "scheduled_tokens": batch.total_new_tokens,
            },
        )

    def _execute_cascade(
        self,
        batch: BatchPlan,
        states: Mapping[int, DocumentState],
    ) -> RunnerOutput:
        from .flashinfer_multigroup import (
            set_cascade_groups,
        )

        if not self.multigroup_cascade and len(batch.chunks) != 1:
            raise RuntimeError(
                "verified cascade execution allows one prefix group"
            )
        scheduler_output, work_requests, groups = (
            self._build_cascade_output(
                batch,
                states,
                multigroup=self.multigroup_cascade,
            )
        )
        started_ns = perf_counter_ns()
        set_cascade_groups(groups if self.multigroup_cascade else None)
        try:
            model_output = self.model_executor.execute_model(scheduler_output)
            if model_output is None:
                model_output = self.model_executor.sample_tokens(None)
        finally:
            set_cascade_groups(None)
        ended_ns = perf_counter_ns()
        sampled = dict(
            zip(model_output.req_ids, model_output.sampled_token_ids)
        )
        answers = {}
        finished = []
        for work_id, req_ids in work_requests.items():
            work_answers = []
            for req_id in req_ids:
                token_ids = sampled.get(req_id, ())
                if not token_ids:
                    raise RuntimeError(
                        f"cascade tail {req_id} returned no answer token"
                    )
                work_answers.append(
                    1 if token_ids[0] in self.query.yes_token_ids else 0
                )
            answers[work_id] = tuple(work_answers)
            finished.extend(req_ids)
        self._finished_pending.update(finished)
        return RunnerOutput(
            answers=answers,
            started_ns=started_ns,
            ended_ns=ended_ns,
            extra={
                "vllm_commit": VLLM_COMMIT,
                "cascade": True,
                "prefix_groups": len(batch.chunks),
                "tails": sum(chunk.k for chunk in batch.chunks),
            },
        )

    def _build_cascade_output(
        self,
        batch: BatchPlan,
        states: Mapping[int, DocumentState],
        *,
        multigroup: bool,
    ):
        from .flashinfer_multigroup import CascadeGroup

        (
            CachedRequestData,
            NewRequestData,
            ScheduledEncoderInputStats,
            SchedulerOutput,
        ) = _scheduler_types()
        requests = []
        scheduled = {}
        work_requests = {}
        groups = []
        for chunk in batch.chunks:
            if chunk.kind is not WorkKind.FUSED_FILTER:
                raise ValueError("cascade batch contains non-fused work")
            state = states[chunk.document_id]
            allocation = self.kv.allocation(chunk.owner)
            common_pages = (
                chunk.cached_prefix_tokens // self.kv.page_size_tokens
            )
            shared_pages = list(allocation.page_ids[:common_pages])
            unique_pages = list(allocation.page_ids[common_pages:])
            boundary_tokens = (
                state.body_tokens - chunk.cached_prefix_tokens
            )
            req_ids = []
            page_offset = 0
            for offset in range(chunk.k):
                stage = chunk.filter_start + offset
                question = list(self.query.question_token_ids[stage])
                tail_tokens = boundary_tokens + len(question)
                tail_pages = ceil(
                    tail_tokens / self.kv.page_size_tokens
                )
                pages = (
                    shared_pages
                    + unique_pages[page_offset:page_offset + tail_pages]
                )
                page_offset += tail_pages
                req_id = f"{chunk.work_id}-tail-{offset}"
                req_ids.append(req_id)
                prompt = (
                    list(self.query.body_token_ids[state.document_id])
                    + question
                )
                requests.append(NewRequestData(
                    req_id=req_id,
                    prompt_token_ids=prompt,
                    mm_features=[],
                    sampling_params=self.sampling_params,
                    pooling_params=None,
                    block_ids=self._block_groups(pages),
                    num_computed_tokens=chunk.cached_prefix_tokens,
                    lora_request=None,
                    prompt_embeds=None,
                    prompt_is_token_ids=None,
                    prefill_token_ids=prompt,
                ))
                scheduled[req_id] = tail_tokens
            if page_offset != len(unique_pages):
                raise RuntimeError(
                    "fused KV pages were not partitioned exactly"
                )
            work_requests[chunk.work_id] = req_ids
            groups.append(CascadeGroup(
                request_count=chunk.k,
                shared_blocks=common_pages,
                shared_page_ids=tuple(shared_pages),
            ))
        output = SchedulerOutput(
            scheduled_new_reqs=requests,
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens=scheduled,
            total_num_scheduled_tokens=sum(scheduled.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=(
                [0] * self.kv_cache_groups
                if multigroup
                else [groups[0].shared_blocks] * self.kv_cache_groups
            ),
            finished_req_ids=set(self._finished_pending),
            free_encoder_mm_hashes=[],
            scheduled_encoder_input_stats=ScheduledEncoderInputStats(),
            preempted_req_ids=set(),
            has_structured_output_requests=False,
            pending_structured_output_tokens=False,
            num_invalid_spec_tokens=None,
            kv_connector_metadata=None,
            ec_connector_metadata=None,
            new_block_ids_to_zero=None,
            kv_cache_block_copies=None,
            num_spec_tokens_to_schedule=0,
        )
        self._finished_pending.clear()
        return output, work_requests, groups

    def _build_scheduler_output(
        self,
        batch: BatchPlan,
        states: Mapping[int, DocumentState],
    ):
        (
            CachedRequestData,
            NewRequestData,
            ScheduledEncoderInputStats,
            SchedulerOutput,
        ) = _scheduler_types()
        new_requests = []
        cached_ids = []
        cached_new_blocks = []
        cached_computed = []
        cached_output = []
        all_token_ids = {}
        num_scheduled_tokens = {}

        for chunk in batch.chunks:
            req_id = chunk.work_id
            prompt = self._prompt_tokens(chunk, states[chunk.document_id])
            allocation = self.kv.allocation(chunk.owner)
            block_ids = list(allocation.page_ids)
            computed = (
                chunk.token_start
                if chunk.kind is WorkKind.PREFILL
                else chunk.cached_prefix_tokens + chunk.token_start
            )
            num_scheduled_tokens[req_id] = chunk.new_tokens
            known = self._known.get(req_id)
            if known is None:
                sampling = (
                    self.sampling_params
                    if chunk.kind in (
                        WorkKind.INITIAL_FILTER,
                        WorkKind.FILTER,
                        WorkKind.DECODE,
                    )
                    else None
                )
                new_requests.append(NewRequestData(
                    req_id=req_id,
                    prompt_token_ids=prompt,
                    mm_features=[],
                    sampling_params=sampling,
                    pooling_params=None,
                    block_ids=self._block_groups(block_ids),
                    num_computed_tokens=computed,
                    lora_request=None,
                    prompt_embeds=None,
                    prompt_is_token_ids=None,
                    prefill_token_ids=prompt,
                ))
                self._known[req_id] = _KnownRequest(
                    prompt_token_ids=prompt,
                    block_ids=block_ids,
                    computed_tokens=computed,
                    output_tokens=0,
                )
            else:
                new_blocks = block_ids[len(known.block_ids):]
                cached_ids.append(req_id)
                cached_new_blocks.append(
                    self._block_groups(new_blocks) if new_blocks else None
                )
                cached_computed.append(computed)
                cached_output.append(known.output_tokens)
                all_token_ids[req_id] = prompt
                known.prompt_token_ids = prompt
                known.block_ids = block_ids
                known.computed_tokens = computed

        cached = CachedRequestData(
            req_ids=cached_ids,
            resumed_req_ids=set(),
            new_token_ids=[[] for _ in cached_ids],
            all_token_ids=all_token_ids,
            new_block_ids=cached_new_blocks,
            num_computed_tokens=cached_computed,
            num_output_tokens=cached_output,
        )
        output = SchedulerOutput(
            scheduled_new_reqs=new_requests,
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * self.kv_cache_groups,
            finished_req_ids=set(self._finished_pending),
            free_encoder_mm_hashes=[],
            scheduled_encoder_input_stats=ScheduledEncoderInputStats(),
            preempted_req_ids=set(),
            has_structured_output_requests=False,
            pending_structured_output_tokens=False,
            num_invalid_spec_tokens=None,
            kv_connector_metadata=None,
            ec_connector_metadata=None,
            new_block_ids_to_zero=None,
            kv_cache_block_copies=None,
            num_spec_tokens_to_schedule=0,
        )
        self._finished_pending.clear()
        return output

    def _prompt_tokens(
        self,
        chunk,
        state: DocumentState,
    ) -> list[int]:
        body = list(self.query.body_token_ids[state.document_id])
        if chunk.kind is WorkKind.PREFILL:
            return body
        if chunk.kind is WorkKind.INITIAL_FILTER:
            question = list(
                self.query.question_token_ids[chunk.filter_start]
            )
            return body + question
        if chunk.k != 1:
            raise NotImplementedError("standard vLLM runner supports k=1")
        question = list(self.query.question_token_ids[chunk.filter_start])
        return body + question

    def _read_answers(
        self,
        batch: BatchPlan,
        model_output,
    ) -> dict[str, tuple[int, ...]]:
        if model_output is None:
            return {}
        sampled = dict(
            zip(model_output.req_ids, model_output.sampled_token_ids)
        )
        answers = {}
        for chunk in batch.chunks:
            if chunk.kind not in (
                WorkKind.INITIAL_FILTER,
                WorkKind.FILTER,
                WorkKind.DECODE,
            ):
                continue
            token_ids = sampled.get(chunk.work_id, ())
            if token_ids:
                answers[chunk.work_id] = (
                    1 if token_ids[0] in self.query.yes_token_ids else 0,
                )
                known = self._known[chunk.work_id]
                known.output_tokens += 1
        return answers

    def _mark_completed(self, batch: BatchPlan) -> None:
        for chunk in batch.chunks:
            if chunk.token_end < chunk.total_new_tokens:
                known = self._known[chunk.work_id]
                known.computed_tokens += chunk.new_tokens
                continue
            if chunk.work_id in self._known:
                self._finished_pending.add(chunk.work_id)
                del self._known[chunk.work_id]

    def _block_groups(
        self,
        page_ids: list[int],
    ) -> tuple[list[int], ...]:
        if self.kv_cache_groups != 1:
            raise NotImplementedError(
                "hybrid KV cache groups are not supported"
            )
        return (list(page_ids),)


def _scheduler_types():
    try:
        from vllm.v1.core.sched.output import (
            CachedRequestData,
            NewRequestData,
            ScheduledEncoderInputStats,
            SchedulerOutput,
        )
    except ImportError as error:
        raise RuntimeError(
            "vLLM 0.26.0 is required for the model-runner adapter"
        ) from error
    return (
        CachedRequestData,
        NewRequestData,
        ScheduledEncoderInputStats,
        SchedulerOutput,
    )


def initialize_model_executor(vllm_config):
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_configs,
    )
    from vllm.v1.core.single_type_kv_cache_manager import (
        register_all_kvcache_specs,
    )
    from vllm.v1.executor.abstract import Executor

    executor_class = Executor.get_class(vllm_config)
    executor = executor_class(vllm_config)
    register_all_kvcache_specs(vllm_config)
    specs = executor.get_kv_cache_specs()
    available_memory = executor.determine_available_memory()
    configs = get_kv_cache_configs(
        vllm_config,
        specs,
        available_memory,
    )
    scheduler_config = generate_scheduler_kv_cache_config(configs)
    vllm_config.cache_config.num_gpu_blocks = scheduler_config.num_blocks
    if scheduler_config.kv_cache_groups:
        vllm_config.cache_config.block_size = min(
            group.kv_cache_spec.block_size
            for group in scheduler_config.kv_cache_groups
        )
    vllm_config.validate_block_size()
    executor.initialize_from_config(configs)
    return executor, scheduler_config
