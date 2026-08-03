"""Pinned vLLM model-runner adapter without the vLLM scheduler."""

from __future__ import annotations

from dataclasses import dataclass
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
    ):
        self.query = query
        self.kv = kv
        self.model_executor = model_executor
        self.sampling_params = sampling_params
        self.kv_cache_groups = int(kv_cache_groups)
        self._known: dict[str, _KnownRequest] = {}
        self._finished_pending: set[str] = set()

    def execute(
        self,
        batch: BatchPlan,
        states: Mapping[int, DocumentState],
    ) -> RunnerOutput:
        if any(chunk.kind is WorkKind.FUSED_FILTER for chunk in batch.chunks):
            raise NotImplementedError(
                "fused filter batches require the FlashInfer runner"
            )
        scheduler_output = self._build_scheduler_output(batch, states)
        started_ns = perf_counter_ns()
        model_output = self.model_executor.execute_model(scheduler_output)
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
        new_pages_to_zero = []

        for chunk in batch.chunks:
            req_id = chunk.work_id
            prompt = self._prompt_tokens(chunk, states[chunk.document_id])
            allocation = self.kv.allocation(chunk.owner)
            block_ids = list(allocation.page_ids)
            computed = chunk.cached_prefix_tokens + chunk.token_start
            num_scheduled_tokens[req_id] = chunk.new_tokens
            if chunk.kv_pages_added:
                new_pages_to_zero.extend(
                    block_ids[-chunk.kv_pages_added:]
                )
            known = self._known.get(req_id)
            if known is None:
                sampling = (
                    self.sampling_params
                    if chunk.kind in (WorkKind.FILTER, WorkKind.DECODE)
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
                    prefill_token_ids=None,
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
            new_block_ids_to_zero=sorted(set(new_pages_to_zero)),
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
            if chunk.kind not in (WorkKind.FILTER, WorkKind.DECODE):
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
