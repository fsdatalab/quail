"""Single-query runtime with explicit batches and KV ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter_ns
from typing import Mapping, Protocol

from .batching import (
    BatchChunk,
    BatchPlan,
    VariableLengthBatchPacker,
    WorkItem,
    WorkKind,
)
from .kv import KVPageAllocator
from .protocol import FilterQuery
from .trace import (
    EngineStepTrace,
    ScheduledChunkTrace,
    TraceRecorder,
)


class DocumentStatus(str, Enum):
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"


@dataclass
class DocumentState:
    document_id: int
    body_tokens: int
    stage: int = 0
    prefill_offset: int = 0
    filter_offset: int = 0
    status: DocumentStatus = DocumentStatus.ACTIVE
    answers: list[int] = field(default_factory=list)

    @property
    def kv_owner(self) -> tuple[str, int]:
        return ("document", self.document_id)


@dataclass(frozen=True)
class RunnerOutput:
    answers: Mapping[str, tuple[int, ...]]
    started_ns: int
    ended_ns: int
    extra: Mapping[str, object] = field(default_factory=dict)


class BatchModelRunner(Protocol):
    def execute(
        self,
        batch: BatchPlan,
        states: Mapping[int, DocumentState],
    ) -> RunnerOutput:
        ...


@dataclass(frozen=True)
class RuntimeResult:
    answers: Mapping[tuple[int, int], int]
    survivors: tuple[int, ...]
    rejected: tuple[int, ...]
    steps: int
    wall_ns: int


class DocEngineRuntime:
    def __init__(
        self,
        *,
        query: FilterQuery,
        runner: BatchModelRunner,
        packer: VariableLengthBatchPacker,
        kv: KVPageAllocator,
        trace: TraceRecorder | None = None,
        speculation_k: int = 1,
        filter_temporary_bytes_per_token: int | None = None,
    ):
        query.validate()
        if speculation_k <= 0:
            raise ValueError("speculation_k must be positive")
        self.query = query
        self.runner = runner
        self.packer = packer
        self.kv = kv
        self.trace = trace or TraceRecorder()
        self.speculation_k = int(speculation_k)
        self.filter_temporary_bytes_per_token = int(
            filter_temporary_bytes_per_token or kv.bytes_per_token
        )
        self.states = {
            document_id: DocumentState(
                document_id=document_id,
                body_tokens=len(tokens),
            )
            for document_id, tokens in enumerate(query.body_token_ids)
        }
        self._last_step_end_ns: int | None = None

    def run(self) -> RuntimeResult:
        started_ns = perf_counter_ns()
        step = 0
        while self._has_active_documents():
            planning_started = perf_counter_ns()
            ready = self._ready_work()
            batch = self.packer.pack(ready, self.kv)
            planning_ended = perf_counter_ns()
            if not batch.chunks:
                raise RuntimeError(
                    "ready work exists but no legal batch can be packed"
                )
            self._reserve_batch(batch)
            output = self.runner.execute(batch, self.states)
            self._release_batch_temporary(batch)
            self._apply_output(batch, output)
            self._record_trace(
                step=step,
                batch=batch,
                output=output,
                planning_ns=planning_ended - planning_started,
            )
            step += 1
        ended_ns = perf_counter_ns()
        answers = {
            (state.document_id, stage + 1): answer
            for state in self.states.values()
            for stage, answer in enumerate(state.answers)
        }
        survivors = tuple(
            state.document_id
            for state in self.states.values()
            if state.status is DocumentStatus.PASSED
        )
        rejected = tuple(
            state.document_id
            for state in self.states.values()
            if state.status is DocumentStatus.REJECTED
        )
        self.kv.validate()
        if self.kv.allocated_pages:
            raise RuntimeError("KV pages remain allocated after query")
        return RuntimeResult(
            answers=answers,
            survivors=survivors,
            rejected=rejected,
            steps=step,
            wall_ns=ended_ns - started_ns,
        )

    def _has_active_documents(self) -> bool:
        return any(
            state.status is DocumentStatus.ACTIVE
            for state in self.states.values()
        )

    def _ready_work(self) -> list[WorkItem]:
        filters = []
        prefills = []
        for state in self.states.values():
            if state.status is not DocumentStatus.ACTIVE:
                continue
            if state.prefill_offset < state.body_tokens:
                prefills.append(self._prefill_work(state))
            else:
                filters.append(self._filter_work(state))
        return filters + prefills

    def _prefill_work(self, state: DocumentState) -> WorkItem:
        return WorkItem(
            work_id=f"prefill-{state.document_id}",
            owner=state.kv_owner,
            document_id=state.document_id,
            filter_start=state.stage,
            k=1,
            kind=WorkKind.PREFILL,
            token_offset=state.prefill_offset,
            total_new_tokens=state.body_tokens,
            cached_prefix_tokens=state.prefill_offset,
            writes_persistent_kv=True,
        )

    def _filter_work(self, state: DocumentState) -> WorkItem:
        k = min(
            self.speculation_k,
            self.query.n_filters - state.stage,
        )
        shared_prefix = (
            state.body_tokens // self.kv.page_size_tokens
        ) * self.kv.page_size_tokens
        boundary_tokens = state.body_tokens - shared_prefix
        new_tokens = sum(
            boundary_tokens + len(self.query.question_token_ids[stage])
            for stage in range(state.stage, state.stage + k)
        )
        return WorkItem(
            work_id=f"filter-{state.document_id}-{state.stage}-{k}",
            owner=("filter", state.document_id, state.stage, k),
            document_id=state.document_id,
            filter_start=state.stage,
            k=k,
            kind=(WorkKind.FUSED_FILTER if k > 1 else WorkKind.FILTER),
            token_offset=state.filter_offset,
            total_new_tokens=new_tokens,
            cached_prefix_tokens=shared_prefix,
            temporary_bytes=(
                (new_tokens + k) * self.filter_temporary_bytes_per_token
            ),
            writes_persistent_kv=True,
            prefix_owner=state.kv_owner,
        )

    def _reserve_batch(self, batch: BatchPlan) -> None:
        self.kv.reserve_temporary(batch.temporary_bytes)
        for chunk in batch.chunks:
            if chunk.kv_pages_added:
                if (
                    chunk.prefix_owner is not None
                    and not self.kv.has_owner(chunk.owner)
                ):
                    self.kv.share_prefix(
                        chunk.prefix_owner,
                        chunk.owner,
                        chunk.cached_prefix_tokens,
                    )
                self.kv.extend(chunk.owner, chunk.new_tokens)

    def _release_batch_temporary(self, batch: BatchPlan) -> None:
        self.kv.release_temporary(batch.temporary_bytes)

    def _apply_output(
        self,
        batch: BatchPlan,
        output: RunnerOutput,
    ) -> None:
        for chunk in batch.chunks:
            state = self.states[chunk.document_id]
            if chunk.kind is WorkKind.PREFILL:
                state.prefill_offset = chunk.token_end
                continue
            state.filter_offset = chunk.token_end
            work = self._filter_work(state)
            if state.filter_offset < work.total_new_tokens:
                continue
            returned = tuple(output.answers.get(chunk.work_id, ()))
            if len(returned) != chunk.k:
                raise RuntimeError(
                    f"runner returned {len(returned)} answers for k={chunk.k}"
                )
            state.filter_offset = 0
            self.kv.free(chunk.owner)
            passed = 0
            for answer in returned:
                value = int(answer)
                state.answers.append(value)
                if value and passed == len(state.answers) - state.stage - 1:
                    passed += 1
                else:
                    break
            if passed < chunk.k:
                state.status = DocumentStatus.REJECTED
                self.kv.free(state.kv_owner)
                continue
            state.stage += chunk.k
            if state.stage >= self.query.n_filters:
                state.status = DocumentStatus.PASSED
                self.kv.free(state.kv_owner)

    def _record_trace(
        self,
        *,
        step: int,
        batch: BatchPlan,
        output: RunnerOutput,
        planning_ns: int,
    ) -> None:
        idle_before_ns = (
            0 if self._last_step_end_ns is None
            else max(0, output.started_ns - self._last_step_end_ns)
        )
        self._last_step_end_ns = output.ended_ns
        prefill_tokens = sum(
            chunk.new_tokens
            for chunk in batch.chunks
            if chunk.kind is WorkKind.PREFILL
        )
        decode_tokens = sum(
            chunk.k
            for chunk in batch.chunks
            if chunk.kind in (WorkKind.FILTER, WorkKind.FUSED_FILTER)
        )
        self.trace.record(EngineStepTrace(
            step=step,
            started_ns=output.started_ns,
            ended_ns=output.ended_ns,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            sequence_count=batch.sequence_count,
            queued_documents=sum(
                state.status is DocumentStatus.ACTIVE
                for state in self.states.values()
            ),
            queued_filter_calls=sum(
                state.status is DocumentStatus.ACTIVE
                and state.prefill_offset >= state.body_tokens
                for state in self.states.values()
            ),
            hbm_used_bytes=self.kv.hbm_bytes_used,
            hbm_free_bytes=self.kv.free_pages * self.kv.page_bytes,
            token_capacity=self.packer.limits.max_new_tokens,
            sequence_capacity=self.packer.limits.max_sequences,
            planning_ns=planning_ns,
            input_preparation_ns=0,
            idle_before_ns=idle_before_ns,
            cache_reset_confirmed=True,
            unused_capacity_reason=batch.unused_capacity_reason,
            request_ids=tuple(chunk.work_id for chunk in batch.chunks),
            chunks=tuple(
                ScheduledChunkTrace(
                    document_id=chunk.document_id,
                    filter_start=chunk.filter_start,
                    k=chunk.k,
                    new_tokens=chunk.new_tokens,
                    cached_prefix_tokens=chunk.cached_prefix_tokens,
                    kv_pages=(
                        self.kv.allocation(chunk.owner).page_ids
                        if self.kv.has_owner(chunk.owner) else ()
                    ),
                )
                for chunk in batch.chunks
            ),
            extra=dict(output.extra),
        ))
