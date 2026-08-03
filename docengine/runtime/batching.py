"""Exact variable-length work items and continuous batch packing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Sequence

from .kv import KVPageAllocator


class WorkKind(str, Enum):
    PREFILL = "prefill"
    INITIAL_FILTER = "initial_filter"
    FILTER = "filter"
    DECODE = "decode"
    FUSED_FILTER = "fused_filter"


@dataclass(frozen=True)
class WorkItem:
    work_id: str
    owner: Hashable
    document_id: int
    filter_start: int
    k: int
    kind: WorkKind
    token_offset: int
    total_new_tokens: int
    cached_prefix_tokens: int
    temporary_bytes: int = 0
    max_chunk_tokens: int | None = None
    useful_probability: float = 1.0
    writes_persistent_kv: bool = True
    prefix_owner: Hashable | None = None
    kv_pages_to_add: int | None = None

    @property
    def remaining_tokens(self) -> int:
        return self.total_new_tokens - self.token_offset

    def validate(self) -> None:
        if self.document_id < 0:
            raise ValueError("document_id cannot be negative")
        if self.filter_start < 0:
            raise ValueError("filter_start cannot be negative")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.token_offset < 0:
            raise ValueError("token_offset cannot be negative")
        if self.total_new_tokens <= 0:
            raise ValueError("total_new_tokens must be positive")
        if self.remaining_tokens <= 0:
            raise ValueError("work item has no remaining tokens")
        if self.cached_prefix_tokens < 0:
            raise ValueError("cached_prefix_tokens cannot be negative")
        if self.temporary_bytes < 0:
            raise ValueError("temporary_bytes cannot be negative")
        if self.max_chunk_tokens is not None and self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")
        if not 0.0 <= self.useful_probability <= 1.0:
            raise ValueError("useful_probability must be between zero and one")


@dataclass(frozen=True)
class BatchLimits:
    max_new_tokens: int
    max_sequences: int
    max_temporary_bytes: int

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.max_sequences <= 0:
            raise ValueError("max_sequences must be positive")
        if self.max_temporary_bytes < 0:
            raise ValueError("max_temporary_bytes cannot be negative")


@dataclass(frozen=True)
class BatchChunk:
    work_id: str
    owner: Hashable
    document_id: int
    filter_start: int
    k: int
    kind: WorkKind
    token_start: int
    token_end: int
    total_new_tokens: int
    cached_prefix_tokens: int
    temporary_bytes: int
    kv_pages_added: int
    useful_probability: float
    prefix_owner: Hashable | None

    @property
    def new_tokens(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class BatchPlan:
    chunks: tuple[BatchChunk, ...]
    total_new_tokens: int
    sequence_count: int
    temporary_bytes: int
    kv_pages_added: int
    unused_capacity_reason: str | None


class VariableLengthBatchPacker:
    def __init__(self, limits: BatchLimits):
        limits.validate()
        self.limits = limits

    def pack(
        self,
        ready: Sequence[WorkItem],
        kv: KVPageAllocator,
    ) -> BatchPlan:
        chunks: list[BatchChunk] = []
        total_tokens = 0
        temporary_bytes = 0
        reserved_pages = 0
        blocked_by_hbm = False
        blocked_by_temporary = False
        scheduled_owners = set()

        for item in ready:
            item.validate()
            if item.owner in scheduled_owners:
                continue
            if len(chunks) >= self.limits.max_sequences:
                break
            token_room = self.limits.max_new_tokens - total_tokens
            if token_room <= 0:
                break
            chunk_tokens = min(item.remaining_tokens, token_room)
            if item.max_chunk_tokens is not None:
                chunk_tokens = min(chunk_tokens, item.max_chunk_tokens)
            temp_room = (
                self.limits.max_temporary_bytes - temporary_bytes
            )
            if item.temporary_bytes > temp_room:
                blocked_by_temporary = True
                continue

            if item.kv_pages_to_add is not None:
                if item.token_offset != 0 or chunk_tokens != item.remaining_tokens:
                    raise ValueError(
                        "explicit KV page counts require one complete chunk"
                    )
                if item.kv_pages_to_add > kv.free_pages - reserved_pages:
                    blocked_by_hbm = True
                    continue
            elif item.writes_persistent_kv:
                free_pages = kv.free_pages - reserved_pages
                hbm_token_limit = self._tokens_that_fit(
                    item, kv, free_pages
                )
                if hbm_token_limit < chunk_tokens:
                    blocked_by_hbm = True
                chunk_tokens = min(chunk_tokens, hbm_token_limit)
            if chunk_tokens <= 0:
                blocked_by_hbm = True
                continue
            if item.kv_pages_to_add is not None:
                pages = item.kv_pages_to_add
            else:
                pages = (
                    kv.additional_pages_needed(item.owner, chunk_tokens)
                    if item.writes_persistent_kv else 0
                )
            chunks.append(BatchChunk(
                work_id=item.work_id,
                owner=item.owner,
                document_id=item.document_id,
                filter_start=item.filter_start,
                k=item.k,
                kind=item.kind,
                token_start=item.token_offset,
                token_end=item.token_offset + chunk_tokens,
                total_new_tokens=item.total_new_tokens,
                cached_prefix_tokens=item.cached_prefix_tokens,
                temporary_bytes=item.temporary_bytes,
                kv_pages_added=pages,
                useful_probability=item.useful_probability,
                prefix_owner=item.prefix_owner,
            ))
            total_tokens += chunk_tokens
            temporary_bytes += item.temporary_bytes
            reserved_pages += pages
            scheduled_owners.add(item.owner)

        reason = self._unused_reason(
            ready=ready,
            chunks=chunks,
            total_tokens=total_tokens,
            blocked_by_hbm=blocked_by_hbm,
            blocked_by_temporary=blocked_by_temporary,
        )
        return BatchPlan(
            chunks=tuple(chunks),
            total_new_tokens=total_tokens,
            sequence_count=len(chunks),
            temporary_bytes=temporary_bytes,
            kv_pages_added=reserved_pages,
            unused_capacity_reason=reason,
        )

    def _tokens_that_fit(
        self,
        item: WorkItem,
        kv: KVPageAllocator,
        free_pages: int,
    ) -> int:
        if free_pages < 0:
            return 0
        if kv.has_owner(item.owner):
            allocation = kv.allocation(item.owner)
            page_capacity = (
                len(allocation.page_ids) * kv.page_size_tokens
                - allocation.token_count
            )
        else:
            page_capacity = 0
        return page_capacity + free_pages * kv.page_size_tokens

    def _unused_reason(
        self,
        *,
        ready: Sequence[WorkItem],
        chunks: list[BatchChunk],
        total_tokens: int,
        blocked_by_hbm: bool,
        blocked_by_temporary: bool,
    ) -> str | None:
        if total_tokens >= self.limits.max_new_tokens:
            return "token_limit"
        if len(chunks) >= self.limits.max_sequences:
            return "sequence_limit"
        if blocked_by_hbm:
            return "hbm_limit"
        if blocked_by_temporary:
            return "temporary_hbm_limit"
        scheduled_ends = {
            chunk.work_id: chunk.token_end for chunk in chunks
        }
        if all(
            item.work_id in scheduled_ends
            and scheduled_ends[item.work_id] >= item.total_new_tokens
            for item in ready
        ):
            return "no_ready_useful_work"
        return "planner_choice"
