"""Explicit FP8 KV page ownership."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Hashable


Owner = Hashable


class KVCapacityError(RuntimeError):
    pass


class KVOwnershipError(RuntimeError):
    pass


@dataclass(frozen=True)
class KVAllocation:
    owner: Owner
    token_count: int
    page_ids: tuple[int, ...]


@dataclass
class _OwnerState:
    token_count: int
    page_ids: list[int]


class KVPageAllocator:
    def __init__(
        self,
        *,
        total_pages: int,
        page_size_tokens: int,
        bytes_per_token: int,
        activation_reserve_bytes: int = 0,
        audit_on_mutation: bool = False,
    ):
        if total_pages <= 0:
            raise ValueError("total_pages must be positive")
        if page_size_tokens <= 0:
            raise ValueError("page_size_tokens must be positive")
        if bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive")
        if activation_reserve_bytes < 0:
            raise ValueError("activation reserve cannot be negative")
        self.total_pages = int(total_pages)
        self.page_size_tokens = int(page_size_tokens)
        self.bytes_per_token = int(bytes_per_token)
        self.activation_reserve_bytes = int(activation_reserve_bytes)
        self.audit_on_mutation = bool(audit_on_mutation)
        self._free = list(range(self.total_pages - 1, -1, -1))
        self._owners: dict[Owner, _OwnerState] = {}
        self._references: dict[int, set[Owner]] = {
            page_id: set() for page_id in range(self.total_pages)
        }
        self._temporary_bytes = 0

    @property
    def page_bytes(self) -> int:
        return self.page_size_tokens * self.bytes_per_token

    @property
    def allocated_pages(self) -> int:
        return self.total_pages - len(self._free)

    @property
    def free_pages(self) -> int:
        return len(self._free)

    @property
    def kv_bytes_used(self) -> int:
        return self.allocated_pages * self.page_bytes

    @property
    def hbm_bytes_used(self) -> int:
        return (
            self.kv_bytes_used
            + self.activation_reserve_bytes
            + self._temporary_bytes
        )

    @property
    def temporary_bytes(self) -> int:
        return self._temporary_bytes

    def allocation(self, owner: Owner) -> KVAllocation:
        state = self._owners.get(owner)
        if state is None:
            raise KVOwnershipError(f"unknown KV owner {owner!r}")
        return KVAllocation(
            owner=owner,
            token_count=state.token_count,
            page_ids=tuple(state.page_ids),
        )

    def has_owner(self, owner: Owner) -> bool:
        return owner in self._owners

    def additional_pages_needed(self, owner: Owner, token_count: int) -> int:
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        state = self._owners.get(owner)
        if state is None:
            return ceil(token_count / self.page_size_tokens)
        current = ceil(state.token_count / self.page_size_tokens)
        future = ceil(
            (state.token_count + int(token_count)) / self.page_size_tokens
        )
        return future - current

    def allocate(self, owner: Owner, token_count: int) -> KVAllocation:
        if owner in self._owners:
            raise KVOwnershipError(f"KV owner {owner!r} already exists")
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        pages_needed = ceil(token_count / self.page_size_tokens)
        page_ids = self._take_pages(pages_needed)
        self._owners[owner] = _OwnerState(
            token_count=int(token_count),
            page_ids=page_ids,
        )
        for page_id in page_ids:
            self._references[page_id].add(owner)
        self._audit()
        return self.allocation(owner)

    def extend(self, owner: Owner, token_count: int) -> KVAllocation:
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        state = self._owners.get(owner)
        if state is None:
            return self.allocate(owner, token_count)
        if token_count == 0:
            return self.allocation(owner)
        old_pages = ceil(state.token_count / self.page_size_tokens)
        new_total = state.token_count + int(token_count)
        new_pages = ceil(new_total / self.page_size_tokens)
        if state.token_count % self.page_size_tokens:
            last_page = state.page_ids[-1]
            if len(self._references[last_page]) != 1:
                raise KVOwnershipError(
                    "cannot append into a shared partial KV page"
                )
        added = self._take_pages(new_pages - old_pages)
        state.page_ids.extend(added)
        state.token_count = new_total
        for page_id in added:
            self._references[page_id].add(owner)
        self._audit()
        return self.allocation(owner)

    def extend_with_pages(
        self,
        owner: Owner,
        token_count: int,
        page_count: int,
    ) -> KVAllocation:
        if token_count < 0 or page_count < 0:
            raise ValueError("token and page counts cannot be negative")
        state = self._owners.get(owner)
        if state is None:
            state = _OwnerState(token_count=0, page_ids=[])
            self._owners[owner] = state
        minimum = self.additional_pages_needed(owner, token_count)
        if page_count < minimum:
            raise KVOwnershipError(
                f"{page_count} pages cannot hold {token_count} new KV tokens"
            )
        added = self._take_pages(page_count)
        state.page_ids.extend(added)
        state.token_count += int(token_count)
        for page_id in added:
            self._references[page_id].add(owner)
        self._audit()
        return self.allocation(owner)

    def share_prefix(
        self,
        source: Owner,
        target: Owner,
        token_count: int,
    ) -> KVAllocation:
        if target in self._owners:
            raise KVOwnershipError(f"KV owner {target!r} already exists")
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        if token_count % self.page_size_tokens:
            raise KVOwnershipError(
                "shared KV prefixes must end on a page boundary"
            )
        source_state = self._owners.get(source)
        if source_state is None:
            raise KVOwnershipError(f"unknown KV owner {source!r}")
        if token_count > source_state.token_count:
            raise KVOwnershipError("shared prefix exceeds source KV length")
        pages = token_count // self.page_size_tokens
        page_ids = list(source_state.page_ids[:pages])
        self._owners[target] = _OwnerState(
            token_count=int(token_count),
            page_ids=page_ids,
        )
        for page_id in page_ids:
            self._references[page_id].add(target)
        self._audit()
        return self.allocation(target)

    def truncate(self, owner: Owner, token_count: int) -> KVAllocation:
        state = self._owners.get(owner)
        if state is None:
            raise KVOwnershipError(f"unknown KV owner {owner!r}")
        if token_count < 0 or token_count > state.token_count:
            raise ValueError("invalid truncated token count")
        keep_pages = ceil(token_count / self.page_size_tokens)
        removed = state.page_ids[keep_pages:]
        del state.page_ids[keep_pages:]
        state.token_count = int(token_count)
        self._drop_references(owner, removed)
        self._audit()
        return self.allocation(owner)

    def free(self, owner: Owner) -> None:
        state = self._owners.pop(owner, None)
        if state is None:
            raise KVOwnershipError(f"unknown KV owner {owner!r}")
        self._drop_references(owner, state.page_ids)
        self._audit()

    def reserve_temporary(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("temporary byte count cannot be negative")
        self._temporary_bytes += int(byte_count)

    def release_temporary(self, byte_count: int) -> None:
        if byte_count < 0 or byte_count > self._temporary_bytes:
            raise ValueError("invalid temporary byte release")
        self._temporary_bytes -= int(byte_count)

    def _take_pages(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("page count cannot be negative")
        if count > len(self._free):
            raise KVCapacityError(
                f"need {count} KV pages with {len(self._free)} free"
            )
        return [self._free.pop() for _ in range(count)]

    def _audit(self) -> None:
        if self.audit_on_mutation:
            self.validate()

    def _drop_references(self, owner: Owner, page_ids: list[int]) -> None:
        for page_id in page_ids:
            references = self._references[page_id]
            if owner not in references:
                raise KVOwnershipError(
                    f"owner {owner!r} does not reference page {page_id}"
                )
            references.remove(owner)
            if not references:
                self._free.append(page_id)

    def validate(self) -> None:
        free = set(self._free)
        if len(free) != len(self._free):
            raise KVOwnershipError("duplicate KV page in free list")
        for page_id in range(self.total_pages):
            references = self._references[page_id]
            if bool(references) == (page_id in free):
                raise KVOwnershipError(
                    f"KV page {page_id} free/reference state is inconsistent"
                )
        for owner, state in self._owners.items():
            expected = ceil(state.token_count / self.page_size_tokens)
            if expected > len(state.page_ids):
                raise KVOwnershipError(
                    f"owner {owner!r} has wrong KV page count"
                )
            for page_id in state.page_ids:
                if owner not in self._references[page_id]:
                    raise KVOwnershipError(
                        f"owner {owner!r} missing page reference"
                    )
