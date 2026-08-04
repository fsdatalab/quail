"""Fused shared-prefix attention for the pinned vLLM FlashInfer backend.

Replaces the per-step attention plan with a two-level FlashInfer cascade:
level 0 holds KV pages shared by a run of requests (one entry per group),
level 1 holds each request's remaining pages. FlashInfer merges the two
partial results by log-sum-exp, so the output matches ordinary attention.

Two ways to drive it, both off by default:

  set_cascade_groups(groups)   The caller states the groups explicitly
                               (page ids and counts). Every claim is
                               checked against the batch; a wrong claim
                               raises instead of computing a wrong answer.
  set_fused_enabled(True)      Groups are derived from the batch itself:
                               consecutive requests whose block tables
                               start with the same physical pages share
                               those pages by definition, so splitting
                               them into a shared level is always sound.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CascadeGroup:
    """A contiguous run of requests whose sequences start with the same
    KV pages.

    shared_blocks counts whole pages only. A shared prefix that ends
    mid-page must be floored to the page boundary by the caller; the
    partial page belongs to each request's unique pages. FlashInfer can
    only end a non-final cascade level on a page boundary, so a ceil
    count cannot be represented and is rejected at plan time.
    """

    request_count: int
    shared_blocks: int
    shared_page_ids: tuple[int, ...]


_CURRENT_GROUPS: tuple[CascadeGroup, ...] | None = None
_FUSED_ENABLED: bool = False
_STATS = {"cascade_steps": 0, "fallback_steps": 0, "grouped_requests": 0,
          "multi_groups": 0, "singleton_groups": 0}


def set_cascade_groups(groups: Sequence[CascadeGroup] | None) -> None:
    global _CURRENT_GROUPS
    _CURRENT_GROUPS = tuple(groups) if groups is not None else None


def set_fused_enabled(enabled: bool) -> None:
    """Turn block-table-derived grouping on or off for later steps."""
    global _FUSED_ENABLED
    _FUSED_ENABLED = bool(enabled)


def fused_stats() -> dict:
    """Counters since the last reset. A gate must see cascade_steps > 0,
    or its fused-versus-unfused comparison proved nothing."""
    return dict(_STATS)


def reset_fused_stats() -> None:
    for key in _STATS:
        _STATS[key] = 0


def _check_group(
    group: CascadeGroup,
    first_request: int,
    block_tables,
    sequence_lengths,
    query_lengths,
    page_size: int,
) -> None:
    """Reject any group the cascade plan cannot represent faithfully.

    Both checks protect the same two facts the plan relies on: every
    shared page is completely full of KV (level 0 is planned with
    last_page_len = page_size), and every query token sits after the
    shared region (level 0 is planned without a causal mask, so a query
    token inside the shared region would see later shared tokens).
    """
    end = first_request + group.request_count
    shared_tokens = group.shared_blocks * page_size
    for request in range(first_request, end):
        if tuple(
            block_tables[request, : group.shared_blocks].tolist()
        ) != group.shared_page_ids:
            raise RuntimeError(
                "cascade group pages do not match request block tables"
            )
        # Computed tokens can include pages another request of this same
        # batch is writing this step: vLLM registers full blocks in the
        # prefix cache at allocation, before their KV exists. That is
        # sound here for the same reason it is sound for ordinary
        # attention -- the runner scatters every scheduled token's KV
        # before any attention kernel of the layer reads it.
        computed = int(sequence_lengths[request]) - int(
            query_lengths[request]
        )
        if computed < shared_tokens:
            raise RuntimeError(
                "cascade group shared prefix reaches into query tokens; "
                "shared_blocks must be floored to whole computed pages"
            )


def _explicit_groups(
    groups: Sequence[CascadeGroup],
    request_count: int,
    block_tables,
) -> list[tuple[int, CascadeGroup]]:
    """Place caller-stated groups by matching their pages against the
    batch, then require the placements to tile the batch exactly.

    The model runner may reorder requests between scheduling and
    attention, so the caller's group order cannot be trusted; the shared
    page ids identify each group's requests instead.
    """
    if sum(group.request_count for group in groups) != request_count:
        raise RuntimeError(
            "cascade group request count does not match attention batch"
        )
    placed = []
    for group in groups:
        if group.request_count < 1 or group.shared_blocks < 1:
            # A zero-page shared level would hand FlashInfer an empty
            # kv row, a shape vLLM never plans; leave such requests to
            # ordinary attention instead.
            raise RuntimeError(
                "cascade groups need at least one request and one "
                "fully shared page"
            )
        if len(group.shared_page_ids) != group.shared_blocks:
            raise RuntimeError(
                "cascade group page ids do not match shared_blocks"
            )
        matching = [
            request
            for request in range(request_count)
            if tuple(
                block_tables[request, : group.shared_blocks].tolist()
            ) == group.shared_page_ids
        ]
        if len(matching) != group.request_count:
            raise RuntimeError(
                "cascade group pages do not match request block tables"
            )
        if matching != list(range(matching[0], matching[-1] + 1)):
            raise RuntimeError("cascade group requests are not contiguous")
        placed.append((matching[0], group))
    placed.sort(key=lambda row: row[0])
    offset = 0
    for first_request, group in placed:
        if first_request != offset:
            raise RuntimeError("cascade groups do not cover request order")
        offset += group.request_count
    return placed


def _derived_groups(
    request_count: int,
    block_tables,
    sequence_lengths,
    query_lengths,
    page_size: int,
) -> list[tuple[int, CascadeGroup]] | None:
    """Group consecutive requests by their common leading pages.

    Matching physical page ids mean the same KV memory, so grouping can
    never pair requests with different prefixes. Returns None when any
    request has no fully computed shared page, in which case the step
    runs with ordinary attention.
    """
    groups: list[tuple[int, CascadeGroup]] = []
    start = 0
    while start < request_count:
        end = start + 1
        prefix = block_tables[start].tolist()
        # Longest run of requests agreeing on the leading pages, and the
        # longest page prefix they all agree on.
        common = len(prefix)
        while end < request_count:
            row = block_tables[end].tolist()
            agree = 0
            for a, b in zip(prefix, row):
                if a != b:
                    break
                agree += 1
            if agree == 0:
                break
            common = min(common, agree)
            end += 1
        shared_blocks = common
        for request in range(start, end):
            computed = int(sequence_lengths[request]) - int(
                query_lengths[request]
            )
            shared_blocks = min(shared_blocks, computed // page_size)
        if shared_blocks < 1:
            return None
        groups.append((
            start,
            CascadeGroup(
                request_count=end - start,
                shared_blocks=shared_blocks,
                shared_page_ids=tuple(prefix[:shared_blocks]),
            ),
        ))
        start = end
    return groups


def install_fused_patch() -> None:
    import torch
    from vllm.v1.attention.backends.flashinfer import (
        FlashInferMetadataBuilder,
    )

    if getattr(
        FlashInferMetadataBuilder,
        "_docengine_fused_installed",
        False,
    ):
        return
    original = FlashInferMetadataBuilder.build

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        explicit = _CURRENT_GROUPS
        if not explicit and not _FUSED_ENABLED:
            return original(
                self,
                common_prefix_len,
                common_attn_metadata,
                fast_build,
            )
        if common_attn_metadata.causal is not True:
            return original(
                self,
                common_prefix_len,
                common_attn_metadata,
                fast_build,
            )
        metadata = original(
            self,
            0,
            common_attn_metadata,
            fast_build,
        )
        request_count = common_attn_metadata.num_reqs
        page_size = self.page_size
        block_tables = common_attn_metadata.block_table_tensor.cpu()
        query_starts = common_attn_metadata.query_start_loc_cpu
        sequence_lengths = common_attn_metadata.seq_lens_cpu
        query_lengths = query_starts[1:] - query_starts[:-1]
        if explicit:
            ordered = _explicit_groups(
                explicit,
                request_count,
                block_tables,
            )
        else:
            derived = _derived_groups(
                request_count,
                block_tables,
                sequence_lengths,
                query_lengths,
                page_size,
            )
            if derived is None:
                _STATS["fallback_steps"] += 1
                return metadata
            ordered = derived
        for first_request, group in ordered:
            _check_group(
                group,
                first_request,
                block_tables,
                sequence_lengths,
                query_lengths,
                page_size,
            )
        top_query_starts = [0]
        shared_indptr = [0]
        shared_indices = []
        unique_indptr = [0]
        unique_indices = []
        unique_last_page = []
        for first_request, group in ordered:
            group_end = first_request + group.request_count
            top_query_starts.append(int(query_starts[group_end]))
            shared_indices.extend(group.shared_page_ids)
            shared_indptr.append(len(shared_indices))
            for request in range(first_request, group_end):
                sequence_tokens = int(sequence_lengths[request])
                total_blocks = (
                    sequence_tokens + page_size - 1
                ) // page_size
                unique = block_tables[
                    request,
                    group.shared_blocks:total_blocks,
                ].tolist()
                unique_indices.extend(unique)
                unique_indptr.append(len(unique_indices))
                unique_last_page.append(
                    sequence_tokens % page_size or page_size
                )
        wrapper = self._get_cascade_wrapper()
        wrapper.plan(
            qo_indptr_arr=[
                torch.tensor(top_query_starts, dtype=torch.int32),
                query_starts,
            ],
            paged_kv_indptr_arr=[
                torch.tensor(shared_indptr, dtype=torch.int32),
                torch.tensor(unique_indptr, dtype=torch.int32),
            ],
            paged_kv_indices_arr=[
                torch.tensor(shared_indices, dtype=torch.int32),
                torch.tensor(unique_indices, dtype=torch.int32),
            ],
            paged_kv_last_page_len=[
                # Every shared page is full: _check_group required each
                # request's computed tokens to cover shared_blocks whole
                # pages.
                torch.full(
                    (len(ordered),),
                    page_size,
                    dtype=torch.int32,
                ),
                torch.tensor(unique_last_page, dtype=torch.int32),
            ],
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=page_size,
            # FlashInfer applies causal to the last level only and plans
            # every earlier level without a mask (cascade.py, all
            # releases since v0.2.5), which is exactly the two-pass
            # shape vLLM's own cascade path uses. A single flag here is
            # correct; do not split it per level.
            causal=True,
            sm_scale=self.sm_scale,
            window_left=self.window_left,
            logits_soft_cap=self.logits_soft_cap,
            # The cascade run path receives the query exactly as the
            # model produced it: vLLM quantizes q to fp8 only inside the
            # non-cascade trtllm paths, so the plan must state the model
            # dtype, not q_data_type_prefill (fp8 on SM90 with fp8 KV).
            q_data_type=self.model_config.dtype,
            # The cascade run call cannot pass per-layer k/v/q scales
            # (FlashInfer's cascade wrapper has no scale arguments), so
            # a quantized KV cache is only read correctly when those
            # scales are 1.0 -- true for fp8 KV without checkpoint
            # calibration, which vLLM warns about at load. A checkpoint
            # that ships calibrated kv scales must not use this patch.
            kv_data_type=self.kv_cache_dtype,
        )
        metadata.use_cascade = True
        metadata.prefill = None
        metadata.decode = None
        metadata.cascade_wrapper = wrapper
        _STATS["cascade_steps"] += 1
        _STATS["grouped_requests"] += sum(
            group.request_count
            for _, group in ordered
            if group.request_count > 1
        )
        _STATS["multi_groups"] += sum(
            1 for _, group in ordered if group.request_count > 1
        )
        _STATS["singleton_groups"] += sum(
            1 for _, group in ordered if group.request_count == 1
        )
        return metadata

    FlashInferMetadataBuilder.build = build
    FlashInferMetadataBuilder._docengine_fused_installed = True
