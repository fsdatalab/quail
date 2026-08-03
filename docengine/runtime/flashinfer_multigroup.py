"""Multi-document shared-prefix metadata for pinned vLLM FlashInfer."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CascadeGroup:
    request_count: int
    shared_blocks: int


_CURRENT_GROUPS: tuple[CascadeGroup, ...] | None = None


def set_cascade_groups(groups: Sequence[CascadeGroup] | None) -> None:
    global _CURRENT_GROUPS
    _CURRENT_GROUPS = tuple(groups) if groups is not None else None


def install_multigroup_patch() -> None:
    import torch
    from vllm.v1.attention.backends.flashinfer import (
        FlashInferMetadataBuilder,
    )

    if getattr(
        FlashInferMetadataBuilder,
        "_docengine_multigroup_installed",
        False,
    ):
        return
    original = FlashInferMetadataBuilder.build

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        groups = _CURRENT_GROUPS
        if not groups:
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
        request_count = sum(group.request_count for group in groups)
        if request_count != common_attn_metadata.num_reqs:
            raise RuntimeError(
                "cascade group request count does not match attention batch"
            )
        page_size = self.page_size
        block_tables = common_attn_metadata.block_table_tensor.cpu()
        query_starts = common_attn_metadata.query_start_loc_cpu
        sequence_lengths = common_attn_metadata.seq_lens_cpu
        top_query_starts = [0]
        shared_indptr = [0]
        shared_indices = []
        unique_indptr = [0]
        unique_indices = []
        unique_last_page = []
        request_offset = 0
        for group in groups:
            group_end = request_offset + group.request_count
            top_query_starts.append(int(query_starts[group_end]))
            first_table = block_tables[request_offset]
            shared = first_table[:group.shared_blocks].tolist()
            shared_indices.extend(shared)
            shared_indptr.append(len(shared_indices))
            for request in range(request_offset, group_end):
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
                unique_tokens = (
                    sequence_tokens - group.shared_blocks * page_size
                )
                unique_last_page.append(
                    unique_tokens % page_size or page_size
                )
            request_offset = group_end
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
                torch.full(
                    (len(groups),),
                    page_size,
                    dtype=torch.int32,
                ),
                torch.tensor(unique_last_page, dtype=torch.int32),
            ],
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=page_size,
            causal=True,
            sm_scale=self.sm_scale,
            window_left=self.window_left,
            logits_soft_cap=self.logits_soft_cap,
            q_data_type=self.q_data_type_prefill,
            kv_data_type=self.kv_cache_dtype,
        )
        metadata.use_cascade = True
        metadata.prefill = None
        metadata.decode = None
        metadata.cascade_wrapper = wrapper
        return metadata

    FlashInferMetadataBuilder.build = build
    FlashInferMetadataBuilder._docengine_multigroup_installed = True
