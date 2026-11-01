"""Tests for model specs and budget calculations on Qwen3-4B / H100."""

from dataclasses import replace

import pytest

from quail.cost import budgets
from quail.specs import (
    B200,
    H100_SXM,
    L40S,
    QWEN3_4B_FP8,
    QWEN3_5_4B_BF16,
    QWEN3_32B_FP8,
    RTX_PRO_6000_BLACKWELL_SERVER,
)


def test_model_weights_and_kv_memory_budgets():
    # Tied output weights must remain available for input embeddings.
    assert QWEN3_4B_FP8.head_mem_bytes == 0.0
    assert QWEN3_4B_FP8.W_resident == QWEN3_4B_FP8.W_mem
    # The separate 32B output head stays resident for AI.CLASSIFY: its
    # 1,555,824,640 bytes are 5,936 KV tokens (262,144 bytes each, in
    # whole 16-token pages) the arena does not get.
    head = QWEN3_32B_FP8.head_mem_bytes
    assert head == 151_936 * 5_120 * 2
    assert QWEN3_32B_FP8.W_resident == QWEN3_32B_FP8.W_mem
    assert budgets.arena_tokens(QWEN3_32B_FP8, H100_SXM) == 112_304 - 5_936
    # the chunk budget does not move: the 32B chunk is capped by the
    # int32 kernel index, not by memory
    kept = replace(QWEN3_32B_FP8, tied_head=True)
    assert budgets.chunk_budget(QWEN3_32B_FP8, H100_SXM) == \
        budgets.chunk_budget(kept, H100_SXM)

    tokens = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM)
    assert tokens == 362_240     # 22,640 pages of 16 tokens
    assert tokens // 400 == 905
    fp8 = budgets.arena_tokens(replace(QWEN3_4B_FP8, kv_bytes=1.0), H100_SXM)
    assert fp8 == pytest.approx(2 * tokens, rel=0.01)
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        assert budgets.arena_tokens(model, RTX_PRO_6000_BLACKWELL_SERVER) \
            > budgets.arena_tokens(model, H100_SXM)


def test_device_memory_budgets():
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        assert budgets.arena_tokens(model, B200) > budgets.arena_tokens(model, H100_SXM)
        assert budgets.arena_tokens(model, L40S) < budgets.arena_tokens(model, H100_SXM)
    assert budgets.minimum_weight_gpus(QWEN3_32B_FP8, L40S) == 1
    # The int32 kernel index still caps 32B on H100 and B200. On L40S
    # the memory bound would spend the leftover on two activation
    # chunks, so the chunk shrinks until the arena holds that chunk.
    cap = budgets.kernel_index_cap(QWEN3_32B_FP8)
    assert budgets.chunk_budget(QWEN3_32B_FP8, H100_SXM) == cap
    assert budgets.chunk_budget(QWEN3_32B_FP8, B200) == cap
    l40s_chunk = budgets.chunk_budget(QWEN3_32B_FP8, L40S)
    assert l40s_chunk < cap
    full, sliding = budgets.arena_pages(QWEN3_32B_FP8, L40S, l40s_chunk)
    assert sliding == 0
    assert full * budgets.PAGE_TOKENS >= l40s_chunk
















@pytest.mark.parametrize("tokens,expected_slots", [(200, 611)])
def test_hybrid_budget_reserves_checkpoints_and_two_execution_chunks(
        tokens, expected_slots):
    from quail.backends.quail.worker import checkpoint_arena_args
    from quail.cost.retention import retention_pages

    model = replace(QWEN3_5_4B_BF16, w_mem_bytes=8_411_152_384)
    chunk = budgets.chunk_budget(model, H100_SXM)
    assert chunk == 116_508
    assert model.checkpoint_bytes == 51_511_296
    assert budgets.checkpoint_pass_bytes(model, 1_024, chunk) == 4_941 * 2**20
    pool = budgets.hybrid_pool_pages(model, H100_SXM, chunk)
    assert pool == 68_060
    kv_pages, sliding = budgets.arena_pages(model, H100_SXM, chunk, tokens)
    assert sliding == 0
    slots = budgets.checkpoint_slots(model, H100_SXM, chunk, kv_pages)
    assert slots == expected_slots
    cost = -(-tokens // 16) + budgets.checkpoint_pages(model)
    assert slots * cost <= pool < (slots + 1) * cost
    retained = retention_pages(kv_pages * 16, chunk, 16)
    assert retained == kv_pages - budgets.execution_pages(chunk)
    assert retained + slots * budgets.checkpoint_pages(model) <= pool
    assert checkpoint_arena_args(model, H100_SXM, chunk, kv_pages) == {
        "num_state_slots": slots,
        "gdn_state_shape": (24, 32, 128, 128),
        "conv_state_shape": (24, 3, 8_192),
    }
    with pytest.raises(ValueError, match="holds no checkpoint"):
        budgets.hybrid_arena_pages(model, H100_SXM, chunk, 1e7)








