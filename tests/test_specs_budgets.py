"""Tests for model specs and budget calculations on Qwen3-4B / H100."""

from dataclasses import replace

import pytest

from quail.planner import budgets
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_32B_FP8


def test_model_weights_and_kv_memory_budgets():
    # Tied output weights must remain available for input embeddings.
    assert QWEN3_4B_FP8.head_mem_bytes == 0.0
    assert QWEN3_4B_FP8.W_resident == QWEN3_4B_FP8.W_mem
    # Unused rows of the separate 32B output weights can be released.
    head = QWEN3_32B_FP8.head_mem_bytes
    assert head == 151_936 * 5_120 * 2
    assert QWEN3_32B_FP8.W_resident == QWEN3_32B_FP8.W_mem - head
    # the freed bytes become arena tokens at the KV rate:
    # 1,555,824,640 bytes / 262,144 bytes per 32B KV token = 5,935
    kept = replace(QWEN3_32B_FP8, tied_head=True)
    grown = budgets.arena_tokens(QWEN3_32B_FP8, H100_SXM)
    assert grown == 112_312
    assert grown - budgets.arena_tokens(kept, H100_SXM) == 5_935
    # the chunk budget does not move: the 32B chunk is capped by the
    # int32 kernel index, not by memory
    assert budgets.chunk_budget(QWEN3_32B_FP8, H100_SXM) == \
        budgets.chunk_budget(kept, H100_SXM)

    tokens = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM)
    assert tokens == 362_250
    assert tokens // 400 == 905
    fp8 = budgets.arena_tokens(QWEN3_4B_FP8.with_kv_bytes(1.0), H100_SXM)
    assert fp8 == pytest.approx(2 * tokens, rel=0.01)
