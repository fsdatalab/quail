"""Checkpoint sizes, the retained pool, and eviction for a hybrid model.

The numbers are the ones in the hybrid-model plan for Qwen3.5-4B on an H100.
"""

from dataclasses import replace

import pytest
from helpers.fakes import bare_arena

from quail.backends.quail.executor.arena import KVArena, PageArena
from quail.backends.quail.retention import policy
from quail.backends.quail.worker import checkpoint_arena_args
from quail.cost import budgets
from quail.cost.retention import RetentionPolicy, coefficients
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_5_4B_BF16

SPEC = QWEN3_5_4B_BF16
CHUNK = 116_508
# the plan's weights are 2 bytes x the config's tensors; the spec carries
# vLLM's measured model memory, 1.6 percent higher
PLAN = replace(SPEC, w_mem_bytes=8_411_152_384)


def test_checkpoint_sizes():
    assert SPEC.conv_channels == 8_192
    assert SPEC.gdn_state_bytes == 2_097_152
    assert SPEC.conv_state_bytes == 49_152
    assert SPEC.checkpoint_bytes == 51_511_296
    assert budgets.checkpoint_pages(SPEC) == 98.25
    assert QWEN3_4B_FP8.checkpoint_bytes == 0
    assert budgets.checkpoint_pages(QWEN3_4B_FP8) == 0


def test_pool_and_chunk_follow_the_plan():
    assert budgets.chunk_budget(PLAN, H100_SXM) == CHUNK
    assert budgets.checkpoint_pass_bytes(PLAN, 1_024, CHUNK) == 4_941 * 2**20
    assert budgets.hybrid_pool_pages(PLAN, H100_SXM, CHUNK) == 68_060


@pytest.mark.parametrize("tokens,slots", [(200, 611), (1_000, 422),
                                          (4_000, 195), (16_000, 61)])
def test_slots_per_document_length(tokens, slots):
    kv_pages, sliding = budgets.arena_pages(PLAN, H100_SXM, CHUNK, tokens)
    assert sliding == 0
    assert budgets.checkpoint_slots(PLAN, H100_SXM, CHUNK, kv_pages) == slots
    # a document's retained cost fits the pool
    cost = -(-tokens // 16) + budgets.checkpoint_pages(SPEC)
    assert slots * cost <= 68_060 < (slots + 1) * cost


def test_plan_example_splits_the_pool():
    kv_pages, _ = budgets.arena_pages(PLAN, H100_SXM, CHUNK, 1_000)
    assert kv_pages in (26_597, 26_598)


def test_models_without_gdn_layers_keep_their_pages():
    assert budgets.checkpoint_slots(QWEN3_4B_FP8, H100_SXM, 1_000, 5) == 0
    assert checkpoint_arena_args(QWEN3_4B_FP8, H100_SXM, 1_000, 5) == {}
    assert "checkpoint_pages" not in coefficients(QWEN3_4B_FP8, H100_SXM)
    assert budgets.arena_pages(QWEN3_4B_FP8, H100_SXM, 10_000)[0] == (
        budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM, 10_000) // 16)


def test_arena_arguments_name_the_state_shapes():
    args = checkpoint_arena_args(PLAN, H100_SXM, CHUNK, 26_598)
    assert args["n_state_slots"] == 422
    assert args["gdn_state_shape"] == (24, 32, 128, 128)
    assert args["conv_state_shape"] == (24, 3, 8_192)


def test_measured_weights_leave_a_slightly_smaller_pool():
    # 134,612,480 bytes more weights than the plan: 0.4 percent fewer pages
    assert budgets.hybrid_pool_pages(SPEC, H100_SXM, CHUNK) == 67_803
    kv_pages, _ = budgets.arena_pages(SPEC, H100_SXM, CHUNK, 200)
    assert budgets.checkpoint_slots(SPEC, H100_SXM, CHUNK, kv_pages) == 609


def test_priority_divides_by_kv_and_checkpoint_pages():
    uses = {"a": (1, 0)}
    plain = RetentionPolicy(1.0, 0.0, uses)
    hybrid = RetentionPolicy(1.0, 0.0, uses, checkpoint_pages=98.25)
    assert plain.priority(("a", 0), 1_000, 63)[0] == 1_000 / 63
    assert hybrid.priority(("a", 0), 1_000, 63)[0] == 1_000 / (63 + 98.25)
    config = coefficients(SPEC, H100_SXM)
    assert policy(config, uses).checkpoint_pages == 98.25


def test_slots_free_with_their_key():
    arena = PageArena(8, 16, n_slots=2)
    arena.alloc("a", 16)
    arena.alloc("b", 16)
    arena.alloc("c", 16)
    first = arena.alloc_slot("a")
    assert arena.alloc_slot("a") == first
    assert arena.alloc_slot("b") is not None
    assert arena.alloc_slot("c") is None
    arena.free_key("a")
    assert arena.alloc_slot("c") == first


def checkpoint_arena(slots, pages=16):
    arena = bare_arena(KVArena.__new__(KVArena), pages)
    arena.accounting = PageArena(pages, 16, n_slots=slots)
    arena.accounting.configure_retention(
        RetentionPolicy(1.0, 0.0, {"a": (1, 0)}, checkpoint_pages=98.25), pages)
    return arena


def keep(arena, key, tokens):
    assert arena.accounting.alloc(key, tokens) is not None
    arena._rows[key] = arena._capacity_rows[key] = None
    arena.checkpoint_slot(key)
    arena.retain(key, tokens)


def test_a_full_pool_evicts_the_lowest_priority_document_with_its_slot():
    arena = checkpoint_arena(slots=2)
    keep(arena, ("a", 0), 32)
    keep(arena, ("a", 1), 64)
    assert arena.has_checkpoint(("a", 0)) and arena.has_checkpoint(("a", 1))
    arena.accounting.alloc(("a", 2), 32)
    arena._rows[("a", 2)] = arena._capacity_rows[("a", 2)] = None
    assert arena.checkpoint_slot(("a", 2)) is not None
    # per page the 32-token document is worth the same as the 64-token one;
    # the tie goes to the earlier document
    assert not arena.is_resident(("a", 0))
    assert arena.is_resident(("a", 1))
    assert arena.evicted_keys == 1
    assert arena.n_state_slots == 2


def test_a_pinned_document_keeps_its_slot():
    arena = checkpoint_arena(slots=1)
    assert arena.accounting.alloc(("a", 0), 32) is not None
    arena._rows[("a", 0)] = arena._capacity_rows[("a", 0)] = None
    assert arena.checkpoint_slot(("a", 0)) is not None
    arena.accounting.pin(("a", 0))
    arena.accounting.alloc(("a", 1), 32)
    assert arena.checkpoint_slot(("a", 1)) is None
    assert arena.is_resident(("a", 0))
