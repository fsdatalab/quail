"""Checkpoint allocation, retention, and memory accounting."""

from types import SimpleNamespace

from helpers.fakes import bare_arena

from quail.backends.quail.executor.arena import KVArena, PageArena
from quail.backends.quail.retention import policy
from quail.cost.retention import RetentionPolicy, coefficients
from quail.specs import H100_SXM, QWEN3_5_4B_BF16

SPEC = QWEN3_5_4B_BF16


def test_priority_divides_by_kv_and_checkpoint_pages():
    uses = {"a": (1, 0)}
    plain = RetentionPolicy(1.0, 0.0, uses)
    hybrid = RetentionPolicy(1.0, 0.0, uses, checkpoint_pages=98.25)
    assert plain.priority(("a", 0), 1_000, 63)[0] == 1_000 / 63
    assert hybrid.priority(("a", 0), 1_000, 63)[0] == 1_000 / (63 + 98.25)
    config = coefficients(SPEC, H100_SXM)
    assert policy(config, uses).checkpoint_pages == 98.25


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
    free_pages = arena.free_pages
    assert not arena.reserve(("a", 1), 32, capacity_tokens=48)
    assert arena.free_pages == free_pages
    assert not arena.is_resident(("a", 1))
    assert arena.is_resident(("a", 0))
    arena.free_key(("a", 0))
    assert arena.reserve(("a", 1), 32, capacity_tokens=48)
    assert arena.has_checkpoint(("a", 1))
    arena.retain(("a", 1), 32)
    slot = arena.checkpoint_slot(("a", 1))
    assert not arena.reserve(("a", 1), 32, capacity_tokens=48, extension=True)
    assert arena.checkpoint_slot(("a", 1)) == slot
    assert arena.is_resident(("a", 1))


def test_checkpoint_slots_are_reused_without_freeing_another_keys_state():
    arena = checkpoint_arena(slots=5)
    first, second = ("a", 0), ("a", 1)
    keep(arena, first, 48)
    keep(arena, second, 48)
    arena.accounting.pin(first)
    arena.accounting.pin(second)
    main = arena.checkpoint_slot(first)
    extension = arena.accounting.alloc_extension(first)
    shared = arena.share_slot(first, 32)
    other = arena.share_slot(second, 16)
    assert len({main, extension, shared, other}) == 4
    assert arena.share_slot(first, 32) == shared
    assert arena.share_slot(first, 16) is None
    assert arena.slots_held([first, second]) == 5

    arena.record_checkpoint(first, 48)
    assert arena.shared_state(first, 48) == main
    assert arena.shared_state(first, 32) == shared
    assert arena.shared_state(first, 16) is None
    arena.hold(first, 1)
    assert arena.free_key(first) == 0
    assert arena.shared_state(first, 32) == shared
    arena.read_state((first,))
    arena.release(first)
    assert arena.shared_state(first, 32) == shared
    arena.done_reading((first,))
    assert arena.accounting.share_slots == {(second, 16): other}
    assert len(arena.accounting.free_slots) == 3
    assert arena.shared_state(first, 48) is None
    reused = arena.share_slot(second, 32)
    assert reused in {main, extension, shared}
    arena.evict_key(second)
    assert len(arena.accounting.free_slots) == 5
    assert not arena.accounting.share_slots


def test_memory_samples_and_events_reset_between_queries():
    arena = checkpoint_arena(slots=4, pages=10)
    arena.sample_memory(tokens=5)
    assert arena.memory_trace == []
    arena.torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda: (30 * 2**30, 80 * 2**30),
        memory_allocated=lambda: 5 * 2**30))
    keep(arena, ("a", 0), 48)
    arena.accounting.pin(("a", 0))
    arena.sample_memory(tokens=1_000, groups=3)
    assert arena.memory_trace[0][1:] == (
        3, 0, 1, 1, 50 * 1024.0, 5 * 1024.0, 1_000, 3, 0)
    arena.log_event("split", groups=2)
    assert arena.events
    arena.reset_stats()
    assert arena.memory_trace == [] and arena.events == []
