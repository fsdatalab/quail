"""Shared-state slots, memory samples, and the borrow rules of a hybrid model."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from fakes import bare_arena

from quail.backends.quail.executor.arena import KVArena, PageArena
from quail.backends.quail.executor.stages import (
    _StageExecutor,
    checkpoint_slot_limit,
    state_share_points,
)


class _Rows(dict):
    """Row tables the tensor-free arena never fills."""

    def pop(self, key, *default):
        return super().pop(key, None)


def arena_with_slots(slots=6, pages=64):
    arena = bare_arena(KVArena.__new__(KVArena), pages)
    arena.accounting = PageArena(pages, 16, n_slots=slots)
    arena._rows = _Rows()
    arena._capacity_rows = _Rows()
    return arena


def resident(arena, key, tokens=48):
    arena.alloc(key, tokens)
    arena.accounting.pin(key)


# ---- the accounting ---------------------------------------------------

def test_share_slot_is_kept_per_key_and_token_count():
    accounting = PageArena(8, 16, n_slots=4)
    accounting.alloc("p", 32)
    first = accounting.alloc_share("p", 16)
    assert accounting.alloc_share("p", 16) == first
    assert accounting.alloc_share("p", 32) != first
    assert len(accounting.free_slots) == 2


def test_share_slot_for_a_key_that_is_not_resident_raises():
    accounting = PageArena(8, 16, n_slots=2)
    with pytest.raises(KeyError):
        accounting.alloc_share("ghost", 16)


def test_share_slot_is_none_when_no_slot_is_free():
    accounting = PageArena(8, 16, n_slots=1)
    accounting.alloc("p", 32)
    assert accounting.alloc_slot("p") is not None
    assert accounting.alloc_share("p", 16) is None


def test_freeing_a_key_frees_main_extension_and_shared_slots():
    accounting = PageArena(8, 16, n_slots=5)
    accounting.alloc("p", 32)
    accounting.alloc_slot("p")
    accounting.alloc_extension("p")
    accounting.alloc_share("p", 16)
    accounting.alloc_share("p", 24)
    assert not accounting.free_slots[:0] and len(accounting.free_slots) == 1
    accounting.free_key("p")
    assert sorted(accounting.free_slots) == [0, 1, 2, 3, 4]
    assert accounting.share_slots == {}


def test_freeing_one_key_keeps_another_keys_shared_slots():
    accounting = PageArena(8, 16, n_slots=4)
    accounting.alloc("p", 32)
    accounting.alloc("q", 32)
    kept = accounting.alloc_share("q", 16)
    accounting.alloc_share("p", 16)
    accounting.free_key("p")
    assert accounting.share_slots == {("q", 16): kept}
    assert len(accounting.free_slots) == 3


# ---- the arena --------------------------------------------------------

def test_shared_state_finds_a_point_or_the_document_end():
    arena = arena_with_slots()
    resident(arena, "p")
    slot = arena.share_slot("p", 32)
    assert arena.shared_state("p", 32) == slot
    assert arena.shared_state("p", 16) is None
    main = arena.checkpoint_slot("p")
    arena.record_checkpoint("p", 48)
    assert arena.shared_state("p", 48) == main
    # the checkpoint serves only the length it covers
    assert arena.shared_state("p", 47) is None
    assert arena.shared_state("ghost", 48) is None


def test_a_share_point_wins_over_the_document_end_at_the_same_length():
    arena = arena_with_slots()
    resident(arena, "p")
    main = arena.checkpoint_slot("p")
    arena.record_checkpoint("p", 32)
    shared = arena.share_slot("p", 32)
    assert arena.shared_state("p", 32) == shared != main


def test_share_slot_evicts_a_retained_key_when_none_is_free():
    arena = arena_with_slots(slots=2)
    old = ("r", 0)
    resident(arena, "p")
    resident(arena, old)
    arena.checkpoint_slot(old)
    arena.record_checkpoint(old, 48)
    arena.checkpoint_slot("p")
    assert not arena.accounting.free_slots
    arena.retain(old, 48, priority=(0,))
    slot = arena.share_slot("p", 16)
    assert slot is not None
    assert not arena.is_resident(old)
    assert arena.evicted_keys == 1


def test_share_slot_is_none_when_every_slot_is_pinned():
    arena = arena_with_slots(slots=2)
    resident(arena, "p")
    resident(arena, "q")
    arena.checkpoint_slot("p")
    arena.checkpoint_slot("q")
    assert arena.share_slot("p", 16) is None


def test_a_held_parent_keeps_its_shared_slots_until_released():
    arena = arena_with_slots()
    parent = ("r", 0)
    resident(arena, parent)
    arena.share_slot(parent, 16)
    arena.hold(parent, 1)
    assert arena.free_key(parent) == 0
    assert (parent, 16) in arena.accounting.share_slots
    arena.release(parent)
    assert arena.accounting.share_slots == {}


def test_evicting_a_retained_parent_frees_its_shared_slots():
    arena = arena_with_slots(slots=3)
    parent = ("r", 0)
    resident(arena, parent)
    arena.share_slot(parent, 16)
    arena.checkpoint_slot(parent)
    arena.retain(parent, 48, priority=(0,))
    arena.evict_key(parent)
    assert len(arena.accounting.free_slots) == 3
    assert arena.events[-1][1] == "evict"
    assert arena.events[-1][2]["tokens"] == 48


# ---- the memory trace -------------------------------------------------

@pytest.fixture
def cuda_stub():
    cuda = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda: (30 * 2**30, 80 * 2**30),
        memory_allocated=lambda: 5 * 2**30)
    return SimpleNamespace(cuda=cuda)


def test_a_sample_records_occupancy_and_device_memory(cuda_stub):
    arena = arena_with_slots(slots=4, pages=10)
    arena.torch = cuda_stub
    resident(arena, "p", tokens=48)
    arena.checkpoint_slot("p")
    arena.sample_memory(tokens=1_000, groups=3)
    row = arena.memory_trace[0]
    assert row[1:] == (3, 0, 1, 1, 50 * 1024.0, 5 * 1024.0, 1_000, 3, 0)


def test_sampling_does_nothing_without_a_gpu():
    arena = arena_with_slots()
    arena.sample_memory(tokens=5)
    assert arena.memory_trace == []
    arena.torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False))
    arena.sample_memory(tokens=5)
    assert arena.memory_trace == []


def test_reset_stats_clears_the_trace_and_the_events(cuda_stub):
    arena = arena_with_slots()
    arena.torch = cuda_stub
    arena.sample_memory()
    arena.log_event("split", groups=2)
    assert arena.memory_trace and arena.events
    arena.reset_stats()
    assert arena.memory_trace == [] and arena.events == []


# ---- the scheduler's rules --------------------------------------------

def test_share_points_group_distinct_counts_by_parent():
    parents = [None, 0, 0, 0, 2, None]
    shared = [0, 32, 48, 32, 16, 0]
    assert state_share_points(parents, shared) == [
        (32, 48), (), (16,), (), (), ()]


def test_share_points_ignore_a_borrow_of_zero_tokens():
    assert state_share_points([None, 0], [0, 0]) == [(), ()]
    assert state_share_points([], []) == []


@pytest.mark.parametrize("slots,stages,shares,limit", [
    (580, 1, False, 580),
    (580, 2, False, 290),
    (580, 1, True, 290),
    (580, 3, True, 193),
    (1, 3, True, 1),
    (0, 1, False, 1),
])
def test_slot_limit_counts_each_slot_a_document_can_hold(
        slots, stages, shares, limit):
    assert checkpoint_slot_limit(slots, stages, shares) == limit


def borrow_checker(parent_start, prefix_length, shared, saved=None,
                   allowed=True, kept=True):
    arena = SimpleNamespace(
        shared_state=lambda key, tokens: (
            None if saved is None else saved.get((key, tokens))))
    executor = SimpleNamespace(
        borrowing=SimpleNamespace(
            tree_shared=[0, shared],
            shared=lambda parent: parent_start),
        prefix_lengths=[100, prefix_length],
        # the parent saves a state at the length its child borrows
        share_points=[(shared,) if kept else (), ()],
        arena=arena, keys=["p", "c"])
    return _StageExecutor._state_borrow_check(
        executor, lambda doc, parent, same_chunk: allowed)


def test_a_borrow_must_start_before_the_borrowers_own_end():
    check = borrow_checker(0, prefix_length=64, shared=64)
    assert not check(1, 0, True)
    assert borrow_checker(0, 65, 64)(1, 0, True)


def test_a_borrow_must_start_after_the_parents_own_start():
    assert not borrow_checker(32, 100, 32)(1, 0, True)
    assert not borrow_checker(48, 100, 32)(1, 0, True)
    assert borrow_checker(32, 100, 48)(1, 0, True)


def test_a_parent_from_an_earlier_chunk_needs_its_saved_state():
    assert not borrow_checker(0, 100, 32)(1, 0, False)
    assert not borrow_checker(0, 100, 32, saved={("p", 16): 3})(1, 0, False)
    assert borrow_checker(0, 100, 32, saved={("p", 32): 3})(1, 0, False)


def test_a_parent_in_the_same_chunk_will_write_its_state():
    assert borrow_checker(0, 100, 32)(1, 0, True)


def test_the_arenas_own_rule_can_still_refuse():
    assert not borrow_checker(0, 100, 32, allowed=False)(1, 0, True)
    assert not borrow_checker(
        0, 100, 32, saved={("p", 32): 3}, allowed=False)(1, 0, False)


@pytest.fixture(autouse=True)
def _no_torch_leak(monkeypatch):
    # sample_memory reads self.torch; keep a stub module out of sys.modules
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))


# ---- the event log ----------------------------------------------------

def event_of(arena, **detail):
    executor = SimpleNamespace(arena=arena)
    _StageExecutor._log_memory_event(executor, "split", groups=4, **detail)


@pytest.mark.parametrize("message,reason", [
    ("no free checkpoint slot for ('r', 3): all 580 belong to documents "
     "in use", "slots"),
    ("unified suffix pages exceed the free KV arena", "pages"),
    ("rows exceed the chunk", "rows"),
])
def test_a_split_is_logged_with_what_ran_out(message, reason):
    arena = arena_with_slots()
    event_of(arena, reason=message)
    when, kind, detail = arena.events[0]
    assert (kind, detail) == ("split", {"groups": 4, "reason": reason})


def test_an_arena_without_a_log_ignores_events():
    event_of(SimpleNamespace(), reason="no free checkpoint slot")


def test_a_borrow_at_a_length_the_parent_did_not_keep_is_refused():
    # a small pool keeps only some share lengths; the other borrowers
    # compute their whole prefix
    assert borrow_checker(0, 100, 32)(1, 0, True)
    assert not borrow_checker(0, 100, 32, kept=False)(1, 0, True)
    assert not borrow_checker(0, 100, 32, kept=False,
                              saved={("p", 32): 3})(1, 0, False)


def test_a_borrow_of_the_parents_whole_prefix_needs_no_kept_point():
    # the parent's document-end checkpoint serves its own length
    assert borrow_checker(0, 200, 100, kept=False)(1, 0, True)
